//! `babiniku-fetch` (issue #65): downloads the official checkpoints
//! from Hugging Face and converts them to the fp32 safetensors the
//! engines load — pure Rust, no Python required, per the CLAUDE.md
//! tooling policy ("anything a user runs is Rust").
//!
//! Golden/fixture generators are NOT this tool's job: those stay
//! Python by design (they must run the official implementations).
//!
//! ```sh
//! babiniku-fetch <seedvc|vevo> [--ckpt-dir <dir>] [--yes]
//! ```
//!
//! Nested-checkpoint reading relies on the candle fork's recursive
//! pickle traversal (`read_pth_tensor_info` descends `{net: {cfm: …}}`
//! with dotted keys).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut cmd: Option<String> = None;
    let mut ckpt_dir: Option<PathBuf> = None;
    let mut yes = false;
    let mut from_pth: Option<PathBuf> = None;
    let mut out_prefix = "seedvc".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ckpt-dir" => {
                ckpt_dir = Some(PathBuf::from(args.next().context("--ckpt-dir <dir>")?))
            }
            "--yes" | "-y" => yes = true,
            // seedvc only: convert a *locally* fine-tuned checkpoint
            // (official train.py's ft_model.pth) instead of downloading
            // the base weights from HF — see fetch_seedvc's doc comment.
            "--from-pth" => {
                from_pth = Some(PathBuf::from(args.next().context("--from-pth <ft_model.pth>")?))
            }
            "--out-prefix" => out_prefix = args.next().context("--out-prefix <name>")?,
            "--help" | "-h" => {
                println!(
                    "usage: babiniku-fetch <seedvc|vevo> [--ckpt-dir <dir>] [--yes]\n       babiniku-fetch seedvc --from-pth <ft_model.pth> [--out-prefix <name>] [--ckpt-dir <dir>] [--yes]"
                );
                return Ok(());
            }
            c if cmd.is_none() && !c.starts_with('-') => cmd = Some(c.to_string()),
            other => bail!("unknown argument {other:?} (try --help)"),
        }
    }
    let ckpt = match ckpt_dir {
        Some(d) => d,
        None => default_ckpt_dir()?,
    };
    std::fs::create_dir_all(&ckpt).with_context(|| format!("cannot create {}", ckpt.display()))?;

    match cmd.as_deref() {
        Some("seedvc") => fetch_seedvc(&ckpt, yes, from_pth.as_deref(), &out_prefix),
        Some("vevo") => fetch_vevo(&ckpt, yes),
        Some(other) => {
            bail!("unknown engine {other:?} — supported: seedvc, vevo (meanvc/xvc: #65)")
        }
        None => bail!("usage: babiniku-fetch <seedvc|vevo> [--ckpt-dir <dir>] [--yes]"),
    }
}

/// Mirror of the demo's resolution: `./ckpt` in a checkout, else the
/// platform data directory.
fn default_ckpt_dir() -> Result<PathBuf> {
    let local = PathBuf::from("ckpt");
    if local.is_dir() {
        return Ok(local);
    }
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    Ok(base
        .context("cannot determine a data directory")?
        .join("babiniku/ckpt"))
}

fn confirm_gpl(yes: bool) -> Result<()> {
    eprintln!("Seed-VC weights and the seedvc engine crate are GPL-3.0.");
    eprintln!("Downloading is fine for local use; distributing builds that");
    eprintln!("include them carries GPL obligations (see crates/seedvc).");
    if yes {
        return Ok(());
    }
    eprint!("continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    if !matches!(line.trim(), "y" | "Y" | "yes") {
        bail!("aborted");
    }
    Ok(())
}

/// `crates/vevo` is MIT OR Apache-2.0 (Amphion's code is MIT — no
/// GPL-style crate gate needed), but the released **weights** are
/// CC-BY-NC-4.0. This prompt is the enforcement point: local/personal
/// use is unaffected, but distributing anything built from these
/// weights carries the NonCommercial restriction (see crates/vevo and
/// docs/vevo.md).
fn confirm_nc(yes: bool) -> Result<()> {
    eprintln!("Vevo (Amphion) weights are CC-BY-NC-4.0. The vevo crate's code");
    eprintln!("is MIT OR Apache-2.0, but these specific weights — and anything");
    eprintln!("you produce with them — are licensed for non-commercial use only.");
    if yes {
        return Ok(());
    }
    eprint!("continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    if !matches!(line.trim(), "y" | "Y" | "yes") {
        bail!("aborted");
    }
    Ok(())
}

/// Plain HTTPS download for checkpoints not mirrored on Hugging Face
/// (torchaudio's model hub, a file checked into a GitHub repo).
fn download(url: &str) -> Result<Vec<u8>> {
    eprintln!("fetching {url} …");
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

/// Reads a 1-D little-endian float32 `.npy` array (the only shape/dtype
/// this tool needs): magic, version, header dict, then raw data.
fn read_npy_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if &bytes[..6] != b"\x93NUMPY" {
        bail!("not an .npy file");
    }
    let major = bytes[6];
    let (header_len, header_start) = if major >= 2 {
        (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12)
    } else {
        (u16::from_le_bytes(bytes[8..10].try_into()?) as usize, 10)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])?;
    if !header.contains("'<f4'") {
        bail!("expected '<f4' dtype, got header {header:?}");
    }
    let data = &bytes[header_start + header_len..];
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Reads `mean.npy`/`std.npy` out of an `.npz` (a plain zip archive).
fn read_npz_mean_std(bytes: &[u8]) -> Result<(Vec<f32>, Vec<f32>)> {
    let reader = std::io::Cursor::new(bytes);
    let mut zip = zip::ZipArchive::new(reader)?;
    let read_member =
        |zip: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>, name: &str| -> Result<Vec<f32>> {
            let mut f = zip
                .by_name(name)
                .with_context(|| format!("{name} missing from npz"))?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            read_npy_f32(&buf)
        };
    let mean = read_member(&mut zip, "mean.npy")?;
    let std = read_member(&mut zip, "std.npy")?;
    Ok((mean, std))
}

/// Folds a weight-normalized tensor where `dim` is the **kept**
/// dimension (magnitude `g` varies along `dim`, the L2 norm is taken
/// over every other dimension) — the convention
/// `nn.utils.parametrizations.weight_norm(conv, dim=2)` uses for
/// torchaudio's HuBERT positional-conv embedding. This differs from
/// [`fold_weight_norm`]'s dim-0 convention (BigVGAN, RepCodec).
fn fold_weight_norm_dim(v: &Tensor, g: &Tensor, dim: usize) -> Result<Tensor> {
    let other_dims: Vec<usize> = (0..v.dims().len()).filter(|&d| d != dim).collect();
    let mut norm = v.sqr()?;
    for d in other_dims {
        norm = norm.sum_keepdim(d)?;
    }
    let norm = norm.sqrt()?;
    Ok(v.broadcast_mul(&g.broadcast_div(&norm)?)?)
}

fn hf_file(repo: &str, file: &str) -> Result<PathBuf> {
    eprintln!("fetching {repo} :: {file} …");
    let api = hf_hub::api::sync::Api::new()?;
    Ok(api.model(repo.to_string()).get(file)?)
}

/// Reads a `.pth`/`.bin`/`.pt` into name → fp32 tensor, keeping only
/// keys under `prefix` (stripped) when given.
fn read_pth(path: &Path, prefix: Option<&str>) -> Result<HashMap<String, Tensor>> {
    let all = candle_core::pickle::read_all(path)?;
    let mut out = HashMap::new();
    for (name, t) in all {
        let name = match prefix {
            Some(p) => match name.strip_prefix(p) {
                Some(rest) => rest.to_string(),
                None => continue,
            },
            None => name,
        };
        out.insert(name, t.to_dtype(DType::F32)?);
    }
    if out.is_empty() {
        bail!("no tensors under prefix {prefix:?} in {}", path.display());
    }
    Ok(out)
}

/// Folds `weight_g`/`weight_v` pairs into plain `weight` tensors
/// (`torch.nn.utils.remove_weight_norm`): `w = v · g / ‖v‖₂` with the
/// norm over all dims except 0.
fn fold_weight_norm(sd: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    let mut done: Vec<String> = Vec::new();
    for (k, g) in sd.iter() {
        let Some(base) = k.strip_suffix("weight_g") else {
            continue;
        };
        let v = sd
            .get(&format!("{base}weight_v"))
            .with_context(|| format!("missing weight_v for {k}"))?;
        let dims: Vec<usize> = (1..v.dims().len()).collect();
        let mut norm = v.sqr()?;
        for d in dims {
            norm = norm.sum_keepdim(d)?;
        }
        let norm = norm.sqrt()?;
        let w = v.broadcast_mul(&g.broadcast_div(&norm)?)?;
        out.insert(format!("{base}weight"), w);
        done.push(k.clone());
        done.push(format!("{base}weight_v"));
    }
    for (k, t) in sd {
        if !done.contains(&k) {
            out.insert(k, t);
        }
    }
    Ok(out)
}

fn save(map: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    candle_core::safetensors::save(map, path)?;
    eprintln!("wrote {} ({} tensors)", path.display(), map.len());
    Ok(())
}

fn fetch_seedvc(ckpt: &Path, yes: bool, from_pth: Option<&Path>, out_prefix: &str) -> Result<()> {
    confirm_gpl(yes)?;
    let dev = Device::Cpu;
    let _ = dev;

    // Main checkpoint: DiT (cfm) + length regulator. The checkpoint's
    // style_encoder is dead weight; the real speaker encoder is the
    // standalone funasr CAM++ (see docs/seedvc.md).
    //
    // `--from-pth` swaps in a *locally* fine-tuned checkpoint (the
    // official `train.py`'s `ft_model.pth` — `state = {'net': {key:
    // model[key].state_dict() for key in model}}`, i.e. the exact same
    // `net.cfm.`/`net.length_regulator.` shape the HF release has,
    // since fine-tuning only ever touches these two submodules — the
    // frozen encoders/vocoder below are unaffected and still come from
    // HF either way). `--out-prefix` avoids clobbering the base
    // `seedvc_*` files when converting a fine-tune alongside them.
    let main = match from_pth {
        Some(p) => p.to_path_buf(),
        None => hf_file(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        )?,
    };
    // The HF release's cfm/length_regulator were saved from a DDP-
    // wrapped model, which PyTorch auto-prefixes every key with
    // "module." — our loaders (dit.rs's `vb.pp("module.estimator")`,
    // regulator.rs's `vb.pp("module")`) hardcode that prefix. A
    // single-GPU `train.py` fine-tune has no DDP wrapper, so its state
    // dict lacks it; add it back so the same loaders work for both
    // without touching the already golden-tested load paths.
    let add_module_prefix = |sd: HashMap<String, Tensor>| -> HashMap<String, Tensor> {
        if from_pth.is_some() && !sd.keys().any(|k| k.starts_with("module.")) {
            sd.into_iter().map(|(k, v)| (format!("module.{k}"), v)).collect()
        } else {
            sd
        }
    };
    save(
        &add_module_prefix(read_pth(&main, Some("net.cfm."))?),
        &ckpt.join(format!("{out_prefix}_dit.safetensors")),
    )?;
    save(
        &add_module_prefix(read_pth(&main, Some("net.length_regulator."))?),
        &ckpt.join(format!("{out_prefix}_regulator.safetensors")),
    )?;
    if from_pth.is_some() {
        // Fine-tuning never touches these; reuse the base files so the
        // engine has a complete checkpoint set under out_prefix too.
        for (suffix, base_name) in [
            ("campplus", "seedvc_campplus"),
            ("bigvgan", "seedvc_bigvgan"),
            ("whisper", "seedvc_whisper"),
        ] {
            let src = ckpt.join(format!("{base_name}.safetensors"));
            let dst = ckpt.join(format!("{out_prefix}_{suffix}.safetensors"));
            if src.exists() && !dst.exists() {
                std::fs::copy(&src, &dst)
                    .with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
            }
        }
        eprintln!("{out_prefix} checkpoints ready under {}", ckpt.display());
        return Ok(());
    }

    // CAM++ speaker encoder (flat state dict).
    let camp = hf_file("funasr/campplus", "campplus_cn_common.bin")?;
    save(
        &read_pth(&camp, None)?,
        &ckpt.join("seedvc_campplus.safetensors"),
    )?;

    // BigVGAN v2 vocoder (weight norm folded like remove_weight_norm()).
    let bv = hf_file(
        "nvidia/bigvgan_v2_22khz_80band_256x",
        "bigvgan_generator.pt",
    )?;
    let bv_sd = read_pth(&bv, Some("generator.")).or_else(|_| read_pth(&bv, None))?;
    save(
        &fold_weight_norm(bv_sd)?,
        &ckpt.join("seedvc_bigvgan.safetensors"),
    )?;

    // Whisper-small encoder (already safetensors upstream; subset).
    let wh = hf_file("openai/whisper-small", "model.safetensors")?;
    let all = candle_core::safetensors::load(&wh, &Device::Cpu)?;
    let enc: HashMap<String, Tensor> = all
        .into_iter()
        .filter(|(k, _)| k.starts_with("model.encoder."))
        .map(|(k, t)| Ok((k, t.to_dtype(DType::F32)?)))
        .collect::<Result<_>>()?;
    save(&enc, &ckpt.join("seedvc_whisper.safetensors"))?;

    eprintln!("seedvc checkpoints ready under {}", ckpt.display());
    Ok(())
}

/// HF safetensors repos hold multiple checkpoints under subfolders
/// (`tokenizer/vq8192/model.safetensors`, etc); loads one, keeping
/// only keys under `prefix` (stripped) when given, dropping the rest.
fn read_safetensors_prefix(
    path: &Path,
    prefix: Option<&str>,
    exclude_prefix: Option<&str>,
) -> Result<HashMap<String, Tensor>> {
    let all = candle_core::safetensors::load(path, &Device::Cpu)?;
    let mut out = HashMap::new();
    for (name, t) in all {
        if let Some(ex) = exclude_prefix {
            if name.starts_with(ex) {
                continue;
            }
        }
        let name = match prefix {
            Some(p) => match name.strip_prefix(p) {
                Some(rest) => rest.to_string(),
                None => continue,
            },
            None => name,
        };
        out.insert(name, t.to_dtype(DType::F32)?);
    }
    Ok(out)
}

fn fetch_vevo(ckpt: &Path, yes: bool) -> Result<()> {
    confirm_nc(yes)?;

    // ---- HuBERT-large layer-18 extractor (torchaudio's hub, NOT on HF) ----
    let bytes =
        download("https://download.pytorch.org/torchaudio/models/hubert_fairseq_large_ll60k.pth")?;
    let tmp = ckpt.join("_dl_hubert.pth");
    std::fs::write(&tmp, &bytes)?;
    // Unlike the Python-side `state_dict()` (wrapped by torchaudio's
    // `_Wav2Vec2Model`, prefixing every key with "model."), the raw
    // checkpoint on torch hub is the *inner* `Wav2Vec2Model`'s state
    // dict directly — no prefix to strip.
    let mut raw = read_pth(&tmp, None)?;
    std::fs::remove_file(&tmp).ok();

    // pos_conv_embed weight-norm uses dim=2 (magnitude per kernel
    // position), unlike every other weight-normed conv in this
    // codebase — fold before filtering the rest.
    let g = raw
        .remove("encoder.transformer.pos_conv_embed.conv.weight_g")
        .context("missing pos_conv g")?;
    let v = raw
        .remove("encoder.transformer.pos_conv_embed.conv.weight_v")
        .context("missing pos_conv v")?;
    raw.insert(
        "encoder.transformer.pos_conv_embed.conv.weight".to_string(),
        fold_weight_norm_dim(&v, &g, 2)?,
    );

    const NUM_LAYERS: usize = 18; // Vevo only ever reads layer 18 of 24.
    let mut hubert = HashMap::new();
    for (k, v) in raw {
        if let Some(rest) = k.strip_prefix("encoder.transformer.layers.") {
            let idx: usize = rest.split('.').next().unwrap().parse()?;
            if idx >= NUM_LAYERS {
                continue;
            }
        }
        // torchaudio's Transformer-level `layer_norm` is never applied
        // on the extract_features()/get_intermediate_outputs() path
        // HUBERT_LARGE uses (layer_norm_first=False at the Transformer
        // level, distinct from each EncoderLayer's own True) — dead
        // weight for this port, see crates/vevo/src/hubert.rs.
        if k == "encoder.transformer.layer_norm.weight"
            || k == "encoder.transformer.layer_norm.bias"
        {
            continue;
        }
        hubert.insert(k, v);
    }
    save(&hubert, &ckpt.join("vevo_hubert.safetensors"))?;

    // ---- HuBERT layer-18 z-norm stats (checked into the Amphion repo) ----
    let npz = download("https://raw.githubusercontent.com/open-mmlab/Amphion/main/models/vc/vevo/config/hubert_large_l18_mean_std.npz")?;
    let (mean, std) = read_npz_mean_std(&npz)?;
    let dev = Device::Cpu;
    let stats = HashMap::from([
        ("mean".to_string(), Tensor::from_vec(mean, 1024, &dev)?),
        ("std".to_string(), Tensor::from_vec(std, 1024, &dev)?),
    ]);
    save(&stats, &ckpt.join("vevo_hubert_stats.safetensors"))?;

    // ---- RepCodec fvq8192 content-style tokenizer (encoder + VQ only —
    // the decoder half is dead weight for inference_fm, see crates/vevo). ----
    let rc = hf_file("amphion/Vevo", "tokenizer/vq8192/model.safetensors")?;
    let mut repcodec = read_safetensors_prefix(&rc, None, Some("decoder."))?;
    let vq_g = repcodec
        .remove("quantizer.quantizers.0.in_project.weight_g")
        .context("missing repcodec in_project.weight_g")?;
    let vq_v = repcodec
        .remove("quantizer.quantizers.0.in_project.weight_v")
        .context("missing repcodec in_project.weight_v")?;
    let vq_b = repcodec
        .remove("quantizer.quantizers.0.in_project.bias")
        .context("missing repcodec in_project.bias")?;
    let vq_cb = repcodec
        .remove("quantizer.quantizers.0.codebook.weight")
        .context("missing repcodec codebook")?;
    repcodec.retain(|k, _| !k.starts_with("quantizer.quantizers.0.") && !k.starts_with("decoder."));
    let folded = fold_weight_norm_dim(&vq_v, &vq_g, 0)?.squeeze(2)?; // [8, 1024, 1] -> [8, 1024]
    repcodec.insert("quantizer.in_project.weight".to_string(), folded);
    repcodec.insert("quantizer.in_project.bias".to_string(), vq_b);
    repcodec.insert("quantizer.codebook.weight".to_string(), vq_cb);
    save(&repcodec, &ckpt.join("vevo_repcodec.safetensors"))?;

    // ---- FM DiffLlama converter (plain fp32, no folding needed) ----
    let fm = hf_file(
        "amphion/Vevo",
        "acoustic_modeling/Vq8192ToMels/model.safetensors",
    )?;
    save(
        &read_safetensors_prefix(&fm, None, None)?,
        &ckpt.join("vevo_fmt.safetensors"),
    )?;

    // ---- Vocos vocoder (plain fp32, no folding needed) ----
    let voc = hf_file(
        "amphion/Vevo",
        "acoustic_modeling/Vocoder/model.safetensors",
    )?;
    save(
        &read_safetensors_prefix(&voc, None, None)?,
        &ckpt.join("vevo_vocos.safetensors"),
    )?;

    eprintln!("vevo checkpoints ready under {}", ckpt.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_fold_matches_torch_semantics() {
        // w = v * g / ||v||_2 with the norm over dims 1.. (torch
        // remove_weight_norm); checked against a hand computation.
        let dev = Device::Cpu;
        let v = Tensor::from_vec(vec![3f32, 4.0, 0.0, 5.0], (2, 2, 1), &dev).unwrap();
        let g = Tensor::from_vec(vec![10f32, 1.0], (2, 1, 1), &dev).unwrap();
        let mut sd = HashMap::new();
        sd.insert("conv.weight_v".to_string(), v);
        sd.insert("conv.weight_g".to_string(), g);
        sd.insert(
            "conv.bias".to_string(),
            Tensor::zeros(2, DType::F32, &dev).unwrap(),
        );
        let out = fold_weight_norm(sd).unwrap();
        assert!(out.contains_key("conv.weight"));
        assert!(out.contains_key("conv.bias"));
        assert!(!out.contains_key("conv.weight_v"));
        let w: Vec<f32> = out["conv.weight"].flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: ||(3,4)|| = 5, g=10 → (6, 8). Row 1: ||(0,5)|| = 5, g=1 → (0, 1).
        let want = [6.0f32, 8.0, 0.0, 1.0];
        for (a, b) in w.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn weight_norm_fold_dim_matches_hubert_pos_conv_semantics() {
        // dim=2 (kept dim): g varies along dim 2 only, norm is taken
        // over dims (0, 1) for each dim-2 slice independently — the
        // convention torchaudio's HuBERT positional conv embedding
        // uses (`weight_norm(conv, dim=2)`), distinct from
        // fold_weight_norm's dim-0 convention.
        let dev = Device::Cpu;
        // v: [2, 2, 2], two "kernel positions" (last dim) each with a
        // 2x2 block. Position 0: (3,4,0,0) -> norm 5. Position 1:
        // (0,0,6,8) -> norm 10.
        let v = Tensor::from_vec(
            vec![3f32, 0.0, 4.0, 0.0, 0.0, 6.0, 0.0, 8.0],
            (2, 2, 2),
            &dev,
        )
        .unwrap();
        let g = Tensor::from_vec(vec![10f32, 1.0], (1, 1, 2), &dev).unwrap();
        let w = fold_weight_norm_dim(&v, &g, 2).unwrap();
        let got: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        // position 0 scaled by 10/5=2, position 1 scaled by 1/10=0.1.
        let want = [6.0f32, 0.0, 8.0, 0.0, 0.0, 0.6, 0.0, 0.8];
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn npy_f32_roundtrip() {
        // Minimal handwritten .npy: magic + v1.0 header + 3 f32s.
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }";
        let mut padded = header.to_string();
        // npy pads the header so data starts at a 64-byte boundary,
        // total-header-including-padding ending in '\n'.
        let prefix_len = 10; // magic(6) + version(2) + header_len(2)
        while (prefix_len + padded.len() + 1) % 64 != 0 {
            padded.push(' ');
        }
        padded.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        bytes.extend_from_slice(padded.as_bytes());
        for v in [1.0f32, -2.5, 3.25] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let got = read_npy_f32(&bytes).unwrap();
        assert_eq!(got, vec![1.0, -2.5, 3.25]);
    }

    #[test]
    fn default_ckpt_dir_prefers_checkout() {
        // Run from the workspace root in CI/dev: ./ckpt may or may not
        // exist, but the function must return SOME sensible path.
        let d = default_ckpt_dir().unwrap();
        assert!(d.to_string_lossy().contains("ckpt"));
    }
}
