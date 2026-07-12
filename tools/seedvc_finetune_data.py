#!/usr/bin/env python3
"""Curate raw/noisy long-form audio into Seed-VC few-shot fine-tune clips
(issue #81, built during the #80 PoC).

Three-stage filter, in order:

1. Whisper transcription (30 s windows) + character-diversity heuristic —
   drops long non-verbal repetition-loop stretches (moaning, filler
   sounds, etc.) that Whisper hallucinates into repeated-character
   garbage rather than real words.
2. Silero VAD — keeps only segments a real trained speech detector
   accepts, rejecting breath/mouth-noise artifacts a cruder RMS+spectral-
   flatness heuristic let through in earlier iterations of this script.
3. pyin voiced-ratio filter — true whisper (the vocal register, not the
   ASR model) has no F0 periodicity even on vowels, so this is a direct
   physical test for "is this actually whispered" as opposed to
   normally-voiced speech.

Then: merge across natural pauses, concatenate, re-split into <=30 s
clips at silence boundaries for the official train.py's per-file
duration bound (data/ft_dataset.py, issue #80).

Requires (beyond tools/requirements.txt): transformers, librosa,
soundfile. Silero VAD downloads via torch.hub on first run.

```sh
pip install -r tools/requirements.txt transformers librosa soundfile
python3 tools/seedvc_finetune_data.py --input long_recording.mp3 --output-dir clips/
```
"""

import argparse
import re
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf
import torch


def transcribe_chunks(path: str, chunk_s: int = 30):
    """Whisper transcript per chunk_s window; returns [(start_s, text), ...]."""
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch.cuda.is_available() else -1,
        generate_kwargs={"language": "japanese", "task": "transcribe"},
    )
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    )
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    sr = 16000
    n_chunks = len(audio) // (sr * chunk_s) + 1
    results = []
    for i in range(n_chunks):
        s, e = i * sr * chunk_s, min(len(audio), (i + 1) * sr * chunk_s)
        seg = audio[s:e]
        if len(seg) < sr:
            continue
        out = asr({"array": seg, "sampling_rate": sr})
        results.append((i * chunk_s, out["text"].strip()))
    return results


def char_diversity(text: str) -> float:
    if len(text) < 3:
        return 0.0
    return len(set(text)) / len(text)


def talk_like_ranges(transcript, chunk_s: int, diversity_threshold: float, max_chars: int = 250):
    """Merge contiguous non-loop chunks into candidate (start, end) ranges."""
    ranges = []
    cur_start = None
    for t, text in transcript:
        is_talk = char_diversity(text) > diversity_threshold and len(text) < max_chars
        if is_talk and cur_start is None:
            cur_start = t
        if not is_talk and cur_start is not None:
            ranges.append((cur_start, t))
            cur_start = None
    if cur_start is not None:
        ranges.append((cur_start, transcript[-1][0] + chunk_s))
    return ranges


def vad_segments(wav16: np.ndarray, sr: int, threshold: float):
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
    )
    get_speech_timestamps = utils[0]
    ts = get_speech_timestamps(
        torch.from_numpy(wav16),
        model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=250,
        speech_pad_ms=100,
    )
    return [(s["start"] / sr, s["end"] / sr) for s in ts]


def voiced_ratio(seg: np.ndarray, sr: int) -> float:
    f0, voiced_flag, voiced_prob = librosa.pyin(
        seg, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=sr, frame_length=1024, hop_length=256,
    )
    if not len(voiced_flag):
        return 0.0
    return float(np.nanmean(voiced_flag.astype(float)))


def merge_ranges(ranges, merge_gap_s: float, min_dur_s: float):
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for t0, t1 in ranges[1:]:
        if t0 - merged[-1][1] < merge_gap_s:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    return [(a, b) for a, b in merged if b - a >= min_dur_s]


def silence_split(wav: np.ndarray, sr: int, max_clip_s: float, min_silence_s: float = 0.15,
                   noise_db: str = "-30dB") -> list:
    """Split a concatenated wav into <=max_clip_s pieces at silence midpoints."""
    tmp = "/tmp/_seedvc_finetune_data_concat.wav"
    sf.write(tmp, wav, sr)
    p = subprocess.run(
        ["ffmpeg", "-i", tmp, "-af", f"silencedetect=noise={noise_db}:d={min_silence_s}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
    mids = [(s + e) / 2 for s, e in zip(starts, ends) if e > s]

    total = len(wav) / sr
    cuts, last = [0.0], 0.0
    while last < total - 2:
        cands = [m for m in mids if last + 5 < m <= last + max_clip_s]
        cut = cands[-1] if cands else min(last + max_clip_s, total)
        cuts.append(cut)
        last = cut
    if cuts[-1] < total:
        cuts.append(total)
    cuts = sorted(set(round(c, 2) for c in cuts))

    clips = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 2:
            clips.append(wav[int(a * sr):int(b * sr)])
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source audio file (any ffmpeg-readable format)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sr", type=int, default=22050, help="output clip sample rate (matches seed-vc training)")
    ap.add_argument("--diversity-threshold", type=float, default=0.35,
                     help="stage 1: min char-diversity ratio to count as talk, not a repetition loop")
    ap.add_argument("--vad-threshold", type=float, default=0.75, help="stage 2: Silero VAD speech-probability cutoff")
    ap.add_argument("--voiced-ratio-threshold", type=float, default=0.35,
                     help="stage 3: min pyin voiced-frame ratio to reject whispered-register segments")
    ap.add_argument("--merge-gap-s", type=float, default=1.0, help="bridge pauses shorter than this across stages")
    ap.add_argument("--max-clip-s", type=float, default=30.0, help="official train.py's per-file duration bound")
    args = ap.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/3] Whisper transcription + repetition-loop filter...", file=sys.stderr)
    transcript = transcribe_chunks(args.input)
    ranges = talk_like_ranges(transcript, chunk_s=30, diversity_threshold=args.diversity_threshold)
    talk_dur = sum(e - s for s, e in ranges)
    print(f"  {len(ranges)} candidate ranges, {talk_dur/60:.1f}min", file=sys.stderr)

    wav16_full, _ = librosa.load(args.input, sr=16000, mono=True)
    wav_out_full, _ = librosa.load(args.input, sr=args.sr, mono=True)

    print("[2/3] Silero VAD...", file=sys.stderr)
    vad_all = []
    for s, e in ranges:
        si, ei = int(s * 16000), int(e * 16000)
        segs = vad_segments(wav16_full[si:ei], 16000, args.vad_threshold)
        vad_all.extend((s + a, s + b) for a, b in segs)
    vad_dur = sum(b - a for a, b in vad_all)
    print(f"  {len(vad_all)} VAD segments, {vad_dur/60:.1f}min", file=sys.stderr)

    print("[3/3] pyin voiced-ratio filter (rejects whispered register)...", file=sys.stderr)
    kept = []
    for s, e in vad_all:
        si, ei = int(s * args.sr), int(e * args.sr)
        seg = wav_out_full[si:ei]
        if len(seg) < 1024:
            continue
        if voiced_ratio(seg, args.sr) >= args.voiced_ratio_threshold:
            kept.append((s, e))
    kept_dur = sum(b - a for a, b in kept)
    print(f"  {len(kept)} segments pass, {kept_dur/60:.1f}min", file=sys.stderr)

    merged = merge_ranges(kept, args.merge_gap_s, min_dur_s=1.0)
    pieces = []
    for a, b in merged:
        pieces.append(wav_out_full[int(a * args.sr):int(b * args.sr)])
        pieces.append(np.zeros(int(args.sr * 0.2), dtype=np.float32))
    concat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    clips = silence_split(concat, args.sr, args.max_clip_s)
    for i, clip in enumerate(clips):
        sf.write(f"{args.output_dir}/clip_{i:03d}.wav", clip, args.sr, subtype="PCM_16")

    total = sum(len(c) for c in clips) / args.sr
    print(f"done: {len(clips)} clips, {total/60:.1f}min total, written to {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
