# CLAUDE.md

Guidance for AI assistants (and humans) working in the **babiniku.rs** repository (formerly meanvc2.rs).

> **Read [`README.md`](README.md) and the relevant issue first — every task
> starts there (rules 4 & 6).** Deviations from the paper and the remaining
> work are tracked as GitHub issues (#1–#6 at the time of writing).

## Project summary

babiniku.rs is a real-time voice-conversion toolkit; its first engine is an unofficial Rust implementation of **MeanVC 2: Robust
Low-Latency Streaming Zero-Shot Voice Conversion**
([arXiv:2606.09050](https://arxiv.org/abs/2606.09050)).

- **Language / stack:** pure Rust.
- **GPU: driver-only — do NOT introduce a dependency on the full CUDA
  Toolkit.** CPU real-time is the baseline and a core feature; GPU stays an
  opt-in cargo feature (`cuda`/`metal`) that must never be required for
  build, tests, or the demo. Keep setup low-friction.
- **ML runtime:** the [`m96-chan/candle`](https://github.com/m96-chan/candle)
  fork of candle, as a **git dependency** in `Cargo.toml` (not a submodule) —
  kept in sync with upstream; patch only when we hit a bug or a slow path.
- **Scope:** the trainable core of the paper — UTTE, the FRC-scheduled DiT
  decoder, the mean-flows objective/sampler, and the streaming chunk driver.
  The frozen external models (Fast-U2++ BNF extractor, ECAPA-TDNN speaker
  encoder, Vocos vocoder) are abstracted as traits in
  `crates/vc-core/src/encoders.rs`; concrete backends are tracked in issue #4.
- **Fidelity to the paper is the priority:** where the paper is silent we
  follow common DiT/flow-matching practice, but every known deviation must be
  documented (module docs + README "Project status") and tracked as an issue.
- **GPU:** optional (`--features cuda` / `--features metal`); CPU is the
  baseline target — the paper's headline result is single-core CPU streaming.

## Development rules (must follow)

These are hard rules. Do not skip them.

### 1. Develop with TDD
Practice test-driven development. Write a failing test first, make it pass with
the simplest change, then refactor (red → green → refactor). New behavior should
be accompanied by tests; do not add functionality without a test that covers it.

### 2. Always run a demo before pushing
*(“Demo” here means demonstrating your change — it is unrelated to the
TUI app, which is the `babiniku` binary in `crates/babiniku`; the app
shed its “demo” name in #67.)*
**Never push without first demonstrating the change actually works.** Running the
test suite is necessary but not sufficient — exercise the real behavior (run the
relevant example / component and observe it) before every `git push`. If a
change cannot be demoed for some reason, say so explicitly instead of pushing
silently. See [Demo before push](#demo-before-push) for the procedure.

### 3. Treat the documentation on GitHub as the source of truth
For any API, library, or tool manual, consult the **documentation published on
GitHub** as the most up-to-date reference. Do not rely on memory or training
data for API details — verify against the current upstream docs. This matters
especially for the candle fork and other fast-moving dependencies. For model
behavior, the paper (arXiv:2606.09050) is the source of truth.

### 4. Read `README.md` first
**Before starting any task, read [`README.md`](README.md).** It is the canonical
overview of the project's purpose, architecture, and current direction. Ground
your work in it before touching code or docs.

### 5. Update `README.md` when you finish
**When an implementation is done, update [`README.md`](README.md)** so it stays
accurate — features, status checkboxes, usage, and any changed architecture or
commands. A change is not "done" until the README reflects it. Do this before
pushing (it is part of the demo/push checklist below).

### 6. Read the relevant issue before implementing
**Before writing any code, read the GitHub issue(s) covering the work**, plus any
linked/related issues. Understand the scope, task list, acceptance criteria, and
dependencies first. If no issue exists for the work, create one (or ask) before
implementing.

### 7. Update the issue to mark the work complete
**The issue is the definition of done.** When the work is finished, update the
relevant issue — check off completed tasks, note what was implemented/decided,
and close it (or mark it complete). Implementation is **not** "done" until the
issue is updated. Do this before/at push time.

### 8. Ask instead of guessing when there's ambiguity or a design gap
**If multiple interpretations/approaches are plausible, or the issue/design is
incomplete or inconsistent, stop and ask — do not guess and implement.** Don't
silently pick one path or paper over a design deficiency. Surface the ambiguity,
lay out the options with a recommendation, and get a decision before proceeding.
When the answer matters, record it in the issue/docs so it isn't re-litigated.
For gaps in the *paper* itself, follow common practice, but document the
assumption and track it in an issue (see #2, #5).

## Demo before push

A "demo" means exercising the real behavior of your change and observing the
result — not just a green test suite.

The canonical demo is the streaming vertical slice (synthetic BNFs →
UTTE → FRC-DiT → 1-NFE mel chunks):

```bash
cargo run --release --example streaming_demo
```

Confirm it emits every chunk, the first packet arrives after the expected
look-ahead, and the reported per-chunk latency / RTF has not regressed. As
components land (Vocos backend, real BNF extractors, training), extend the
demo and update this command.

When you push, briefly record **what you ran and what you observed** in the
commit message or PR description so the demo is auditable.

## Working with the codebase

### Build & test

```bash
cargo build
cargo test
cargo clippy --all-targets
```

### Before pushing — checklist
- [ ] `README.md` read at the start of the task (rule 4)
- [ ] Relevant issue(s) read before implementing (rule 6)
- [ ] Ambiguities / design gaps raised and resolved, not guessed (rule 8)
- [ ] Tests written first / updated (TDD)
- [ ] `cargo test` passes
- [ ] Change demoed against real behavior, and what you ran/observed is recorded (rule 2)
- [ ] Any API usage verified against current GitHub docs (rule 3)
- [ ] `README.md` updated to reflect the change (rule 5)
- [ ] Issue updated / tasks checked off / closed (rule 7)

## Conventions

- **Tooling language policy (maintainer decision, 2026-07):**
  - **Anything a USER runs is Rust** — demo, engines, setup/weight
    conversion. End users must never need a Python environment
    (`tools/convert_*.py` are slated for Rust replacements; see the
    Rust-toolchain issue).
  - **Quality-assurance comparisons stay Python — by design.** Golden
    fixtures and parity references (`tools/gen_*_fixtures.py`) must be
    produced by the *official* PyTorch implementations: their whole value
    is being an independent ground truth. Porting them to Rust would make
    the goldens self-referential and worthless. Do not "clean this up".
- Keep new code consistent with the surrounding style; run `cargo fmt` and
  `cargo clippy` (both must be clean).
- Tensor shapes follow `[batch, time, dim]`; document the shape of every
  tensor argument and return value in doc comments, as the existing modules do.
- Paper references belong in module-level docs (`//!`) with section numbers
  (e.g. "§3.2"), so code can be audited against the paper.
- CPU single-thread streaming performance is a feature: avoid unnecessary
  allocations/recomputation in the per-chunk path (see issue #6).

## License

MIT OR Apache-2.0. See [`LICENSE-MIT`](LICENSE-MIT) /
[`LICENSE-APACHE`](LICENSE-APACHE).
