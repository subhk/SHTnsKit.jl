# User-Focused Web Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the published documentation into a short, task-oriented path for first-time scientific users while preserving the essential grid imagery and executable examples.

**Architecture:** Apply progressive disclosure in `docs/make.jl`, then shorten the entry pages and specialized guides. Keep maintainer evidence and duplicate legacy pages in the repository but outside the published navigation. Guard the intended information architecture, grid figure, and curated gallery with source-level regression tests, then validate the rendered site visually.

**Tech Stack:** Julia 1.12, Documenter.jl, Markdown, HTML/CSS, SHTnsKit.jl test suite.

### Task 1: Specify the public documentation contract

**Files:**
- Modify: `test/serial/test_examples_gallery.jl`

1. Add a `Web documentation structure` testset that reads `docs/make.jl`, `docs/src/grids.md`, and `docs/src/examples/index.md`.
2. Require the desktop and stacked grid SVGs, their `<picture>` wrapper, and meaningful alternative text.
3. Require a six-snippet curated gallery with no embedded MPI program.
4. Require the public navigation to omit generated examples, Performance Tips, and the SHTns 3.7 parity page.
5. Run `julia --project=. --startup-file=no test/serial/test_examples_gallery.jl` and confirm failures describe the current fourteen-snippet gallery and cluttered navigation.

### Task 2: Simplify navigation and entry pages

**Files:**
- Modify: `docs/make.jl`
- Modify: `docs/src/index.md`
- Modify: `docs/src/quickstart.md`
- Modify: `docs/src/installation.md`

1. Remove Literate script generation and its dynamic navigation section from the web build.
2. Publish only task-oriented Getting Started, User Guide, and Reference groups; move migration to Reference and omit duplicate/maintainer-only pages.
3. Reduce the home page to purpose, four capabilities, one install command, one tested roundtrip, and a task-based route table.
4. Reduce Quick Start to array layout, one scalar roundtrip, a grid choice, and links to specialized workflows.
5. Reduce Installation to core/optional packages, one verification, and three short troubleshooting cases.
6. Run the structure test and confirm only the gallery-size expectation remains red.

### Task 3: Trim specialist guides without hiding necessary workflows

**Files:**
- Modify: `docs/src/gpu.md`
- Modify: `docs/src/distributed.md`
- Modify: `docs/src/performance.md`
- Modify: `docs/src/advanced.md`

1. Keep GPU installation, a device-resident roundtrip, plan reuse, and concise troubleshooting; remove compatibility-wrapper, utility-catalog, and distributed-GPU internals.
2. Replace the distributed guide with when-to-use guidance, package setup, one accurate runnable scalar example, launcher instructions, and a brief replicated-vs-distributed layout decision.
3. Collapse performance advice into four priorities: reuse, batch, keep data resident, and benchmark end to end.
4. Remove advanced-page duplication already covered by Quick Start and Performance while retaining packed storage, operators/diagnostics, rotations, axisymmetric transforms, and AD entry points.

### Task 4: Curate and continuously execute the gallery

**Files:**
- Modify: `docs/src/examples/index.md`
- Modify: `test/serial/test_examples_gallery.jl`

1. Replace decorative banners and difficulty cards with a one-paragraph task index.
2. Keep six useful examples: scalar roundtrip, power spectrum, vector decomposition, stream function, benchmarking, and rotation.
3. Replace three embedded MPI programs with links and commands for `examples/parallel_roundtrip.jl` and `examples/parallel_fft_roundtrip.jl`.
4. Make every retained snippet self-checking where an invariant is available.
5. Run the focused gallery/structure test and require green.

### Task 5: Refresh repository evidence and verify the rendered site

**Files:**
- Modify: `test/fixtures/compatibility/task16_gate.toml`
- Modify: `test/fixtures/compatibility/task16_local_commands.txt`
- Modify: `test/fixtures/compatibility/task16_local_summary.log`

1. Compute the exact audited-tree digest and refresh the excluded readiness-evidence files and hashes.
2. Run the focused Task 16 gate and the full recorded `Pkg.test()` command.
3. Run `julia --project=docs --startup-file=no docs/make.jl` and require a clean Documenter build.
4. Serve `docs/build`, inspect Home, Quick Start, Gallery, and Grid Types at desktop and mobile widths, and confirm both grid image variants render.
5. Run `git diff --check`, review the branch diff against `origin/main`, commit the implementation, push `fix/gallery-examples`, and update PR #47.
