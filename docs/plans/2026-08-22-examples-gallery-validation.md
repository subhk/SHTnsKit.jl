# Examples Gallery Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every embedded or directly linked example in the web documentation's Examples Gallery execute successfully and report scientifically meaningful accuracy.

**Architecture:** Execute the Markdown's serial Julia fences directly from the serial test suite so the tested source is exactly what readers copy. Keep MPI validation in the existing MPI examples workflow, and make numerical examples use band-limited fields plus explicit tolerances so projection error cannot masquerade as a successful roundtrip.

**Tech Stack:** Julia 1.10–1.12, Test, BenchmarkTools, MPI.jl, PencilArrays.jl, PencilFFTs.jl, Documenter.jl.

### Task 1: Add an executable gallery regression test

**Files:**
- Create: `test/serial/test_examples_gallery.jl`
- Modify: `test/serial/runtests.jl`
- Modify: `Project.toml`

1. Parse all `julia` fences from `docs/src/examples/index.md` and assert the expected inventory of 14 blocks.
2. Execute every non-MPI block in an isolated Julia module.
3. Assert every reported serial roundtrip error is below `1e-9` and rotation energy is preserved.
4. Run the test and verify the current rotation, vector, and benchmark examples fail for the observed reasons.

### Task 2: Correct the serial gallery examples

**Files:**
- Modify: `docs/src/examples/index.md`

1. Replace the vector field with analytically band-limited colatitude/azimuthal components and use physical vector-energy helpers.
2. Replace the nonexistent dense rotation wrapper with `analysis_packed`, `SHTRotation`, `shtns_rotation_apply_real`, and `synthesis_packed`.
3. Replace the benchmark field's non-band-limited term and bound BenchmarkTools sampling time.
4. Run `test/serial/test_examples_gallery.jl` and verify all 11 serial examples pass.

### Task 3: Correct the linked MPI roundtrip example

**Files:**
- Modify: `examples/parallel_roundtrip.jl`
- Modify: `.github/workflows/mpi-examples.yml`

1. Generate a deterministic band-limited scalar field from global spherical coordinates.
2. Fill each PencilArray from `range_local`, assert both serial and distributed errors are below `1e-10`, and exit nonzero on failure.
3. Point the second workflow example step at `parallel_fft_roundtrip.jl`, matching the gallery.
4. Run both scripts with two MPI ranks and verify they pass.

### Task 4: Verify the full documentation surface

**Files:**
- Modify: `docs/make.jl`

1. Re-run all 14 embedded Julia examples (11 serial, 3 MPI).
2. Re-run both gallery-linked scripts with two MPI ranks.
3. Generate Literate pages as display-only Julia fences so normal docs builds do not execute fragmented MPI and benchmark scripts.
4. Build the documentation with `julia --project=docs docs/make.jl`.
5. Run the relevant serial package tests and inspect the final diff.
