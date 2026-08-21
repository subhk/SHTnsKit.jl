# Web Documentation Sync Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the published Documenter site accurately describe SHTnsKit v2.0.0 and verify its primary CPU examples against the current API.

**Architecture:** Keep `docs/src` as the published source of truth. Replace stale legacy recipes with short executable examples, expose the existing grid and normalization guides in site navigation, add a focused v2 migration page, and preserve hardware-specific examples as non-executed snippets. Treat the full Documenter build plus focused Julia smoke commands as the acceptance gate.

**Tech Stack:** Julia 1.10–1.12, Documenter.jl, Literate.jl, SHTnsKit.jl, CUDA.jl/AMDGPU.jl extension documentation, MPI/PencilArrays documentation.

### Task 1: Repair readiness evidence after the branch update

**Files:**
- Modify: `test/fixtures/compatibility/task16_gate.toml`
- Modify: `test/fixtures/compatibility/task16_local_summary.log`

1. Run `test/serial/test_final_parity_gate.jl` and confirm the audited-tree digest fails.
2. Compute the digest on the exact PR head.
3. Update the recorded digest and summary hash.
4. Re-run the focused gate and expect 4,765 passing checks.

### Task 2: Align site structure and migration guidance

**Files:**
- Modify: `docs/make.jl`
- Create: `docs/src/migration.md`
- Modify: `.gitignore`

1. Add Grid Types, Normalization and Phase, and Migrating to v2.0 to the Documenter navigation.
2. Document removed compatibility APIs, typed devices, distributed-plan changes, axisymmetric scaling, and stricter validation.
3. Ignore generated `docs/src/literated` output so local builds do not dirty the worktree.

### Task 3: Replace stale user-facing examples

**Files:**
- Modify: `docs/src/index.md`
- Modify: `docs/src/installation.md`
- Modify: `docs/src/quickstart.md`
- Modify: `docs/src/gpu.md`
- Modify: `docs/src/advanced.md`
- Modify: `docs/src/performance.md`
- Modify: `docs/src/performance_tips.md`
- Modify: `docs/src/grids.md`
- Modify: `docs/src/norms.md`
- Modify: `docs/src/api/index.md`
- Modify: `README.md`

1. Remove calls to deleted getters, thread controls, and legacy `SHTnsConfig`/`analyze` APIs.
2. Correct in-place output/input argument order.
3. Document vendor-neutral CUDA/AMDGPU array dispatch and explicit CUDA compatibility fallback.
4. Replace oversized speculative advanced/performance recipes with current plan, batch, packed, diagnostic, and device APIs.
5. Make primary CPU examples executable Documenter `@example` blocks.

### Task 4: Verify and publish

**Files:**
- Verify all modified files.

1. Execute the corrected CPU smoke examples.
2. Run `julia --project=docs docs/make.jl` and require a successful render with doctests.
3. Run the focused Task 16 gate again.
4. Run `git diff --check` and review the complete diff.
5. Commit and push the update to `fix/review-findings-20260821`.
