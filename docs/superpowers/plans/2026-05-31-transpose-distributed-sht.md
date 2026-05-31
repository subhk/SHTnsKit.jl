# Transpose-Based Distributed SHT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a plan-based, batched, transpose-based distributed scalar+vector SHT to SHTnsKit that leaves spectral coefficients distributed (m-pencil, O(lmax²/p)) and strong-scales to high rank counts — replacing gather/Allreduce with an Alltoall transpose.

**Architecture:** Per local radial slab `(lev,θ,φ)`: distributed real-FFT along φ (PencilFFTs) → `transpose!` (PencilArrays Alltoall) to a θ-local/m-distributed layout (lev batch rides along) → fully-local Legendre using the bounded normalized rows from the lmax-fix → m-pencil spectral output. Synthesis is the exact reverse. Built in the existing `SHTnsKitParallelExt`, alongside (not replacing) the dense gather-based `dist_analysis`.

**Tech Stack:** Julia 1.11, MPI.jl 0.20, PencilArrays 0.19, PencilFFTs 0.15, FFTW. Reuses `Plm_norm_row!` / `Plm_norm_and_dPdtheta_row!` / `Plm_norm_dPdtheta_over_sinth_row!` (already on `main`).

**Spec:** `docs/superpowers/specs/2026-05-31-transpose-distributed-sht-design.md`.

---

## Environment (read first)

- `git`/`juliaup` launcher is broken; use the direct binary: `JL=~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia`.
- Work in the isolated worktree created at execution (subagent-driven-development creates it via `git worktree add ../SHTnsKit-transpose -b transpose-dist-sht`). Commit per task **on that branch** (authorized). Do NOT touch `main` or other checkouts.
- MPI env that `dev`s the worktree + adds parallel deps:
  ```
  mkdir -p /tmp/tdsht-mpi
  $JL --project=/tmp/tdsht-mpi -e 'using Pkg; Pkg.develop(path="<WORKTREE_ABS_PATH>"); Pkg.add(["MPI","PencilArrays","PencilFFTs"])'
  ```
  Launcher `/tmp/tdsht-mpi/launch.jl`:
  ```julia
  using MPI
  n=parse(Int,ARGS[1]); script=ARGS[2]; jl=Base.julia_cmd().exec[1]
  mpiexec() do exe; run(`$exe -n $n $jl --project=/tmp/tdsht-mpi $script`); end
  ```
  Run MPI tests: `$JL --project=/tmp/tdsht-mpi /tmp/tdsht-mpi/launch.jl <nranks> <script.jl>`. (System mpiexec is incompatible OpenMPI — must use MPI.jl's `mpiexec()`.)
- The extension only activates when MPI **and** PencilArrays **and** PencilFFTs are all loaded.

## File Structure

- Create `ext/ParallelTransposeTransforms.jl` — the whole transpose path: `DistTransposePlan`, `dist_analysis!`/`dist_synthesis!` (scalar), sphtor/qst variants, `allocate_spectral`. `include`d from `ext/SHTnsKitParallelExt.jl`.
- Modify `ext/SHTnsKitParallelExt.jl` — `include("ParallelTransposeTransforms.jl")` and export the new names.
- Modify `src/SHTnsKit.jl` — add function stubs (`error("Parallel extension not loaded...")`) + exports for `DistTransposePlan`, `dist_analysis!`, `dist_synthesis!`, `dist_analysis_sphtor!`, `dist_synthesis_sphtor!`, `dist_analysis_qst!`, `dist_synthesis_qst!`, `allocate_spectral`.
- Create `test/parallel/test_transpose_sht.jl` — MPI round-trip + batching + vector tests.
- Create `benchmark/transpose_scaling.jl` — strong-scaling measurement (θ-decomp vs transpose; per-field vs batched).

---

## Task 1: Spike — establish & record the pencil/transpose choreography

**This is a research spike. Its deliverable is a VALIDATED, RECORDED sequence of `Pencil`/`PencilFFTs`/`transpose!` calls that round-trips a field through (θ,φ)-dist → φ-FFT → θ-local/m-dist → back. All later tasks depend on the exact configs found here.**

**Files:** Create (scratch) `/tmp/tdsht-mpi/spike.jl`.

- [ ] **Step 1: Write the spike harness** (candidate API — PencilArrays 0.19 / PencilFFTs 0.15; correct it empirically if calls differ):

```julia
using MPI; MPI.Init()
using PencilArrays, PencilFFTs, MPI
using PencilFFTs: Transforms
comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm)

nlat, nlon = 24, 48          # 2D field; no batch yet
# 2D-distributed (θ,φ) spatial pencil
pen_sp = Pencil((nlat, nlon), (1, 2), comm)
# PencilFFTs: FFT only along φ (dim 2): NoTransform on θ, RFFT on φ
tr = (Transforms.NoTransform(), Transforms.RFFT())
fft_plan = PencilFFTPlan(pen_sp, tr)
f  = PencilArray{Float64}(undef, pen_sp); rand!(parent(f))
F  = allocate_output(fft_plan)            # complex, φ→m (nlon÷2+1 bins), some pencil
mul!(F, fft_plan, f)                      # distributed rFFT along φ
# Now redistribute F so that θ (dim1) is LOCAL and m (dim2) is DISTRIBUTED.
pen_m = Pencil(size_global(pencil(F)), (2,), comm)   # decompose only dim2 (m) → θ local
Fm = PencilArray{eltype(F)}(undef, pen_m)
transpose!(Fm, F)                          # the Alltoall
# checks
if rank == 0
    println("F pencil dims: ", size_global(pencil(F)), " decomp ", decomposition(pencil(F)))
    println("Fm local θ extent = ", size(parent(Fm),1), " (want == nlat=$nlat, i.e. θ complete)")
end
@assert size(parent(Fm), 1) == nlat "θ not local after transpose"
# round-trip: transpose back, irFFT, compare
F2 = similar(F); transpose!(F2, Fm)
f2 = similar(f); ldiv!(f2, fft_plan, F2)   # inverse rFFT
err = maximum(abs.(parent(f2) .- parent(f)))
gerr = MPI.Allreduce(err, MPI.MAX, comm)
rank==0 && println("round-trip err = ", gerr)
@assert gerr < 1e-12
rank==0 && println("SPIKE OK")
MPI.Finalize()
```

- [ ] **Step 2: Run at 1 and 4 ranks**, iterating the API until it works:

Run: `$JL --project=/tmp/tdsht-mpi /tmp/tdsht-mpi/launch.jl 4 /tmp/tdsht-mpi/spike.jl`
Expected: `θ not local` assert and `round-trip err` pass; prints `SPIKE OK`.
If `Pencil`/`PencilFFTPlan`/`allocate_output`/`transpose!`/`decomposition`/`size_global` signatures differ in the installed versions, FIX the calls (consult `PencilArrays`/`PencilFFTs` docs via the REPL `?` or `names(PencilArrays)`), until the round-trip passes. The validated calls are the deliverable.

- [ ] **Step 3: Add the batch (lev) dim** — extend the spike to a 3D pencil `(nlev, nlat, nlon)` decomposed on `(2,3)` (θ,φ distributed; lev local batch), transforms `(NoTransform, NoTransform, RFFT)`, and transpose to decompose only the m dim so θ AND lev are local. Re-run at 4 ranks, confirm `SPIKE OK` with `nlev=3`.

- [ ] **Step 4: RECORD the working config** in a comment block at the top of `/tmp/tdsht-mpi/spike.jl` AND in the task report: the exact `Pencil(...)`, `PencilFFTPlan(...)`, output-pencil construction, and `transpose!` calls that worked, for both 2D and batched. Report these verbatim — Tasks 2–6 use them.

- [ ] **Step 5: Commit the spike** as documentation:
```bash
mkdir -p benchmark && cp /tmp/tdsht-mpi/spike.jl benchmark/transpose_spike.jl
git add benchmark/transpose_spike.jl
git commit -m "spike: validated PencilFFTs+transpose! choreography for distributed SHT"
```

**If the choreography cannot be made to round-trip (e.g. PencilFFTs can't do single-dim FFT as expected): STOP, report BLOCKED with what failed — the controller will reconsider (e.g. fall back to approach B for the φ-FFT).**

---

## Task 2: `DistTransposePlan` struct + constructor

**Files:** Create `ext/ParallelTransposeTransforms.jl`; Modify `ext/SHTnsKitParallelExt.jl` (add `include`), `src/SHTnsKit.jl` (stub + export `DistTransposePlan`, `allocate_spectral`). Test: `test/parallel/test_transpose_sht.jl`.

- [ ] **Step 1: Stub + export in `src/SHTnsKit.jl`** (near the other distributed stubs/exports):
```julia
export DistTransposePlan, allocate_spectral
DistTransposePlan(cfg, pencil; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
allocate_spectral(plan) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
```

- [ ] **Step 2: Write the failing plan-construction test** (`test/parallel/test_transpose_sht.jl`):
```julia
using MPI; MPI.Init()
using SHTnsKit, PencilArrays, PencilFFTs, Test
comm = MPI.COMM_WORLD
lmax=32; nlat=lmax+2; nlon=2*lmax+1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)
pen = Pencil((nlat, nlon), (1,2), comm)
plan = DistTransposePlan(cfg, pen; use_rfft=true)
@test plan.cfg === cfg
@test plan.nlat == nlat && plan.nlon == nlon
@test !isempty(plan.m_local)                       # this rank owns some m
@test all(0 .<= plan.m_local .<= cfg.mmax)
Alm = allocate_spectral(plan)
@test Alm isa PencilArray
MPI.Finalize()
```
Run at 1 rank: `$JL --project=/tmp/tdsht-mpi /tmp/tdsht-mpi/launch.jl 1 test/parallel/test_transpose_sht.jl` → FAIL (extension stub error / undefined).

- [ ] **Step 3: Implement `DistTransposePlan`** in `ext/ParallelTransposeTransforms.jl` using the Task-1 config. Struct fields:
```julia
struct DistTransposePlan{TF, TT, TP}
    cfg::SHTnsKit.SHTConfig
    nlat::Int; nlon::Int; lmax::Int; mmax::Int
    spatial_pencil::TP            # input (θ,φ) pencil (lev batch handled per-call or as 3D pencil)
    fft_plan::TF                  # PencilFFTPlan (φ rFFT)  — from Task 1
    m_pencil                      # spectral-side pencil: θ-local, m-dist  (from Task 1)
    transpose_buf                 # PencilArray for the m-pencil Fourier data
    m_local::Vector{Int}          # global m indices this rank owns (interleaved order)
    NP::Vector{Matrix{Float64}}   # NP[mi][l+1, θ] = P̄_l^m(θ) for each local m (bounded normalized)
    # (NdP added in Task 6 for vector)
end
```
Constructor: build `fft_plan` and the m-pencil exactly as Task 1 recorded; determine `m_local` from the m-pencil's local range; build `NP` tables for each local m via `SHTnsKit.Plm_norm_row!` over all θ (cfg.x). Implement `SHTnsKit.allocate_spectral(plan)` to return a spectral `PencilArray{ComplexF64}` on a pencil with global dims `(lmax+1, mmax+1)` decomposed on m (matching `m_local`). Add `include("ParallelTransposeTransforms.jl")` to `ext/SHTnsKitParallelExt.jl`.

- [ ] **Step 4: Run the test at 1 and 4 ranks** → PASS. Confirm `m_local` partitions `0:mmax` exactly once across ranks (add a temporary `MPI.Allreduce` count check; assert union == 0:mmax).

- [ ] **Step 5: Commit:**
```bash
git add ext/ParallelTransposeTransforms.jl ext/SHTnsKitParallelExt.jl src/SHTnsKit.jl test/parallel/test_transpose_sht.jl
git commit -m "feat(dist): DistTransposePlan struct + constructor (PencilFFTs + transpose + local-m tables)"
```

---

## Task 3: Scalar `dist_analysis!`

**Files:** Modify `ext/ParallelTransposeTransforms.jl`, `src/SHTnsKit.jl` (stub+export `dist_analysis!`). Test: extend `test/parallel/test_transpose_sht.jl`.

- [ ] **Step 1: Stub+export** `dist_analysis!(plan, Alm, f) = error(...)` in `src/SHTnsKit.jl`.

- [ ] **Step 2: Failing round-trip-vs-serial test:**
```julia
@testset "transpose dist_analysis vs serial" begin
    lmax=64; nlat=lmax+2; nlon=2*lmax+1
    cfg=create_gauss_config(lmax,nlat;nlon=nlon)
    # band-limited field from known coeffs (serial reference)
    a0=zeros(ComplexF64,lmax+1,lmax+1)
    for m in 0:lmax,l in m:lmax; sc=1/(1+l)^2; a0[l+1,m+1]= m==0 ? complex(sc) : complex(sc,0.5sc); end
    f_full = SHTnsKit.synthesis(cfg,a0;real_output=true)
    pen = Pencil((nlat,nlon),(1,2),comm)
    f = PencilArray{Float64}(undef, pen)
    r = PencilArrays.range_local(pen)            # global θ,φ ranges of this rank
    for (jl,jg) in enumerate(r[2]), (il,ig) in enumerate(r[1]); parent(f)[il,jl]=f_full[ig,jg]; end
    plan = DistTransposePlan(cfg, pen; use_rfft=true)
    Alm = allocate_spectral(plan)
    dist_analysis!(plan, Alm, f)
    a_ref = SHTnsKit.analysis(cfg, f_full)       # dense serial
    # compare this rank's owned m-columns
    rA = PencilArrays.range_local(pencil(Alm))   # (l-range, m-range)
    err = 0.0
    for (jl,mg) in enumerate(rA[2]), (il,lg) in enumerate(rA[1])
        err = max(err, abs(parent(Alm)[il,jl] - a_ref[lg, mg]))
    end
    gerr = MPI.Allreduce(err, MPI.MAX, comm)
    @test gerr < 1e-8
end
```
Run at 1,2,4 ranks → FAIL.

- [ ] **Step 3: Implement `dist_analysis!`** (using Task-1 choreography):
```julia
function SHTnsKit.dist_analysis!(plan::DistTransposePlan, Alm::PencilArray, f::PencilArray)
    mul!(plan.fft_buf, plan.fft_plan, f)              # rFFT along φ -> F[θ_dist, m]
    transpose!(plan.transpose_buf, plan.fft_buf)      # -> F[θ_local, m_dist]
    F = parent(plan.transpose_buf)                    # local: (nlat, n_m_local)
    A = parent(Alm)                                   # local: (lmax+1, n_m_local)
    fill!(A, zero(eltype(A)))
    scaleφ = plan.cfg.cphi; w = plan.cfg.w
    @inbounds for (mi, m) in enumerate(plan.m_local)
        NP = plan.NP[mi]                              # (lmax+1, nlat)
        col = mi                                      # local m column index
        for i in 1:plan.nlat
            wi = w[i]; Fi = F[i, col]
            for l in m:plan.lmax
                A[l+1, col] += (wi * NP[l+1, i] * scaleφ) * Fi
            end
        end
    end
    return Alm
end
```
(`plan.fft_buf` = `allocate_output(fft_plan)` cached in the plan; add it to the struct + constructor. The `col` mapping from `mi` to the local Fourier column must match the m-pencil layout from Task 1 — verify the local column order equals `plan.m_local` order; if PencilArrays orders m-columns by global index, set `plan.m_local = collect(local m range)` in that same order rather than interleaved, OR carry an index map. Resolve against the Task-1 layout.)

- [ ] **Step 4: Run at 1,2,4 ranks** → PASS (gerr < 1e-8). If the m-column mapping is off, fix `m_local`/the index map until correct.

- [ ] **Step 5: Commit:**
```bash
git add ext/ParallelTransposeTransforms.jl src/SHTnsKit.jl test/parallel/test_transpose_sht.jl
git commit -m "feat(dist): transpose-based scalar dist_analysis! (m-pencil output)"
```

---

## Task 4: Scalar `dist_synthesis!`

**Files:** Modify `ext/ParallelTransposeTransforms.jl`, `src/SHTnsKit.jl`. Test: extend test file.

- [ ] **Step 1: Stub+export** `dist_synthesis!(plan, f, Alm) = error(...)`.

- [ ] **Step 2: Failing full round-trip test** (f → Alm → f̂, and vs serial synthesis):
```julia
@testset "transpose dist_synthesis round-trip" begin
    lmax=64; nlat=lmax+2; nlon=2*lmax+1
    cfg=create_gauss_config(lmax,nlat;nlon=nlon)
    a0=zeros(ComplexF64,lmax+1,lmax+1)
    for m in 0:lmax,l in m:lmax; sc=1/(1+l)^2; a0[l+1,m+1]= m==0 ? complex(sc) : complex(sc,0.5sc); end
    f_full=SHTnsKit.synthesis(cfg,a0;real_output=true)
    pen=Pencil((nlat,nlon),(1,2),comm); f=PencilArray{Float64}(undef,pen)
    r=PencilArrays.range_local(pen)
    for (jl,jg) in enumerate(r[2]),(il,ig) in enumerate(r[1]); parent(f)[il,jl]=f_full[ig,jg]; end
    plan=DistTransposePlan(cfg,pen;use_rfft=true)
    Alm=allocate_spectral(plan); dist_analysis!(plan,Alm,f)
    f2=PencilArray{Float64}(undef,pen); dist_synthesis!(plan,f2,Alm)
    lerr=maximum(abs.(parent(f2).-parent(f))); gerr=MPI.Allreduce(lerr,MPI.MAX,comm)
    @test gerr < 1e-8
end
```
Run 1,2,4 ranks → FAIL.

- [ ] **Step 3: Implement `dist_synthesis!`** (reverse): local inverse Legendre per local m → `transpose_buf[θ_local,m]`, `transpose!` back to `fft_buf[θ_dist,m]`, `ldiv!(f, fft_plan, fft_buf)` (inverse rFFT). Inverse Legendre:
```julia
inv_scaleφ = SHTnsKit.phi_inv_scale(plan.cfg)
@inbounds for (mi, m) in enumerate(plan.m_local)
    NP = plan.NP[mi]
    for i in 1:plan.nlat
        acc = zero(ComplexF64)
        for l in m:plan.lmax
            acc += NP[l+1, i] * A[l+1, mi]
        end
        F[i, mi] = inv_scaleφ * acc
    end
end
# then transpose! transpose_buf -> fft_buf; ldiv!(f, fft_plan, fft_buf)
```

- [ ] **Step 4: Run 1,2,4 ranks** → PASS. Also assert the result vs serial `synthesis(cfg, a0)` gathered (gerr<1e-8).

- [ ] **Step 5: Commit:** `git commit -m "feat(dist): transpose-based scalar dist_synthesis! (round-trip verified)"`.

---

## Task 5: Batching over the radial-level dimension

**Files:** Modify `ext/ParallelTransposeTransforms.jl` (accept a leading `lev` dim), test file.

- [ ] **Step 1: Failing batched test** — build a 3D pencil `(nlev, nlat, nlon)` decomposed on `(2,3)`, fill each level from a distinct band-limited field, `dist_analysis!`/`dist_synthesis!`, assert per-level round-trip < 1e-8 at nlev=4, ranks 1,2,4. (Use the Task-1 batched config.)

- [ ] **Step 2: Generalize** the plan + transforms to accept the leading batch dim: `fft_plan` and `m_pencil`/`transpose_buf` built for the 3D pencil; the Legendre loops gain an outer `for lev in 1:nlev_local` (the batch); buffers sized `(nlev_local, nlat, n_m_local)`. The plan detects batch vs 2D from the input pencil's `ndims` (3 ⇒ batched, 2 ⇒ single).

- [ ] **Step 3: Run** → PASS (nlev=4). Add a check that the number of `transpose!`/FFT collectives is independent of `nlev` (instrument with a counter or reason in a comment + assert one `transpose!` call per analysis regardless of nlev).

- [ ] **Step 4: Commit:** `git commit -m "feat(dist): batch transpose SHT over radial-level dim (one Alltoall per slab)"`.

---

## Task 6: Vector spheroidal/toroidal `dist_analysis_sphtor!` / `dist_synthesis_sphtor!`

**Files:** Modify `ext/ParallelTransposeTransforms.jl` (+ `NdP` tables in the plan), `src/SHTnsKit.jl`, test file.

- [ ] **Step 1: Add `NdP`/derivative tables to the plan** built from `SHTnsKit.Plm_norm_and_dPdtheta_row!` for each local m (store the θ-derivative `dP̄/dθ` and `P̄/sinθ` per (m,l,θ) needed by the sphtor kernel). Update the constructor; gate behind a `with_vector=true` kwarg (default true) to avoid the cost when only scalar is used.

- [ ] **Step 2: Stub+export** `dist_analysis_sphtor!`, `dist_synthesis_sphtor!`.

- [ ] **Step 3: Failing vector round-trip test** — band-limited `(S0,T0)` (l≥1), serial `synthesis_sphtor` → `(Vt,Vp)` full, scatter to (θ,φ) pencils, `dist_analysis_sphtor!` → `(Slm,Tlm)` m-pencils, compare to serial `analysis_sphtor`; then `dist_synthesis_sphtor!` round-trip. lmax 64, ranks 1,2,4, rtol 1e-7.

- [ ] **Step 4: Implement** the two functions: two φ-rFFTs (Vt,Vp) + transposes (or one transpose on a stacked buffer), then the local sphtor Legendre using `dP̄/dθ`, `P̄/sinθ`, and `coeff = wi*scaleφ/(l*(l+1))` — mirror the serial `_sphtor_analysis_kernel!` / `_sphtor_synthesis_kernel` math (already in `src/kernels.jl`) but reading the plan's normalized derivative tables and writing/reading the m-pencil `Slm`/`Tlm`.

- [ ] **Step 5: Run 1,2,4 ranks** → PASS. **Commit:** `git commit -m "feat(dist): transpose-based vector (sphtor) dist transforms"`.

---

## Task 7: QST `dist_analysis_qst!` / `dist_synthesis_qst!`

**Files:** Modify `ext/ParallelTransposeTransforms.jl`, `src/SHTnsKit.jl`, test file.

- [ ] **Step 1: Stub+export** the two qst functions.
- [ ] **Step 2: Failing qst round-trip test** (Q via scalar, S/T via sphtor), lmax 64, ranks 2,4, rtol 1e-7.
- [ ] **Step 3: Implement** by delegation: `dist_analysis_qst!(plan, Qlm, Slm, Tlm, Vr, Vt, Vp)` calls `dist_analysis!(plan, Qlm, Vr)` + `dist_analysis_sphtor!(plan, Slm, Tlm, Vt, Vp)`; synthesis analogously.
- [ ] **Step 4: Run** → PASS. **Commit:** `git commit -m "feat(dist): transpose-based QST dist transforms (delegating)"`.

---

## Task 8: m-truncation + load balance

**Files:** Modify `ext/ParallelTransposeTransforms.jl`, test file.

- [ ] **Step 1:** Ensure the FFT→transpose moves only `m = 0..mmax` columns (not all `nlon÷2+1` bins). If PencilFFTs output has `nlon÷2+1` m-bins, transpose only the `1:mmax+1` sub-pencil/view (confirm PencilArrays transposes the truncated view; if not, slice into a `(nlat, mmax+1)` buffer before transpose). Add a test asserting results unchanged (round-trip still <1e-8) — this is an optimization, correctness must be identical.
- [ ] **Step 2:** Make `m_local` use the interleaved order (0,mmax,1,mmax−1,…) split across ranks for Legendre load balance, OR confirm PencilArrays' default block m-distribution is acceptable and document the choice. If interleaved, carry an explicit global-m index map (the spectral pencil's physical columns ↔ global m). Add a test asserting every global m in `0:mmax` is owned by exactly one rank and the per-rank Legendre work (Σ (lmax−m+1)) is within ~1.3× of balanced.
- [ ] **Step 3: Run** → PASS (correctness unchanged). **Commit:** `git commit -m "perf(dist): m-truncated transpose + balanced m ownership"`.

---

## Task 9: Strong-scaling benchmark + verification

**Files:** Create `benchmark/transpose_scaling.jl`.

- [ ] **Step 1:** Write a benchmark that, for lmax ∈ {128,256,512} and ranks ∈ {1,2,4(,8)}, times: (a) the transpose path `dist_analysis!`+`dist_synthesis!` (this plan), (b) the existing θ-decomposition `dist_analysis`/`dist_synthesis` (dense), and reports wall time (slowest-rank min-of-N via `MPI.Barrier`+`Allreduce(MAX)`) + speedup vs 1 rank. Also compare batched (nlev=8) vs per-field (loop nlev=1×8).
- [ ] **Step 2: Run** at 2 and 4 ranks (via `mpiexec()` launcher). Record the speedup curves in the task report.
- [ ] **Step 3: Assert the design goal:** at the highest tested rank count, the transpose path's speedup ≥ the θ-decomposition's speedup (it should not plateau earlier), and batched < per-field wall time. If the transpose path does NOT win at the tested scale, report DONE_WITH_CONCERNS with the numbers (on a small on-node test the gather path may still be competitive — note the expected cross-over is at larger rank counts / cross-node).
- [ ] **Step 4: Commit:** `git add benchmark/transpose_scaling.jl && git commit -m "bench(dist): transpose vs gather strong-scaling"`.

---

## Task 10: API polish, docs, memory

**Files:** `src/SHTnsKit.jl`, `docs/src/distributed.md`, memory.

- [ ] **Step 1:** Confirm all new names are exported and the stubs error cleanly without the extension (`$JL --project=. -e 'using SHTnsKit; try DistTransposePlan(nothing,nothing) catch e; println(e); end'` → "Parallel extension not loaded").
- [ ] **Step 2:** Add a "Transpose-based distributed transforms (scalable)" section to `docs/src/distributed.md`: when to use (high rank counts / distributed spectral), the `(lev,θ,φ)`→`(lev,l,m)` layout, a runnable example mirroring the test, and a note that it complements (not replaces) the dense `dist_analysis`.
- [ ] **Step 3:** Run the full parallel test file once more at 4 ranks → all PASS. Run a serial `$JL --project=. -e 'using SHTnsKit'` load check (ext not loaded) → OK.
- [ ] **Step 4:** Update memory `[[parallel-cpu-scaling]]` with the transpose path's existence + measured scaling.
- [ ] **Step 5: Commit:** `git commit -m "docs(dist): document transpose-based distributed SHT + scaling"`. Then request merge permission (finishing-a-development-branch).

---

## Self-Review

- **Spec coverage:** plan struct (T2), scalar analysis/synthesis (T3,T4), batching (T5), vector sphtor (T6) + qst (T7), m-truncation + load balance (T8), strong-scaling verification (T9), API/docs (T10), and the highest-risk pencil choreography front-loaded as a validated spike (T1). ✔
- **Placeholder note:** the PencilArrays/PencilFFTs call signatures in T1/T2 are CANDIDATE and explicitly validated/corrected in the T1 spike before any dependent task — this is the deliberate spike-first structure for a research unknown, not a vague placeholder. Every SHT-logic step (Legendre loops, tests) has exact code.
- **Type consistency:** `DistTransposePlan` fields (`cfg,nlat,nlon,lmax,mmax,spatial_pencil,fft_plan,fft_buf,m_pencil,transpose_buf,m_local,NP,NdP`), `m_local::Vector{Int}`, and the `allocate_spectral`→m-pencil contract are used consistently across T2–T8. `dist_analysis!(plan,Alm,f)` / `dist_synthesis!(plan,f,Alm)` arg order fixed at T3/T4 and reused.
- **Risk gate:** T1 has an explicit BLOCKED path (if PencilFFTs can't do the single-dim FFT, reconsider) so the dependent tasks aren't built on a broken foundation.
