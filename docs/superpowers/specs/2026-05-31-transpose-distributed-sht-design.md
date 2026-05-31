# Transpose-Based Distributed SHT — Design Spec

**Date:** 2026-05-31
**Status:** Approved design, pending implementation plan.
**Prerequisite (done):** lmax≥151 Legendre overflow fix (merged to `main`, commit `44a59c5`) — provides the bounded `Plm_norm_row!` / `Plm_norm_and_dPdtheta_row!` / `Plm_norm_dPdtheta_over_sinth_row!` used here.

## Problem

SHTnsKit's current distributed transforms (`dist_analysis`/`dist_synthesis`) use a **gather + Allreduce** design: when φ is distributed they `Allgatherv!` the full longitude onto every rank, and the spectral combine is an `Allreduce` of the full `(lmax+1, mmax+1)` matrix (replicated on every rank). Communication is O(lmax²) **fixed** (independent of rank count `p`) and the spectral result is **replicated** (O(lmax²)/rank). Measured: it strong-scales on-node only to a handful of ranks, then plateaus; it cannot scale to high rank counts or distribute spectral memory. SHTnsKit has **0 Alltoall** today.

## Goal

A **transpose-based** distributed scalar + vector SHT (the distributed-FFT pattern): redistribute data between θ-local and m-local layouts with an **Alltoall transpose**, run the Legendre step on fully-local data (no reduction, no gather), and leave the spectral coefficients **distributed** (m-pencil, O(lmax²/p) per rank). This strong-scales to high rank counts and distributes spectral memory. Verified to beat the current θ-decomposition at higher `p`.

## Scope (this spec)

- **In:** scalar (analysis/synthesis) AND vector (spheroidal/toroidal + QST) transpose-based distributed transforms; plan-based, **batched over a leading radial-level dimension**; m-pencil distributed spectral output.
- **Out (untouched):** the existing gather-based `dist_analysis`/`dist_synthesis` (kept as the dense-output convenience path); GPU; non-batched single-field (it is the batch=1 special case).

## Approach (chosen: A)

Reuse the two dependencies GeoDynamo already loads:
- **PencilFFTs** for the distributed real FFT along φ (it owns the φ-transpose + FFT, fused and production-tuned).
- **PencilArrays.transpose!** for the θ↔m redistribution (production-tuned `Alltoallv` with efficient packing, optional non-blocking/overlap).

Rejected: hand-rolled `MPI.Alltoallv!` (B — theoretical edge from triangular packing / m-truncation either doesn't apply to the rectangular Fourier-data transpose or is also achievable within A by transposing only the m≤mmax sub-pencil; realistically matches A at best, more code/risk); extending `DistributedSpectralPlan2D` (C — gather-era baggage, no perf upside).

## Architecture

New file `ext/ParallelTransposeTransforms.jl` (in the existing parallel extension `SHTnsKitParallelExt`). Public functions declared as stubs in `src/SHTnsKit.jl` (erroring "Parallel extension not loaded") and implemented in the ext, mirroring the existing distributed API pattern.

Layout convention (batched, per local radial slab):
- **Spatial pencil:** `f[lev, θ, φ]` — leading `lev` (radial level) axis is a non-transformed **batch**; `(θ, φ)` are the horizontal dims, 2D-distributed across the process grid. (`lev` extent = however many radial levels are local to this rank; the SHT treats `lev` purely as a batch axis and transposes only `(θ,φ)`.)
- **Spectral pencil (output):** `Alm[lev, l, m]` — `lev` batch, `l` **local** (complete), `m` **distributed** (m-pencil). Memory O(lmax²/p) per rank. Matches GeoDynamo's distributed-spectral concept (it already builds `Pencil(spec_dims,(1,2))`); l-redistribution to a full 2D (l,m) pencil, if GeoDynamo wants it, is a cheap extra `transpose!` left to the caller / a follow-up.

### Component: `DistTransposePlan`

Constructed once from `(cfg, spatial_pencil; use_rfft=true)`; reusable, 0-alloc steady state. Caches:
- the batched **PencilFFTs** rFFT/irFFT plan over the φ dim (with the `lev` batch and θ distributed);
- the batched **PencilArrays.transpose!** plan(s) for the θ↔m redistribution (the intermediate "m-pencil" config: θ local, m distributed, lev batch);
- **bounded normalized Legendre tables** `NP[l, θ]` (and `NdP` for vector) for **this rank's local-m subset only** — built from `Plm_norm_row!` / `Plm_norm_and_dPdtheta_row!` (the lmax-fix functions, |P̄|≲1). Memory O(lmax · nlat · m_local);
- intermediate + output pencil specs and scratch buffers (Fourier buffer, transpose buffers), all sized for the batch extent;
- the **m ownership** chosen with the interleaved order (0, mmax, 1, mmax−1, …) split across ranks for load balance (Legendre work per m ∝ lmax−m+1, decreasing).

### Data flow — scalar analysis `dist_analysis!(plan, Alm, f)`

1. Batched real FFT along φ (PencilFFTs): `f[lev,θ,φ] → F[lev,θ,m]`, keeping only bins `m = 0..mmax` (truncate before transpose → less comm).
2. Batched `transpose!`: `F[lev,θ_dist,m] → F[lev,θ_local,m_dist]` (the Alltoall; `lev` rides along — one collective for the whole slab).
3. Local Legendre, per `(lev, m_local)`: `Alm[lev,l,m] = Σ_θ NP[l,θ] · F[lev,θ,m]` — fully local, no communication.
   Output `Alm[lev,l,m_dist]` (m-pencil).

### Data flow — scalar synthesis `dist_synthesis!(plan, f, Alm)` (exact reverse)

1. Local inverse Legendre, per `(lev,m_local)`: `F[lev,θ_local,m] = Σ_l NP[l,θ] · Alm[lev,l,m]`.
2. Batched `transpose!` back: `F[lev,θ_local,m_dist] → F[lev,θ_dist,m]`.
3. Batched inverse real FFT (PencilFFTs): `F[lev,θ,m] → f[lev,θ,φ]` (Hermitian handled by irFFT).

### Vector (sphtor / qst)

Identical batched transposes. The local Legendre step uses the normalized `dP̄/dθ` and `P̄/sinθ` rows (from the lmax fix) to form spheroidal/toroidal `(S,T)` exactly as the serial sphtor kernels do (which were already migrated to the normalized rows). QST = scalar transform for Q + sphtor for (S,T). Vector spectral output is two/three m-pencils.

## API (new exports)

```julia
plan = DistTransposePlan(cfg::SHTConfig, spatial_pencil::Pencil; use_rfft::Bool=true)
dist_analysis!(plan, Alm::PencilArray, f::PencilArray)          # scalar
dist_synthesis!(plan, f::PencilArray, Alm::PencilArray)
dist_analysis_sphtor!(plan, Slm, Tlm, Vt, Vp)                   # vector
dist_synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm)
dist_analysis_qst!(plan, Qlm, Slm, Tlm, Vr, Vt, Vp)            # QST
dist_synthesis_qst!(plan, Vr, Vt, Vp, Qlm, Slm, Tlm)
```
The existing dense `dist_analysis(cfg, fθφ)` etc. remain unchanged. New names use the `!`-with-plan form to avoid collision and signal the distributed-spectral, preallocated contract. A helper `allocate_spectral(plan)` returns a correctly-shaped output `PencilArray`.

## Error handling

- Validate spatial pencil global dims vs cfg: horizontal dims must be `(nlat, nlon)`; identify the batch (lev) axis.
- Require `use_rfft` representability (`mmax ≤ nlon÷2`); real eltype for rFFT path.
- Validate that `Alm`'s pencil is the plan's spectral pencil (comm/topology match); clear `DimensionMismatch`/`ArgumentError` otherwise.
- The plan owns all comms/topologies derived from the input pencil's communicator — assert consistency at construction.

## Testing

1. **Round-trip vs serial reference:** gather the output m-pencil to a dense `(lmax+1,mmax+1)` matrix and compare to serial `analysis(cfg,·)` / `synthesis(cfg,·)`; lmax ∈ {32,64,128,256,512}, batch ∈ {1, >1}, ranks ∈ {1,2,4}. rtol 1e-8.
2. **Cross-check** vs the existing dense `dist_analysis` (same field, gathered) — agree to 1e-8.
3. **Vector** round-trip (sphtor + qst) at lmax 256, batch>1.
4. **Strong-scaling** (measured via a benchmark script): the transpose path beats the current θ-decomposition at higher rank counts, and batched beats per-field (fewer/larger collectives). Report speedup curves.
5. **Memory / allocation:** spectral footprint O(lmax²/p); 0-alloc steady state after warmup (reused plan/buffers).
6. **MPI launch:** via MPI.jl's `mpiexec()` wrapper (system mpiexec incompatible); env devs SHTnsKit + MPI/PencilArrays/PencilFFTs.

## Integration notes (GeoDynamo)

- GeoDynamo is a 3D (r,θ,φ) solver that already does its own r/θ/φ pencil transposes and has a distributed spectral pencil. It would hand this SHT a `(lev,θ,φ)` pencil for its local radial slab and consume the `(lev,l,m)` m-pencil spectral output. The level dim is GeoDynamo's local radial extent.
- GeoDynamo currently pins **registered SHTnsKit v1.2.9**; consuming this requires a SHTnsKit release + a GeoDynamo dep bump (out of scope here).

## Risks / open implementation questions (resolve in the plan)

- **Pencil choreography:** exact PencilArrays `Pencil`/`transpose!` configuration to go (θ,φ)-2D-dist → (θ-local, m-dist) with a batch axis, and whether PencilFFTs' output pencil can feed `transpose!` directly or needs an explicit intermediate config. Prototype + validate the pencil configs first (plan Task 1).
- **m-truncation in the transpose:** transpose only m≤mmax columns (sub-pencil) rather than all nlon÷2+1 bins — confirm PencilArrays supports transposing the truncated view efficiently.
- **Load balance:** interleaved m-ownership vs PencilArrays' default block distribution — may require a custom permutation/index map; verify the balance empirically.
- **Batched transpose:** confirm `transpose!` + PencilFFTs handle the leading batch (lev) dim as expected (it should, treating it as a non-distributed extra dim).
