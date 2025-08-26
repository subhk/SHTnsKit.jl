# SHTnsKit.jl — Distributed Guide (MPI + PencilArrays)

This guide introduces the distributed (MPI) features of SHTnsKit.jl built on top of PencilArrays and PencilFFTs. It is written for both users and developers — start with the quick-start sections and dive deeper into the API and architecture notes when you need to extend functionality.

Table of contents
- 1. Overview
- 2. Installation and Requirements
- 3. Quick Start (Scalar, Vector, QST)
- 4. rFFT/irFFT Modes and Plans
- 5. Diagnostics (Spectral and Grid)
- 6. Operators and Rotations
- 7. Local/Point Evaluations and Packed Conversions
- 8. Automatic Differentiation (PencilArrays)
- 9. Performance Tips
- 10. Developer Notes (Where to Extend)

---

## 1. Overview

- SHTnsKit.jl provides pure-Julia spherical harmonic transforms (scalar and vector) with optional distributed execution using MPI.
- Distributed execution uses:
  - PencilArrays.jl for data layout and collective communication
  - PencilFFTs.jl for distributed FFTs
- You write the same transforms; pass PencilArrays to run distributed.

Key concepts
- PencilArray layouts:
  - Spatial grids: `(:θ, :φ)` and Fourier planes `(:θ, :k)`/`(:θ, :m)`
  - Spectral matrices: `(:l, :m)`
- Normalization and Condon–Shortley phase follow `cfg.norm` and `cfg.cs_phase` (consistent with the serial API).
- Robert form and associated Legendre table acceleration are supported in distributed vector transforms.

---

## 2. Installation and Requirements

```julia
using Pkg
Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs"])  # optional: "LoopVectorization"
```

MPI setup (OS-specific, then build MPI.jl):
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev

# macOS
brew install open-mpi

# Configure Julia MPI
julia -e 'using Pkg; Pkg.build("MPI")'
```

---

## 3. Quick Start (Scalar, Vector, QST)

Create a config and a Pencil grid
```julia
using SHTnsKit, MPI, PencilArrays, PencilFFTs

MPI.Init()

lmax = 32
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Spatial grid on a Pencil communicator
P = Pencil((:θ, :φ), (nlat, nlon); comm=MPI.COMM_WORLD)
fθφ = PencilArrays.zeros(P; eltype=Float64)

# Fill local block
for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
    fθφ[iθ, iφ] = sin(0.2*(iθ+1)) + cos(0.1*(iφ+1))
end
```

Scalar transforms (distributed)
```julia
# Analysis (use_rfft reduces FFT cost for real fields)
Alm = SHTnsKit.dist_analysis(cfg, fθφ; use_rfft=true)

# Synthesis (prototype_θφ selects the output layout/comm)
fθφ2 = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true, use_rfft=true)
```

Vector (spheroidal/toroidal) transforms (distributed)
```julia
Vt = copy(fθφ);
Vp = copy(fθφ);

# Analysis → Slm, Tlm (respects Robert form and PLM tables)
Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vt, Vp; use_rfft=true)

# Synthesis back to grid
Vt2, Vp2 = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vt, real_output=true, use_rfft=true)
```

QST transforms (distributed)
```julia
Vr = copy(fθφ)
Q, S, T = SHTnsKit.dist_spat_to_SHqst(cfg, Vr, Vt, Vp)
Vr2, Vt2, Vp2 = SHTnsKit.dist_SHqst_to_spat(cfg, Q, S, T; prototype_θφ=Vr, real_output=true, use_rfft=true)
```

Finalize MPI when done:
```julia
MPI.Finalize()
```

---

## 4. rFFT/irFFT Modes and Plans

Using `use_rfft=true` halves azimuthal FFT work for real inputs/outputs.

Direct API
- Scalar: `dist_analysis(...; use_rfft=true)`, `dist_synthesis(...; use_rfft=true)`
- Vector: `dist_spat_to_SHsphtor(...; use_rfft=true)`, `dist_SHsphtor_to_spat(...; use_rfft=true)`
- QST: `dist_SHqst_to_spat(...; use_rfft=true)`

Plan-based API (optional caching of FFT plans)
```julia
aplan = SHTnsKit.DistAnalysisPlan(cfg, fθφ; use_rfft=true)
Alm = zeros(ComplexF64, lmax+1, lmax+1)
SHTnsKit.dist_analysis!(aplan, Alm, fθφ)

spln = SHTnsKit.DistPlan(cfg, fθφ; use_rfft=true)
fθφ_out = similar(fθφ)
SHTnsKit.dist_synthesis!(spln, fθφ_out, PencilArray(Alm))
```

Enable plan caching across calls (optional)
```bash
export SHTNSKIT_CACHE_PENCILFFTS=1
```

---

## 5. Diagnostics (Spectral and Grid)

Pencil-aware (MPI-reduced) diagnostics
- `energy_scalar(cfg, Alm::PencilArray; real_field=true)`
- `energy_vector_l_spectrum(cfg, Slm::PencilArray, Tlm::PencilArray; real_field=true)`
- `enstrophy_m_spectrum(cfg, Tlm::PencilArray; real_field=true)`
- `grid_energy_scalar(cfg, fθφ::PencilArray)`

Roundtrip helpers (for sanity checks)
- `dist_scalar_roundtrip!(cfg, fθφ::PencilArray) -> (rel_local, rel_global)`
- `dist_vector_roundtrip!(cfg, Vtθφ::PencilArray, Vpθφ::PencilArray) -> ((rl_t, rg_t),(rl_p, rg_p))`

---

## 6. Operators and Rotations

Operators
- `dist_apply_laplacian!(cfg, Alm_pencil)` multiplies each l-row by `-l(l+1)` (local, no communication)
- `dist_SH_mul_mx!(cfg, mx, Alm_pencil, R_pencil)` applies banded operators with minimal communication

Rotations (distributed)
- Z rotation:
  - In-place: `dist_SH_Zrotate(cfg, Alm_p, α)`
  - Out-of-place: `dist_SH_Zrotate(cfg, Alm_p, α, R_p)`
- Y rotation (choose strategy):
  - Default (reduced bandwidth): `dist_SH_Yrotate(cfg, Alm_p, β, R_p)` — truncated gather per l
  - Full allgather: `dist_SH_Yrotate_allgatherm!(cfg, Alm_p, β, R_p)`
- Euler composite (ZYZ): `dist_SH_rotate_euler(cfg, Alm_p, α, β, γ, R_p)`
- 90° shortcuts:
  - `dist_SH_Yrotate90(cfg, Alm_p, R_p)` and `dist_SH_Xrotate90(cfg, Alm_p, R_p)`

Choosing Y-rotation variant
- Prefer truncated gather for broad spectra (typical) — gathers only `m ≤ l` per row.
- Use allgather if most rows have `l ≈ mmax` or the communicator is small.

Packed-vector rotation wrappers (dense Qlm → distributed → dense Rlm)
- `dist_SH_Zrotate_packed(cfg, Qlm, α; prototype_lm)`
- `dist_SH_Yrotate_packed(cfg, Qlm, β; prototype_lm)`
- `dist_SH_Yrotate90_packed(cfg, Qlm; prototype_lm)`
- `dist_SH_Xrotate90_packed(cfg, Qlm; prototype_lm)`

`prototype_lm` is a `PencilArray` with dims `(:l, :m)` providing the communicator/layout.

---

## 7. Local/Point Evaluations and Packed Conversions

Local/point (distributed reductions return the same result on all ranks)
- Scalar:
  - `dist_SH_to_point(cfg, Alm_pencil, cost, phi) -> value`
  - `dist_SH_to_lat(cfg, Alm_pencil, cost; nphi=cfg.nlon) -> Vector`
- QST:
  - `dist_SHqst_to_point(cfg, Q_p, S_p, T_p, cost, phi) -> (vr,vt,vp)`
  - `dist_SHqst_to_lat(cfg, Q_p, S_p, T_p, cost; nphi=cfg.nlon) -> (Vr,Vt,Vp)`

Packed conversions (dense/LM ⇄ distributed)
- Real (LM):
  - `dist_spat_to_SH_packed(cfg, fθφ::PencilArray) -> Qlm`
  - `dist_SH_packed_to_spat(cfg, Qlm; prototype_θφ, real_output=true) -> PencilArray`
- Complex (LM_cplx):
  - `dist_spat_cplx_to_SH(cfg, z::PencilArray) -> alm_packed`
  - `dist_SH_to_spat_cplx(cfg, alm_packed; prototype_θφ) -> PencilArray`

---

## 8. Automatic Differentiation (PencilArrays)

Zygote (distributed wrappers)
- Scalar: `g = zgrad_scalar_energy(cfg, fθφ::PencilArray)`
- Vector: `(gVt, gVp) = zgrad_vector_energy(cfg, Vtθφ::PencilArray, Vpθφ::PencilArray)`

ForwardDiff (distributed wrappers)
- Scalar: `g = fdgrad_scalar_energy(cfg, fθφ::PencilArray)`
- Vector: `(gVt, gVp) = fdgrad_vector_energy(cfg, Vtθφ::PencilArray, Vpθφ::PencilArray)`

Both use the distributed transform paths internally and return gradients in the same Pencil layout.

---

## 9. Performance Tips

- Use rFFT/irFFT for real fields: `use_rfft=true` in analysis/synthesis to halve FFT cost.
- Enable PLM tables when running many transforms with fixed grid:
  ```julia
  enable_plm_tables!(cfg)
  # or disable_plm_tables!(cfg)
  ```
- Robert form: for vector transforms, set `robert_form=true` in your config to stabilize polar behavior.
- Normalization/phase: match `cfg.norm` and `cfg.cs_phase` to your data; conversions are handled internally on input/output.
- FFT plan caching: `ENV["SHTNSKIT_CACHE_PENCILFFTS"] = "1"` to reuse PencilFFTs plans.
- Y-rotation strategy: truncated gather typically reduces bandwidth; switch to allgather for high-m–dominated spectra.

---

## 10. Developer Notes (Where to Extend)

Code structure (distributed extension)
- `ext/SHTnsKitParallelExt.jl`: loads the distributed pieces
  - `ext/parallel_transforms.jl`: scalar/vector/QST distributed transforms, roundtrips
  - `ext/parallel_ops_pencil.jl`: spectral operators on PencilArrays
  - `ext/parallel_rotations_pencil.jl`: rotations (Z, Y, Euler, packed wrappers)
  - `ext/parallel_diagnostics.jl`: MPI-reduced energies/spectra
  - `ext/parallel_plans.jl`: minimal plan structs
  - `ext/parallel_dispatch.jl`: routes PencilArray inputs to distributed paths
  - `ext/parallel_local.jl`: local/point evals and packed conversions

Adding a new distributed transform
1) Define the PencilArray version in `parallel_transforms.jl` using:
   - PencilFFTs to go `:φ → :k` and `:k → :φ`
   - PencilArrays.transpose to go `:k → :m` and back as needed
2) Integrate PLM-table and Robert-form branches parallel to scalar/vector implementations
3) Add a dispatch in `parallel_dispatch.jl` so PencilArrays are routed automatically

Testing
- Optional MPI tests can be run by setting `ENV["SHTNSKIT_RUN_MPI_TESTS"]=1` before running package tests.

Conventions & tips
- Use `axes(A, dim)` and `PencilArrays.globalindices(A, dim)` to iterate local/global coordinates.
- Reduce with `MPI.Allreduce!`/`MPI.Allreduce` where appropriate; avoid gathering full matrices unless required by the algorithm.
- Keep normalization/phase conversions consistent by reusing `convert_alm_norm!` helpers.

Have questions or want to contribute a feature? Start from the files above, mirror the scalar patterns, and open a PR — happy to review and iterate.
