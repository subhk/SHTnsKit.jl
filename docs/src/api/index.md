# API Reference

Complete reference for all SHTnsKit.jl functions and types.

## Configuration Management

### Configuration Creation

```julia
create_config(lmax; mmax=lmax, mres=1, nlat=lmax+2, nlon=max(2*lmax+1,4),
              norm=:orthonormal, cs_phase=true, real_norm=false, robert_form=false,
              grid_type=:gauss, phi_scale=:dft) → SHTConfig
```
Create a configuration with specified parameters. Supports Gauss–Legendre grids (`grid_type = :gauss`, default) and regular equiangular grids (`:regular` or `:regular_poles`), forwarding to the matching constructor.

Notes:
- Auto-corrects undersized grids to satisfy accuracy constraints:
  - Gauss: `nlat ≥ lmax+1`
  - Regular: `nlat ≥ lmax+2` (or `lmax+1` with `grid_type = :regular_poles`)
  - All: `nlon ≥ 2*mmax+1`
- φ scaling defaults: Gauss grids use `phi_scale=:dft`, regular/Driscoll-Healy grids use `phi_scale=:quad`. Override per-config or via `ENV["SHTNSKIT_PHI_SCALE"]=dft|quad`.
- A legacy form `create_config(::Type{T}, lmax, nlat, mres; ...)` is also accepted; the type is ignored.

**Example:**
```julia
cfg = create_config(32; nlat=30, nlon=60)  # adjusted to nlat=33, nlon=65
```

---

```julia
create_gauss_config(lmax, nlat; mmax=lmax, mres=1, nlon=max(2*lmax+1,4),
                    norm=:orthonormal, cs_phase=true, real_norm=false,
                    robert_form=false) → SHTConfig
```
Create configuration with Gauss–Legendre grid. Requires `nlat ≥ lmax+1` and `nlon ≥ 2*mmax+1`.

**Example:**
```julia
cfg = create_gauss_config(32, 34; nlon=65)
nlat, nphi = cfg.nlat, cfg.nlon  # 34 × 65
```

---

```julia
create_regular_config(lmax, nlat; mmax=lmax, mres=1, nlon=max(2*lmax+1,4),
                      norm=:orthonormal, cs_phase=true, real_norm=false,
                      robert_form=false, include_poles=false, phi_scale=:quad,
                      precompute_plm=true) → SHTConfig
```
Create configuration with a regular equiangular grid. Set `include_poles=true` to place nodes on the poles (otherwise midpoints are used). By default Legendre tables are precomputed for faster regular-grid transforms.

**Example:**
```julia
cfg = create_regular_config(32, 36; nlon=65)
nlat, nphi = cfg.nlat, cfg.nlon  # 36 × 65
```

---

### Buffer Helpers

```julia
scratch_spatial(cfg::SHTConfig, T::Type=Float64) -> Matrix{T}
scratch_fft(cfg::SHTConfig, T::Type=ComplexF64) -> Matrix{T}
```
Allocate reusable spatial and complex FFT buffers sized to `cfg`. Pass `fft_scratch` to `analysis`/`synthesis` (and their `!` variants) to avoid per-call allocations.

### In-place Usage

- **Serial**: preallocate `alm`, `f_out`, and `fft_scratch` and call `analysis!` / `synthesis!`:
  ```julia
  fft_scratch = scratch_fft(cfg)
  alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
  analysis!(cfg, alm, f; fft_scratch=fft_scratch)
  f_out = scratch_spatial(cfg)
  synthesis!(cfg, f_out, alm; fft_scratch=fft_scratch)
  ```
- **Distributed**: build plans once with `use_rfft=true` and (for vector/QST) `with_spatial_scratch=true` to reuse internal FFT buffers:
  ```julia
  aplan = DistAnalysisPlan(cfg, proto; use_rfft=true)
  vplan = DistSphtorPlan(cfg, proto; use_rfft=true, with_spatial_scratch=true)
  splan = DistPlan(cfg, proto; use_rfft=true)
  dist_analysis!(aplan, Alm, fθφ)
  dist_SHsphtor_to_spat!(vplan, Vt, Vp, S, T; real_output=true)
  dist_synthesis!(splan, fθφ, Alm; real_output=true)
  ```

<!-- GPU configuration is not supported in this package -->

### Configuration Queries

```julia
get_lmax(cfg::SHTConfig) → Int
```
Get maximum spherical harmonic degree.

---

```julia
get_mmax(cfg::SHTConfig) → Int  
```
Get maximum spherical harmonic order.

---

```julia
get_nlat(cfg::SHTConfig) → Int
```
Get number of latitude points in spatial grid.

---

```julia
get_nphi(cfg::SHTConfig) → Int
```
Get number of longitude points in spatial grid.

---

```julia
get_nlm(cfg::SHTConfig) → Int
```
Get total number of (l,m) spectral coefficients.

For the real basis, coefficients are stored for m ≥ 0.

---

```julia
lmidx(cfg::SHTConfig, l::Int, m::Int) → Int
```
Get linear index (1‑based) for spherical harmonic coefficient (l, m≥0).

### Grid Information

```julia
get_theta(cfg::SHTConfig, i::Int) → Real
get_phi(cfg::SHTConfig, j::Int) → Real
```
Access single grid coordinates by index.

```julia
SHTnsKit.create_coordinate_matrices(cfg::SHTConfig) → (θ::Matrix, φ::Matrix)
```
Create colatitude and longitude matrices for the grid.

---

```julia
get_gauss_weights(cfg::SHTConfig) → Vector{Float64}
```
Get Gaussian quadrature weights (for Gauss grids only).

**Returns:** Vector of integration weights

### Configuration Cleanup

```julia
destroy_config(cfg::SHTConfig) → Nothing
```
Free memory associated with configuration. Always call after use.

**Example:**
```julia
cfg = create_gauss_config(32, 32)
# ... use configuration ...
destroy_config(cfg)
```

## Scalar Field Transforms

### Memory Allocation

```julia
allocate_spectral(cfg::SHTConfig) → Vector{Float64}
```
Allocate array for spectral coefficients.

**Returns:** Zero-initialized vector of length `cfg.nlm`

---

```julia
allocate_spatial(cfg::SHTConfig) → Matrix{Float64}
```
Allocate array for spatial field values.

**Returns:** Zero-initialized matrix of size `(cfg.nlat, cfg.nlon)`

### Forward Transform (Synthesis)

```julia
synthesis(cfg::SHTConfig, Alm::Matrix{ComplexF64}; real_output=true) → Matrix
```
Transform from spectral to spatial domain (spherical harmonic synthesis).

**Arguments:**
- `Alm::Matrix{ComplexF64}`: Spectral coefficients of size `(lmax+1, mmax+1)`
- `real_output::Bool`: If true (default), returns real-valued spatial field

**Returns:** Spatial field matrix `(nlat × nlon)`

**Example:**
```julia
cfg = create_gauss_config(16, 18; nlon=33)
# Create bandlimited test coefficients
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm[1, 1] = 1.0  # Y_0^0 constant term
Alm[3, 1] = 0.5  # Y_2^0 term
spatial = synthesis(cfg, Alm)  # 18×33 matrix
destroy_config(cfg)
```

---

```julia
synthesis!(cfg::SHTConfig, f_out::Matrix, Alm::Matrix; real_output=true) → Nothing
```
In-place synthesis (avoids allocation).

**Arguments:**
- `f_out::Matrix`: Output spatial field (modified)
- `Alm::Matrix{ComplexF64}`: Input spectral coefficients

### Backward Transform (Analysis)

```julia
analysis(cfg::SHTConfig, f::Matrix) → Matrix{ComplexF64}
```
Transform from spatial to spectral domain (spherical harmonic analysis).

**Arguments:**
- `f::Matrix{Float64}`: Spatial field `(nlat × nlon)`

**Returns:** Spectral coefficients matrix of size `(lmax+1, mmax+1)`

**Example:**
```julia
cfg = create_gauss_config(16, 18; nlon=33)
# Create bandlimited spatial field (smooth test function)
θ, φ = cfg.θ, cfg.φ
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    spatial[i,j] = 1.0 + 0.5 * cos(θ[i]) + 0.3 * sin(θ[i]) * cos(φ[j])
end
Alm = analysis(cfg, spatial)
destroy_config(cfg)
```

---

```julia
analysis!(cfg::SHTConfig, Alm::Matrix, f::Matrix) → Nothing
```
In-place analysis (avoids allocation).

**Arguments:**
- `Alm::Matrix{ComplexF64}`: Output spectral coefficients (modified)
- `f::Matrix{Float64}`: Input spatial field

## Complex Field Transforms

### Memory Allocation

```julia
allocate_complex_spectral(cfg::SHTConfig) → Vector{ComplexF64}
```
Allocate array for complex spectral coefficients.

---

```julia
allocate_complex_spatial(cfg::SHTConfig) → Matrix{ComplexF64}
```
Allocate array for complex spatial field values.

### Complex Transforms

```julia
SH_to_spat_cplx(cfg::SHTConfig, Alm::Matrix{ComplexF64}) → Matrix{ComplexF64}
```
Complex field synthesis (spectral to spatial for complex-valued fields).

**Example:**
```julia
cfg = create_gauss_config(16, 18; nlon=33)
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm[1, 1] = 1.0 + 0.5im
spatial_complex = SH_to_spat_cplx(cfg, Alm)
destroy_config(cfg)
```

---

```julia
spat_cplx_to_SH(cfg::SHTConfig, f::Matrix{ComplexF64}) → Matrix{ComplexF64}
```
Complex field analysis (spatial to spectral for complex-valued fields).

## Vector Field Transforms  

Vector fields on the sphere are decomposed into **spheroidal** and **toroidal** components:
- **Spheroidal**: Poloidal component (has radial component)
- **Toroidal**: Azimuthal component (purely horizontal)

### Vector Synthesis

```julia
SHsphtor_to_spat(cfg::SHTConfig, Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64}) → (Vθ::Matrix, Vφ::Matrix)
```
Synthesize vector field from spheroidal and toroidal coefficients.

**Arguments:**
- `Slm::Matrix{ComplexF64}`: Spheroidal coefficients `(lmax+1, mmax+1)`
- `Tlm::Matrix{ComplexF64}`: Toroidal coefficients `(lmax+1, mmax+1)`

**Returns:**
- `Vθ::Matrix{Float64}`: Colatitude (θ) component
- `Vφ::Matrix{Float64}`: Longitude (φ) component

**Example:**
```julia
cfg = create_gauss_config(20, 22; nlon=41)
# Create vector field coefficients
Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Slm[2, 1] = 1.0  # l=1, m=0 spheroidal mode
Tlm[3, 2] = 0.5  # l=2, m=1 toroidal mode
Vθ, Vφ = SHsphtor_to_spat(cfg, Slm, Tlm)
destroy_config(cfg)
```

### Vector Analysis

```julia
spat_to_SHsphtor(cfg::SHTConfig, Vθ::Matrix, Vφ::Matrix) → (Slm::Matrix, Tlm::Matrix)
```
Analyze vector field into spheroidal and toroidal components.

**Arguments:**
- `Vθ::Matrix{Float64}`: Colatitude (θ) component
- `Vφ::Matrix{Float64}`: Longitude (φ) component

**Returns:**
- `Slm::Matrix{ComplexF64}`: Spheroidal coefficients
- `Tlm::Matrix{ComplexF64}`: Toroidal coefficients

### Spatial Operators

```julia
SHTnsKit.spatial_derivative_phi(cfg::SHTConfig, spatial::Matrix) → Matrix
```
Exact φ‑derivative using FFT in longitude.

```julia
SHTnsKit.spatial_divergence(cfg::SHTConfig, Vθ::Matrix, Vφ::Matrix) → Matrix
SHTnsKit.spatial_vorticity(cfg::SHTConfig, Vθ::Matrix, Vφ::Matrix) → Matrix
```
Divergence and vertical vorticity of tangential vector fields on the unit sphere.

## Field Rotations

```julia
rotate_real!(cfg::SHTConfig, real_coeffs; alpha=0.0, beta=0.0, gamma=0.0) → Vector
rotate_complex!(cfg::SHTConfig, cplx_coeffs; alpha=0.0, beta=0.0, gamma=0.0) → Vector{Complex}
```
Rotate spectral coefficients in‑place using ZYZ Euler angles. For real fields, use `rotate_real!` or convert with `real_to_complex_coeffs`/`complex_to_real_coeffs`.

## Power Spectrum Analysis

```julia
power_spectrum(cfg::SHTConfig, sh::Vector) → Vector{Float64}
```
Compute spherical harmonic power spectrum.

**Arguments:**
- `sh::Vector{Float64}`: Spectral coefficients

**Returns:** Power per degree `P(l) = Σₘ |aₗᵐ|²`

**Example:**
```julia
cfg = create_gauss_config(32, 32)
# Create bandlimited test coefficients (avoids high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
power = power_spectrum(cfg, sh)  # Length lmax+1
# power[1] = l=0 power, power[2] = l=1 power, etc.
```

## Threading Control

### Thread Management

```julia
set_threading!(flag::Bool) → Bool
get_threading() → Bool
set_fft_threads(n::Integer) → Int
get_fft_threads() → Int
set_optimal_threads!() → (threads::Int, fft_threads::Int)
```
Enable/disable package parallel loops, control FFTW threads, or set a sensible configuration.

**Example:**
```julia
summary = set_optimal_threads!()
println(summary)
set_fft_threads(4)
println(get_fft_threads())
```

<!-- GPU and MPI extensions are not applicable in this pure-Julia implementation. -->

## Error Handling

### Common Errors

- **`BoundsError`**: Invalid lmax/mmax values
- **`AssertionError` / `DimensionMismatch`**: Array size mismatches

### Best Practices

```julia
# Always check array sizes
@assert length(sh) == cfg.nlm "Wrong spectral array size"
@assert size(spatial) == (cfg.nlat, cfg.nlon) "Wrong spatial array size"

# Always destroy configurations
cfg = create_gauss_config(32, 32)
# ... work with cfg ...
destroy_config(cfg)
```

## Helper Functions

lm_from_index(cfg::SHTConfig, idx::Int) → (l::Int, m::Int)

---

lmidx(cfg::SHTConfig, l::Int, m::Int) → Int

<!-- Automatic differentiation specifics are omitted; functions are pure Julia and generally AD-friendly. -->
