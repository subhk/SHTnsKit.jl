#=
================================================================================
SHTnsKit.jl - Fast Spherical Harmonic Transforms for Julia
================================================================================

A high-performance Julia package for computing Spherical Harmonic Transforms (SHT),
essential for applications in climate modeling, geophysics, astrophysics, and
computational physics on spherical domains.

WHAT ARE SPHERICAL HARMONICS?
-----------------------------
Spherical harmonics Y_l^m(θ,φ) are orthogonal basis functions on the sphere,
analogous to Fourier series on a circle. Any square-integrable function on
the sphere can be expanded as:

    f(θ,φ) = Σ_{l=0}^{lmax} Σ_{m=-l}^{l} a_lm * Y_l^m(θ,φ)

where:
- l is the degree (total wavenumber), l ≥ 0
- m is the order (azimuthal wavenumber), |m| ≤ l
- a_lm are the spherical harmonic coefficients
- θ is colatitude (0 at north pole, π at south pole)
- φ is longitude (0 to 2π)

KEY TRANSFORMS
--------------
- Analysis:  f(θ,φ) → a_lm   (spatial to spectral)
- Synthesis: a_lm → f(θ,φ)   (spectral to spatial)

The transforms use:
1. FFT along longitude (φ) for the azimuthal expansion
2. Legendre transform along latitude (θ) using Gauss-Legendre quadrature

PACKAGE STRUCTURE
-----------------
Core files:
- SHTnsKit.jl        : Main module, exports, and file includes
- config.jl          : SHTConfig struct and configuration functions
- legendre.jl        : Legendre polynomial computation (Plm_row!, etc.)
- core_transforms.jl : Basic analysis/synthesis implementations
- plan.jl            : SHTPlan for optimized repeated transforms

Vector transforms:
- sphtor_transforms.jl : Spheroidal/toroidal (2D) vector field transforms
- qst_transforms.jl    : QST (3D) vector field transforms

Utilities:
- gausslegendre.jl   : Gauss-Legendre quadrature points and weights
- normalization.jl   : Spherical harmonic normalization conventions
- fftutils.jl        : FFT wrappers and utilities
- rotations.jl       : Rotation of spherical harmonic coefficients

Diagnostics:
- energy_diagnostics.jl    : Energy calculations and gradients
- spectral_diagnostics.jl  : Power spectrum analysis
- vorticity_diagnostics.jl : Vorticity and enstrophy

Extensions (loaded conditionally):
- ext/SHTnsKitParallelExt.jl : MPI-distributed transforms (requires MPI.jl, PencilArrays.jl)
- ext/SHTnsKitLoopVecExt.jl  : LoopVectorization optimizations

QUICK START
-----------
```julia
using SHTnsKit

# Create configuration for lmax=64 with 96 Gauss points
cfg = create_gauss_config(64, 96)

# Create test data on the grid
f = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = acos(cfg.x[i])
    φ = 2π * (j-1) / cfg.nlon
    f[i,j] = cos(θ)  # Y_1^0 pattern
end

# Transform to spectral space
alm = analysis(cfg, f)

# Transform back to spatial space
f_reconstructed = synthesis(cfg, alm)

# Check roundtrip error
println("Max error: ", maximum(abs, f - f_reconstructed))
```

NORMALIZATION CONVENTIONS
-------------------------
Default: Orthonormal with Condon-Shortley phase
- cfg.norm = :orthonormal (default), :fourpi, :schmidt
- cfg.cs_phase = true (default) includes (-1)^m factor

COEFFICIENT STORAGE
-------------------
Dense format: Matrix of size (lmax+1, mmax+1)
- alm[l+1, m+1] for l ≥ m ≥ 0
- Only non-negative m stored (negative m via conjugate symmetry for real fields)

ENVIRONMENT VARIABLES
--------------------
- SHTNSKIT_PHI_SCALE: "dft" or "quad" for φ scaling convention
- SHTNSKIT_VERBOSE_STORAGE: "1" to print storage optimization info
- SHTNSKIT_CACHE_PENCILFFTS: "0" to disable FFT plan caching (parallel ext)

DEBUGGING TIPS
--------------
1. Check Gauss weights: sum(cfg.w) ≈ 2.0
2. Check grid points: cfg.x contains cos(θ) values, cfg.x[1] ≈ 1 (north pole vicinity)
3. Roundtrip test: f ≈ synthesis(cfg, analysis(cfg, f)) within machine precision
4. Coefficient check: alm[1,1] is the l=0,m=0 (mean) coefficient

================================================================================
=#

module SHTnsKit

# Import required standard libraries
using LinearAlgebra  # For linear algebra operations
using FFTW          # For Fast Fourier Transform operations
using Base.Threads  # For multi-threading support

# Runtime knob for inverse-FFT φ scaling during synthesis.
# Defaults: Gauss grids use "dft" (nlon), regular/equiangular use "quad" (nlon/(2π)).
# Override via cfg.phi_scale=:dft/:quad or ENV SHTNSKIT_PHI_SCALE=dft|quad.
phi_inv_scale(nlon::Integer) = (get(ENV, "SHTNSKIT_PHI_SCALE", "dft") == "quad" ? nlon/(2π) : nlon)

# Include all module source files
include("fftutils.jl")                      # FFT utility functions and helpers
include("layout.jl")                        # Data layout and memory organization
include("mathutils.jl")                      # Mathematical utility functions
include("gausslegendre.jl")                  # Gauss-Legendre quadrature implementation
include("legendre.jl")                       # Legendre polynomial computations
include("normalization.jl")                  # Spherical harmonic normalization
include("config.jl")                         # Configuration and setup functions

# Config-aware φ scaling (requires SHTConfig definition)
function phi_inv_scale(cfg::SHTConfig)
    mode = get(ENV, "SHTNSKIT_PHI_SCALE", "")
    if mode == "quad"
        return cfg.nlon / (2π)
    elseif mode == "dft"
        return cfg.nlon
    end
    if cfg.phi_scale === :quad
        return cfg.nlon / (2π)
    elseif cfg.phi_scale === :dft
        return cfg.nlon
    else
        return cfg.grid_type == :gauss ? cfg.nlon : cfg.nlon / (2π)
    end
end

include("buffer_utils.jl")                   # Common buffer allocation patterns
include("plan.jl")                           # Transform planning and optimization
include("core_transforms.jl")                # Core 2D grid ↔ spectral transforms
include("specialized_transforms.jl")         # Vector and point transforms
include("complex_packed.jl")                  # Complex number packing utilities
include("qst_transforms.jl")                  # QST (3D) vector field operations
include("sphtor_transforms.jl")               # Spheroidal/toroidal (2D) vector operations
include("operators.jl")                       # Differential operators on sphere
include("rotations.jl")                       # Spherical rotation operations
include("local.jl")                           # Local (thread-local) operations
include("energy_diagnostics.jl")              # Energy calculations and gradients
include("spectral_diagnostics.jl")            # Spectral analysis and spectrum functions
include("vorticity_diagnostics.jl")           # Vorticity and enstrophy calculations
include("api_compat.jl")                      # API compatibility layer
include("batch_transforms.jl")                # Batch (multi-field) transforms
include("parallel_dense.jl")                  # Parallel dense matrix operations
include("device_utils.jl")                    # GPU device utilities and management

# ===== CORE CONFIGURATION AND SETUP =====
export SHTConfig, create_gauss_config, create_regular_config, create_config, destroy_config  # Configuration management
export create_gauss_fly_config, set_on_the_fly!, set_use_tables!, is_on_the_fly  # On-the-fly mode
export estimate_table_memory                                          # Memory estimation
export set_south_pole_first!, set_north_pole_first!, is_south_pole_first  # South pole first mode
export create_gauss_config_spf                                        # South pole first config creation
export set_allow_padding!, disable_padding!, is_padding_enabled       # Memory padding control
export get_nlat_padded, get_spat_dist, compute_optimal_padding        # Padding queries
export allocate_padded_spatial, allocate_padded_spatial_batch         # Padded array allocation
export copy_to_padded!, copy_from_padded!, estimate_padding_overhead  # Padding utilities
export create_gauss_config_gpu, set_config_device!, get_config_device, is_gpu_config  # GPU device management
export select_compute_device, device_transfer_arrays                  # Device utilities

# ===== BASIC TRANSFORMS =====
export analysis, synthesis                              # Basic forward/backward transforms
export SHTPlan, analysis!, synthesis!                  # Planned (optimized) transforms

# ===== BATCH TRANSFORMS =====
export set_batch_size!, get_batch_size, reset_batch_size!  # Batch configuration
export analysis_batch, analysis_batch!                  # Batch scalar analysis
export synthesis_batch, synthesis_batch!                # Batch scalar synthesis
export spat_to_SHsphtor_batch, SHsphtor_to_spat_batch  # Batch vector transforms
export spat_to_SHqst_batch, SHqst_to_spat_batch        # Batch 3D vector transforms

# ===== SPATIAL ↔ SPHERICAL HARMONIC TRANSFORMS =====
export spat_to_SHsphtor!, SHsphtor_to_spat!            # In-place spheroidal/toroidal transforms
export spat_to_SH, SH_to_spat, spat_to_SH_l, SH_to_spat_l, spat_to_SH_ml, SH_to_spat_ml, SH_to_point

# ===== INDEXING AND COMPLEX NUMBER UTILITIES =====
export nlm_calc, nlm_cplx_calc, LM_index, LiM_index, im_from_lm, LM_cplx_index, LM_cplx
export spat_cplx_to_SH, SH_to_spat_cplx, SH_to_point_cplx  # Complex number transforms
export fft_phi_backend

# ===== BUFFER HELPERS =====
export scratch_fft, scratch_spatial

# ===== VECTOR FIELD TRANSFORMS =====
export spat_to_SHsphtor, SHsphtor_to_spat, SHsph_to_spat, SHtor_to_spat, SH_to_grad_spat
export divergence_from_spheroidal, divergence_from_spheroidal!, spheroidal_from_divergence, spheroidal_from_divergence!
export vorticity_from_toroidal, vorticity_from_toroidal!, toroidal_from_vorticity, toroidal_from_vorticity!
export SH_to_grad_spat_l
export spat_to_SHqst, SHqst_to_spat, spat_cplx_to_SHqst, SHqst_to_spat_cplx  # Q,S,T decomposition

# ===== LATITUDE-BAND AND M-MODE SPECIFIC TRANSFORMS =====
export SHsphtor_to_spat_l, spat_to_SHsphtor_l, SHsph_to_spat_l, SHtor_to_spat_l
export spat_to_SHsphtor_ml, SHsphtor_to_spat_ml, SHsph_to_spat_ml, SHtor_to_spat_ml
export spat_to_SHqst_l, SHqst_to_spat_l, spat_to_SHqst_ml, SHqst_to_spat_ml
export SHsphtor_to_spat_cplx, spat_cplx_to_SHsphtor, SH_to_grad_spat_ml
export suggest_pencil_grid, set_fft_plan_cache!, enable_fft_plan_cache!, disable_fft_plan_cache!, fft_plan_cache_enabled

# ===== MATRIX OPERATIONS AND DIFFERENTIAL OPERATORS =====
export mul_ct_matrix, st_dt_matrix, SH_mul_mx          # Matrix multiplication utilities
export SH_to_lat, SHqst_to_lat                         # Latitude-specific transforms

# ===== ROTATIONS =====
export SH_Zrotate                                       # Z-axis rotations
export SH_Yrotate, SH_Yrotate90, SH_Xrotate90         # Y and X axis rotations
export SHTRotation, shtns_rotation_create, shtns_rotation_destroy  # Rotation objects
export shtns_rotation_set_angles_ZYZ, shtns_rotation_set_angles_ZXZ
export shtns_rotation_wigner_d_matrix, shtns_rotation_apply_cplx, shtns_rotation_apply_real, shtns_rotation_set_angle_axis

# ===== ENERGY AND DIAGNOSTICS =====
export energy_scalar, energy_vector, enstrophy, vorticity_spectral, vorticity_grid
export grid_energy_scalar, grid_energy_vector, grid_enstrophy
export energy_scalar_l_spectrum, energy_scalar_m_spectrum    # Spectral energy analysis
export energy_vector_l_spectrum, energy_vector_m_spectrum
export enstrophy_l_spectrum, enstrophy_m_spectrum
export energy_scalar_lm, energy_vector_lm, enstrophy_lm

# ===== GRADIENT COMPUTATIONS =====
export grad_energy_scalar_alm, grad_energy_vector_Slm_Tlm, grad_enstrophy_Tlm
export grad_grid_energy_scalar_field, grad_grid_energy_vector_fields, grad_grid_enstrophy_zeta
export energy_scalar_packed, grad_energy_scalar_packed
export energy_vector_packed, grad_energy_vector_packed
export loss_vorticity_grid, grad_loss_vorticity_Tlm, loss_and_grad_vorticity_Tlm

# ===== PERFORMANCE OPTIMIZATIONS =====
export prepare_plm_tables!, enable_plm_tables!, disable_plm_tables!  # Precomputed Legendre tables

# ===== EXTENSION-PROVIDED FUNCTIONS =====
# These functions are implemented in Julia package extensions and only available when
# the corresponding packages are loaded

# GPU Computing functions (SHTnsKitGPUExt extension)
export SHTDevice, CPU_DEVICE, CUDA_DEVICE, AMDGPU_DEVICE  # Device management
export get_device, set_device!, to_device                 # Device utilities
export gpu_analysis, gpu_synthesis, gpu_analysis_safe, gpu_synthesis_safe  # GPU transforms
export gpu_spat_to_SHsphtor, gpu_SHsphtor_to_spat        # GPU vector transforms
export gpu_apply_laplacian!, gpu_legendre!               # GPU operators
export gpu_memory_info, check_gpu_memory, gpu_clear_cache!, estimate_memory_usage  # Memory management
export MultiGPUConfig, create_multi_gpu_config           # Multi-GPU configuration
export get_available_gpus, set_gpu_device                # Multi-GPU device management
export multi_gpu_analysis, multi_gpu_synthesis           # Multi-GPU transforms
export multi_gpu_analysis_streaming, multi_gpu_synthesis_streaming, estimate_streaming_chunks  # Memory streaming

# Optional LoopVectorization-powered helpers (SHTnsKitLoopVecExt extension)
export analysis_turbo, synthesis_turbo                    # Vectorized transforms
export turbo_apply_laplacian!, benchmark_turbo_vs_simd    # Performance utilities

# Automatic Differentiation wrappers (AD extensions: Zygote, ForwardDiff)
export zgrad_scalar_energy, zgrad_vector_energy, zgrad_enstrophy_Tlm      # Zygote gradients
export fdgrad_scalar_energy, fdgrad_vector_energy                         # ForwardDiff gradients
export zgrad_rotation_angles_real, zgrad_rotation_angles_cplx             # Rotation gradients

# Distributed/Parallel computing functions (SHTnsKitParallelExt extension)
export dist_analysis, dist_synthesis                      # Distributed transforms
export dist_scalar_roundtrip!, dist_vector_roundtrip!    # Distributed roundtrip tests
export DistPlan, dist_synthesis!                         # Distributed plans
export DistAnalysisPlan, dist_analysis!                  
export DistSphtorPlan, dist_spat_to_SHsphtor!, dist_SHsphtor_to_spat!  # Distributed vector transforms
export DistQstPlan, dist_spat_to_SHqst!, dist_SHqst_to_spat!           # Distributed Q,S,T transforms
export dist_SH_to_lat, dist_SH_to_point, dist_SHqst_to_point           # Distributed evaluation
export dist_spat_to_SH_packed, dist_SH_packed_to_spat                   # Distributed packed transforms
export dist_spat_cplx_to_SH, dist_SH_to_spat_cplx                      # Distributed complex transforms
export dist_SHqst_to_lat                                                # Distributed Q,S,T to latitude
export dist_SH_rotate_euler                                             # Distributed Euler rotations
export dist_spatial_divergence, dist_spatial_vorticity                  # Distributed vector invariants
export dist_scalar_laplacian, dist_scalar_laplacian!                    # Distributed scalar Laplacian
export dist_SH_Zrotate_packed, dist_SH_Yrotate_packed, dist_SH_Yrotate90_packed, dist_SH_Xrotate90_packed

# ===== EXTENSION FALLBACK FUNCTIONS =====
# These provide informative error messages when extension packages are not loaded

# Parallel extension fallbacks
_fft_plan_cache_enabled_fallback() = false
_fft_plan_cache_set_fallback(flag::Bool; clear::Bool=true) = error("Parallel extension not loaded")
_fft_plan_cache_enable_fallback() = error("Parallel extension not loaded")
_fft_plan_cache_disable_fallback(; clear::Bool=true) = error("Parallel extension not loaded")

const _fft_plan_cache_enabled_cb = Ref{Function}(_fft_plan_cache_enabled_fallback)
const _fft_plan_cache_set_cb = Ref{Function}(_fft_plan_cache_set_fallback)
const _fft_plan_cache_enable_cb = Ref{Function}(_fft_plan_cache_enable_fallback)
const _fft_plan_cache_disable_cb = Ref{Function}(_fft_plan_cache_disable_fallback)

fft_plan_cache_enabled() = _fft_plan_cache_enabled_cb[]()
set_fft_plan_cache!(flag::Bool; clear::Bool=true) = (_fft_plan_cache_set_cb[])(flag; clear=clear)
enable_fft_plan_cache!() = (_fft_plan_cache_enable_cb[])()
disable_fft_plan_cache!(; clear::Bool=true) = (_fft_plan_cache_disable_cb[])(; clear=clear)

Base.@doc """
    fft_plan_cache_enabled() -> Bool

Return whether distributed FFT plan caching is currently enabled.
""" fft_plan_cache_enabled

Base.@doc """
    set_fft_plan_cache!(flag::Bool; clear::Bool=true)

Enable or disable caching of distributed FFT plans. When disabling and `clear=true`, cached plans are freed.
""" set_fft_plan_cache!

Base.@doc """
    enable_fft_plan_cache!()

Convenience wrapper to enable distributed FFT plan caching.
""" enable_fft_plan_cache!

Base.@doc """
    disable_fft_plan_cache!(; clear::Bool=true)

Disable distributed FFT plan caching. Pass `clear=false` to retain existing cache entries.
""" disable_fft_plan_cache!

# ===== PENCIL GRID SUGGESTION =====
function _suggest_pencil_grid_fallback(comm_or_nprocs::Any, nlat::Integer, nlon::Integer;
                                       prefer_square::Bool=true,
                                       allow_one_dim::Bool=true)
    return (1, 1)
end

const _suggest_pencil_grid_cb = Ref{Function}(_suggest_pencil_grid_fallback)

function suggest_pencil_grid(comm_or_nprocs::Any, nlat::Integer, nlon::Integer;
                              prefer_square::Bool=true,
                              allow_one_dim::Bool=true)
    return (_suggest_pencil_grid_cb[])(comm_or_nprocs, Int(nlat), Int(nlon);
                                       prefer_square=prefer_square,
                                       allow_one_dim=allow_one_dim)
end

"""
    suggest_pencil_grid(comm_or_nprocs, nlat, nlon; prefer_square=true, allow_one_dim=true)

Return a suggested MPI pencil decomposition `(p_theta, p_phi)` for a grid of size
`nlat x nlon`. The base package provides a `(1,1)` fallback so the function is
always defined; when `SHTnsKitParallelExt` is loaded the callback is replaced
with an MPI-aware heuristic that favours balanced 2D decompositions.
"""
suggest_pencil_grid

# GPU extension fallbacks
get_device() = error("GPU extension not loaded. Install and load CUDA.jl or AMDGPU.jl with GPUArrays and KernelAbstractions")
set_device!(::Any) = error("GPU extension not loaded")
to_device(::Any, ::Any) = error("GPU extension not loaded")
gpu_analysis(::SHTConfig, ::Any; kwargs...) = error("GPU extension not loaded")
gpu_synthesis(::SHTConfig, ::Any; kwargs...) = error("GPU extension not loaded")
gpu_spat_to_SHsphtor(::SHTConfig, ::Any, ::Any; kwargs...) = error("GPU extension not loaded")
gpu_SHsphtor_to_spat(::SHTConfig, ::Any, ::Any; kwargs...) = error("GPU extension not loaded")
gpu_apply_laplacian!(::SHTConfig, ::Any; kwargs...) = error("GPU extension not loaded")
gpu_legendre!(::Any, ::Any, ::Any; kwargs...) = error("GPU extension not loaded")

# Default fallbacks if extensions are not loaded (use broad signatures to avoid overwriting)
zgrad_scalar_energy(::SHTConfig, ::Any) = error("Zygote extension not loaded")
zgrad_vector_energy(::SHTConfig, ::Any, ::Any) = error("Zygote extension not loaded")
zgrad_enstrophy_Tlm(::SHTConfig, ::Any) = error("Zygote extension not loaded")
fdgrad_scalar_energy(::SHTConfig, ::Any) = error("ForwardDiff extension not loaded")
fdgrad_vector_energy(::SHTConfig, ::Any, ::Any) = error("ForwardDiff extension not loaded")
zgrad_rotation_angles_real(::SHTConfig, ::Any, ::Any, ::Any, ::Any) = error("Zygote extension not loaded")
zgrad_rotation_angles_cplx(::Any, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Zygote extension not loaded")
dist_analysis(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_scalar_roundtrip!(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_vector_roundtrip!(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spat_to_SHsphtor(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SHsphtor_to_spat(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spat_to_SHqst(::SHTConfig, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SHqst_to_spat(::SHTConfig, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_analysis!(::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis!(::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spat_to_SHsphtor!(::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SHsphtor_to_spat!(::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spat_to_SHqst!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SHqst_to_spat!(::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_to_lat(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_to_point(::SHTConfig, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
dist_SHqst_to_point(::SHTConfig, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
dist_spat_to_SH_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_packed_to_spat(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spat_cplx_to_SH(::SHTConfig, ::Any) = error("Parallel extension not loaded")
dist_SH_to_spat_cplx(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SHqst_to_lat(::SHTConfig, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_rotate_euler(::SHTConfig, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
dist_SH_Zrotate_packed(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_Yrotate_packed(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_Yrotate90_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_Xrotate90_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spatial_divergence(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spatial_vorticity(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_scalar_laplacian(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_scalar_laplacian!(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_apply_laplacian!(::SHTConfig, ::Any) = error("Parallel extension not loaded")
dist_SH_mul_mx!(::SHTConfig, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")

# ===== PARALLEL ROTATION FUNCTIONS =====
# Parallel rotations fallbacks (PencilArray-based)
Dist = SHTnsKit  # Alias for distributed operations

# Non-bang (out-of-place) and in-place rotation variants
function dist_SH_Zrotate(::SHTConfig, ::Any, ::Any); error("Parallel extension not loaded"); end          # Out-of-place Z rotation
function dist_SH_Zrotate(::SHTConfig, ::Any, ::Any, ::Any); error("Parallel extension not loaded"); end   # In-place Z rotation
function dist_SH_Yrotate_allgatherm!(::SHTConfig, ::Any, ::Any, ::Any); error("Parallel extension not loaded"); end    # Y rotation with full gather
function dist_SH_Yrotate_truncgatherm!(::SHTConfig, ::Any, ::Any, ::Any); error("Parallel extension not loaded"); end # Y rotation with truncated gather
function dist_SH_Yrotate(::SHTConfig, ::Any, ::Any, ::Any); error("Parallel extension not loaded"); end              # General Y rotation
function dist_SH_Yrotate90(::SHTConfig, ::Any, ::Any); error("Parallel extension not loaded"); end                   # 90° Y rotation
function dist_SH_Xrotate90(::SHTConfig, ::Any, ::Any); error("Parallel extension not loaded"); end                   # 90° X rotation

# ===== LOOPVECTORIZATION EXTENSION FALLBACKS =====
# LoopVectorization extension fallbacks (keep methods broad so extensions can specialize)
analysis_turbo(::Any, ::Any) = error("LoopVectorization extension not loaded")                    # Vectorized analysis
synthesis_turbo(::Any, ::Any; real_output::Bool=true) = error("LoopVectorization extension not loaded")  # Vectorized synthesis
turbo_apply_laplacian!(::Any, ::Any) = error("LoopVectorization extension not loaded")            # Vectorized Laplacian
benchmark_turbo_vs_simd(::Any; kwargs...) = error("LoopVectorization extension not loaded")      # Performance comparison

# ===== LOW-LEVEL SHTNS LIBRARY INTERFACE =====
# Direct bindings to the underlying SHTns C library functions

# Grid type flags
export SHT_GAUSS, SHT_AUTO, SHT_REGULAR, SHT_REG_FAST, SHT_QUICK_INIT, SHT_REGULAR_POLES, SHT_GAUSS_FLY
export SHT_REG_DCT                                                        # DCT-based regular grid

# Data layout flags
export SHT_NATIVE_LAYOUT, SHT_THETA_CONTIGUOUS, SHT_PHI_CONTIGUOUS        # Data layout control

# Option flags
export SHT_NO_CS_PHASE, SHT_REAL_NORM, SHT_SCALAR_ONLY                    # Normalization options
export SHT_SOUTH_POLE_FIRST, SHT_ALLOW_PADDING                            # Grid ordering and padding
export SHT_LOAD_SAVE_CFG, SHT_ALLOW_GPU                                   # Config caching and GPU
export SHT_ROBERT_FORM, SHT_FP32                                          # Transform options

# Library information
export shtns_verbose, shtns_print_version, shtns_get_build_info

# Initialization and configuration
export shtns_init, shtns_create, shtns_set_grid, shtns_set_grid_auto, shtns_create_with_grid
export shtns_use_threads, shtns_reset, shtns_destroy, shtns_unset_grid, shtns_robert_form

# Spherical harmonic values and weights
export sh00_1, sh10_ct, sh11_st, shlm_e1, shtns_gauss_wts

# Debugging and Legendre functions
export shtns_print_cfg, legendre_sphPlm_array, legendre_sphPlm_deriv_array

# Memory management
export shtns_malloc, shtns_free, shtns_set_many

# Helper macros (grid coordinate utilities)
export PHI_DEG, PHI_RAD, THETA_DEG, THETA_RAD                             # Coordinate conversions
export NSPAT_ALLOC, NLM_ALLOC                                             # Allocation size helpers

# Configuration persistence
export save_config, load_config                                           # Config save/load

end # module SHTnsKit
