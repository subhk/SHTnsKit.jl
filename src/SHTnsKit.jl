
# Julia package for fast spherical harmonic transforms using the SHTns library
# SHTns (Spherical Harmonic Transform numerical software) provides efficient 
# computation of Spherical Harmonic Transforms for scientific computing applications
module SHTnsKit

# Import required standard libraries
using LinearAlgebra  # For linear algebra operations
using FFTW          # For Fast Fourier Transform operations
using Base.Threads  # For multi-threading support

# Runtime knob for inverse-FFT φ scaling during synthesis.
# Set ENV SHTNSKIT_PHI_SCALE to "quad" to use nlon/(2π) to match φ quadrature,
# otherwise default to "dft" which uses nlon to cancel FFT's 1/n.
phi_inv_scale(nlon::Integer) = (get(ENV, "SHTNSKIT_PHI_SCALE", "dft") == "quad" ? nlon/(2π) : nlon)

# Include all module source files
include("fftutils.jl")      # FFT utility functions and helpers
include("layout.jl")        # Data layout and memory organization
include("mathutils.jl")     # Mathematical utility functions
include("gausslegendre.jl") # Gauss-Legendre quadrature implementation
include("legendre.jl")      # Legendre polynomial computations
include("normalization.jl") # Spherical harmonic normalization
include("config.jl")        # Configuration and setup functions
include("plan.jl")          # Transform planning and optimization
include("transform.jl")     # Core transform implementations
include("complex_packed.jl")# Complex number packing utilities
include("vector.jl")        # Vector field operations
include("operators.jl")     # Differential operators on sphere
include("rotations.jl")     # Spherical rotation operations
include("local.jl")         # Local (thread-local) operations
include("diagnostics.jl")   # Diagnostic and analysis tools
include("api_compat.jl")    # API compatibility layer
include("parallel_dense.jl")# Parallel dense matrix operations

# ===== CORE CONFIGURATION AND SETUP =====
export SHTConfig, create_gauss_config, create_config, destroy_config  # Configuration management

# ===== BASIC TRANSFORMS =====
export analysis, synthesis                              # Basic forward/backward transforms
export SHTPlan, analysis!, synthesis!                  # Planned (optimized) transforms

# ===== SPATIAL ↔ SPHERICAL HARMONIC TRANSFORMS =====
export spat_to_SHsphtor!, SHsphtor_to_spat!            # In-place spheroidal/toroidal transforms
export spat_to_SH, SH_to_spat, spat_to_SH_l, SH_to_spat_l, spat_to_SH_ml, SH_to_spat_ml, SH_to_point

# ===== INDEXING AND COMPLEX NUMBER UTILITIES =====
export nlm_calc, nlm_cplx_calc, LM_index, LiM_index, im_from_lm, LM_cplx_index
export spat_cplx_to_SH, SH_to_spat_cplx, SH_to_point_cplx  # Complex number transforms
export fft_phi_backend

# ===== VECTOR FIELD TRANSFORMS =====
export spat_to_SHsphtor, SHsphtor_to_spat, SHsph_to_spat, SHtor_to_spat, SH_to_grad_spat
export spat_to_SHqst, SHqst_to_spat, spat_cplx_to_SHqst, SHqst_to_spat_cplx  # Q,S,T decomposition

# ===== LATITUDE-BAND AND M-MODE SPECIFIC TRANSFORMS =====
export SHsphtor_to_spat_l, spat_to_SHsphtor_l, SHsph_to_spat_l, SHtor_to_spat_l
export spat_to_SHsphtor_ml, SHsphtor_to_spat_ml
export spat_to_SHqst_l, SHqst_to_spat_l, spat_to_SHqst_ml, SHqst_to_spat_ml
export SHsphtor_to_spat_cplx, spat_cplx_to_SHsphtor

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
export dist_SH_Zrotate_packed, dist_SH_Yrotate_packed, dist_SH_Yrotate90_packed, dist_SH_Xrotate90_packed

# ===== EXTENSION FALLBACK FUNCTIONS =====
# These provide informative error messages when extension packages are not loaded
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
# LoopVectorization extension fallbacks (broad signatures to avoid overwriting)
analysis_turbo(::SHTConfig, ::Any) = error("LoopVectorization extension not loaded")                    # Vectorized analysis
synthesis_turbo(::SHTConfig, ::Any; real_output::Bool=true) = error("LoopVectorization extension not loaded")  # Vectorized synthesis
turbo_apply_laplacian!(::SHTConfig, ::Any) = error("LoopVectorization extension not loaded")            # Vectorized Laplacian
benchmark_turbo_vs_simd(::SHTConfig; kwargs...) = error("LoopVectorization extension not loaded")      # Performance comparison
# ===== LOW-LEVEL SHTNS LIBRARY INTERFACE =====
# Direct bindings to the underlying SHTns C library functions
export shtns_verbose, shtns_print_version, shtns_get_build_info           # Library information
export shtns_init, shtns_create, shtns_set_grid, shtns_set_grid_auto, shtns_create_with_grid  # Initialization
export shtns_use_threads, shtns_reset, shtns_destroy, shtns_unset_grid, shtns_robert_form     # Configuration
export sh00_1, sh10_ct, sh11_st, shlm_e1, shtns_gauss_wts               # Spherical harmonic values and weights
export shtns_print_cfg, legendre_sphPlm_array, legendre_sphPlm_deriv_array  # Debugging and Legendre functions
export shtns_malloc, shtns_free, shtns_set_many                          # Memory management

end # module SHTnsKit
