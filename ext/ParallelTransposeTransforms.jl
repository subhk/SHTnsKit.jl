#=
================================================================================
ParallelTransposeTransforms.jl — Transpose-based distributed SHT (Task 2)
================================================================================

DistTransposePlan wraps a PencilFFTs plan whose FORWARD pass performs

    rFFT(φ)  +  internal θ↔m transpose

landing data in **θ-LOCAL / m-DISTRIBUTED** layout, which is exactly what the
Legendre stage needs (no extra communication before Legendre contraction).

Data-flow overview
------------------
Analysis (spatial → spectral):
  PencilArray[φ_local, θ_dist, lev]   ← allocate_spatial(plan)
    → mul!(F_buf, fft_plan, f_spatial) → F_buf[m_dist, θ_local, lev]
    → Legendre contraction per local m → Alm[l_local, m_local, lev]

Synthesis (spectral → spatial):
  (reverse of the above, Task 3+)

Key invariants
--------------
* `plan.m_local[mi]` is the 0-based global m index this rank owns at slot mi.
* `plan.NP[mi]` is the (lmax+1, nlat) matrix P̄_l^{m_local[mi]}(cos θ_i).
* The m-distribution of `F_buf` (PencilFFTs output, decomposed on dim 1) is
  guaranteed to match the m-distribution of `spectral_pencil` (Alm, decomposed
  on dim 2) because PencilArrays uses the same block-distribution formula for
  both.  This is asserted in the constructor.
================================================================================
=#

import LinearAlgebra: mul!, ldiv!

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

"""
    DistTransposePlan

Plan for a transpose-based distributed spherical harmonic transform.

Constructed via `DistTransposePlan(cfg; comm, nlev, use_rfft)`.

# Fields
- `cfg`             : SHTConfig (replicated across ranks)
- `nlat`, `nlon`, `lmax`, `mmax`, `nlev` : grid / spectral dimensions
- `comm`            : MPI communicator
- `fft_plan`        : `PencilFFTPlan((nlon, nlat), …)` — rFFT(φ) + internal transpose
- `F_buf`           : pre-allocated output of `fft_plan` (m-distributed, θ-local)
- `spectral_pencil` : `Pencil` for Alm arrays, global `(lmax+1, mmax+1)`, m-decomposed
- `m_local`         : 0-based global m indices owned by this rank (in F-dim-1 order)
- `NP`              : `NP[mi]` = `(lmax+1, nlat)` matrix of P̄_l^{m_local[mi]}(cos θ_i)
"""
struct DistTransposePlan{TP, TFB, TSP}
    cfg           :: SHTnsKit.SHTConfig
    nlat          :: Int
    nlon          :: Int
    lmax          :: Int
    mmax          :: Int
    nlev          :: Int
    comm          :: MPI.Comm
    fft_plan      :: TP                      # PencilFFTPlan
    F_buf         :: TFB                     # allocate_output(fft_plan): m-dist/θ-local + extra (nlev,)
    spectral_pencil :: TSP                   # Pencil for Alm: global (lmax+1,mmax+1), m-dist on dim2
    m_local       :: Vector{Int}             # 0-based global m indices this rank owns
    NP            :: Vector{Matrix{Float64}} # NP[mi] = (lmax+1, nlat) normalized Legendre table
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

function SHTnsKit.DistTransposePlan(
        cfg::SHTnsKit.SHTConfig;
        comm   :: MPI.Comm = MPI.COMM_WORLD,
        nlev   :: Int      = 1,
        use_rfft :: Bool   = true)

    nlat = cfg.nlat
    nlon = cfg.nlon
    lmax = cfg.lmax
    mmax = cfg.mmax

    use_rfft || error("DistTransposePlan currently requires use_rfft=true")

    # 1. Build the PencilFFTs plan: global (φ, θ) = (nlon, nlat),
    #    rFFT on dim1 (φ), NoTransform on dim2 (θ).
    #    extra_dims=(nlev,) carries radial levels as a trailing LOCAL dimension.
    proc     = (MPI.Comm_size(comm),)
    fft_plan = PencilFFTPlan(
        (nlon, nlat),
        (Transforms.RFFT(), Transforms.NoTransform()),
        proc,
        comm;
        extra_dims = (nlev,),
    )

    # 2. Allocate the spectral output buffer: shape (m_local, θ_ALL, nlev),
    #    decomposed on dim1 (m).
    F_buf = allocate_output(fft_plan)

    # 3. Determine which m values this rank owns (from F_buf's pencil).
    #    dim1 of F_buf's pencil is the m dimension (after the internal FFT+transpose).
    mr     = range_local(pencil(F_buf))     # tuple of ranges; mr[1] = local m-range (1-based)
    # Global m index (0-based) = 1-based pencil index − 1
    all_m_0based = collect(mr[1]) .- 1
    # Keep only m within [0, mmax]; the rFFT output has nlon÷2+1 bins (0..nlon÷2).
    # For the canonical setting nlon ≥ 2*mmax+1 the rFFT bins 0..mmax are all valid.
    keep   = findall(m -> 0 <= m <= mmax, all_m_0based)
    m_local = all_m_0based[keep]

    # 4. Build the spectral Pencil for Alm: global (lmax+1, mmax+1),
    #    decomposed on dim2 (the m axis).
    #    PencilArrays uses the same block-distribution as PencilFFTs,
    #    so dim2 of this pencil matches dim1 of F_buf's pencil for 0..mmax.
    spectral_pencil = Pencil((lmax + 1, mmax + 1), (2,), comm)

    # 5. Assert alignment: the m-range from spectral_pencil (dim2, 1-based) must
    #    equal m_local (after converting to 0-based).  This catches any future
    #    divergence in distribution strategies.
    sp_mr    = range_local(spectral_pencil)       # (l-range, m-range) 1-based
    sp_m_0   = collect(sp_mr[2]) .- 1             # 0-based m from spectral_pencil
    if sp_m_0 != m_local
        error("m-distribution mismatch on rank $(MPI.Comm_rank(comm)): " *
              "F_buf m_local=$m_local  ≠  spectral_pencil m=$sp_m_0")
    end

    # 6. Pre-compute normalized Legendre tables for each local m.
    #    NP[mi][l+1, i] = P̄_l^{m_local[mi]}(cos θ_i)  for i=1..nlat, l=0..lmax
    P_buf = Vector{Float64}(undef, lmax + 1)
    NP    = Vector{Matrix{Float64}}(undef, length(m_local))
    for (mi, m) in enumerate(m_local)
        tbl = Matrix{Float64}(undef, lmax + 1, nlat)
        for i in 1:nlat
            SHTnsKit.Plm_norm_row!(P_buf, cfg.x[i], lmax, m)
            @inbounds for l in 0:lmax
                tbl[l + 1, i] = P_buf[l + 1]
            end
        end
        NP[mi] = tbl
    end

    return DistTransposePlan(
        cfg, nlat, nlon, lmax, mmax, nlev, comm,
        fft_plan, F_buf, spectral_pencil,
        m_local, NP,
    )
end

# ---------------------------------------------------------------------------
# Analysis: spatial → spectral
# ---------------------------------------------------------------------------

"""
    dist_analysis!(plan::DistTransposePlan, Alm::PencilArray, f::PencilArray) -> Alm

Distributed scalar spherical harmonic analysis using the transpose approach.

Takes a spatial field `f` (global `(nlon, nlat)`, θ-distributed, produced by
`allocate_spatial(plan)`) and writes spherical harmonic coefficients into `Alm`
(global `(lmax+1, mmax+1)`, m-distributed, produced by `allocate_spectral(plan)`).

The forward pass is:
1. `mul!(F_buf, fft_plan, f)` — rFFT(φ) + internal pencil transpose → `F_buf[mi, θ, lev]`
   where θ is now fully local on every rank and m is distributed across ranks.
2. Legendre contraction per local m:
   `Alm[l+1, mi, lev] = Σ_i w[i] · NP[mi][l+1, i] · cphi · F[mi, i, lev]`
"""
function SHTnsKit.dist_analysis!(plan::DistTransposePlan, Alm::PencilArray, f::PencilArray)
    # Step 1: rFFT(φ) + internal pencil transpose.
    # After mul!, F_buf has logical dims (m, θ) with permutation (2,1), so the
    # physical parent storage order is (θ, m, lev):
    #   parent(F_buf)[i, mi, lev]   where i=theta index, mi=m-slot index
    # This is because PencilFFTs outputs F with Permutation(2,1) (physical dim1
    # = logical dim2 = θ, physical dim2 = logical dim1 = m).
    # Single FFT + Alltoall collective for the ENTIRE nlev batch.
    # This is the key amortization: one MPI collective serves all radial levels,
    # so cost is O(1) in nlev rather than O(nlev).
    mul!(plan.F_buf, plan.fft_plan, f)

    F = parent(plan.F_buf)   # (nlat, n_m_local, nlev)  — physical storage (θ, m, lev)
    A = parent(Alm)          # (lmax+1, n_m_local, nlev)

    fill!(A, zero(eltype(A)))

    w       = plan.cfg.w
    scaleφ  = plan.cfg.cphi   # 2π/nlon — converts unnormalized rFFT sum to integral
    lmax    = plan.lmax
    nlat    = plan.nlat
    nlev    = plan.nlev

    # Step 2: Legendre contraction.
    # Access pattern: NP[mi] is (lmax+1, nlat) column-major; iterating i in the
    # inner loop walks NP columns sequentially. F[i, mi, lev] is also column-major
    # friendly with i as the fast index (dim1 of parent).
    @inbounds for lev in 1:nlev
        for (mi, m) in enumerate(plan.m_local)
            NP_mi = plan.NP[mi]          # (lmax+1, nlat) matrix for this m
            for i in 1:nlat
                wi_cphi_Fi = w[i] * scaleφ * F[i, mi, lev]
                for l in m:lmax
                    A[l+1, mi, lev] += NP_mi[l+1, i] * wi_cphi_Fi
                end
            end
        end
    end
    return Alm
end

# ---------------------------------------------------------------------------
# Synthesis: spectral → spatial
# ---------------------------------------------------------------------------

"""
    dist_synthesis!(plan::DistTransposePlan, f::PencilArray, Alm::PencilArray) -> f

Distributed scalar spherical harmonic synthesis using the transpose approach.

Takes spectral coefficients `Alm` (global `(lmax+1, mmax+1)`, m-distributed,
produced by `allocate_spectral(plan)`) and writes the reconstructed spatial
field into `f` (global `(nlon, nlat)`, θ-distributed, produced by
`allocate_spatial(plan)`).

The reverse pass is:
1. Legendre expansion per local m:
   `F[i, mi, lev] = inv_scaleφ · Σ_{l=m}^{lmax} NP[mi][l+1, i] · Alm[l+1, mi, lev]`
2. `ldiv!(f, fft_plan, F_buf)` — inverse transpose + irFFT(φ) → real spatial field.
"""
function SHTnsKit.dist_synthesis!(plan::DistTransposePlan, f::PencilArray, Alm::PencilArray)
    A = parent(Alm)           # (lmax+1, n_m_local, nlev)
    F = parent(plan.F_buf)    # (nlat, n_m_local, nlev)  — physical storage (θ fast)

    fill!(F, zero(eltype(F)))

    inv_scaleφ = SHTnsKit.phi_inv_scale(plan.cfg)
    lmax = plan.lmax
    nlat = plan.nlat
    nlev = plan.nlev

    # Legendre expansion: for each local m, sum over l → F[i, mi, lev]
    # NP[mi] is (lmax+1, nlat) column-major; iterating i (fast dim of F) is cache-friendly.
    @inbounds for lev in 1:nlev
        for (mi, m) in enumerate(plan.m_local)
            NP_mi = plan.NP[mi]          # (lmax+1, nlat) matrix for this m
            for i in 1:nlat
                acc = zero(ComplexF64)
                for l in m:lmax
                    acc += NP_mi[l+1, i] * A[l+1, mi, lev]
                end
                F[i, mi, lev] = inv_scaleφ * acc
            end
        end
    end

    # Single inverse Alltoall + irFFT for the ENTIRE nlev batch (one collective).
    ldiv!(f, plan.fft_plan, plan.F_buf)
    return f
end

# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

"""
    allocate_spatial(plan::DistTransposePlan) -> PencilArray

Return a freshly allocated real PencilArray in the spatial input layout
expected by `plan`: global `(nlon, nlat)` with θ distributed, φ local,
plus `extra_dims=(nlev,)` as a trailing local dimension.

The caller fills this array then passes it to the forward transform.
"""
SHTnsKit.allocate_spatial(plan::DistTransposePlan) = allocate_input(plan.fft_plan)

"""
    allocate_spectral(plan::DistTransposePlan) -> PencilArray

Return a freshly allocated complex PencilArray in the spectral layout:
global `(lmax+1, mmax+1)` with m distributed, l fully local, plus
`nlev` as a trailing local dimension.
"""
function SHTnsKit.allocate_spectral(plan::DistTransposePlan)
    return PencilArray{ComplexF64}(undef, plan.spectral_pencil, plan.nlev)
end
