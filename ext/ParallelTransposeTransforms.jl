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
