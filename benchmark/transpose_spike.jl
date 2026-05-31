# =============================================================================
# VALIDATED transpose-based distributed-SHT data choreography (Task 1 spike)
# PencilArrays 0.19.10, PencilFFTs 0.15.3, MPI.jl 0.20.26, Julia 1.11.1
#
# KEY FACT about PencilFFTs: an N-dim global array uses an (N-1)-dim MPI
# decomposition. You CANNOT keep both (θ,φ) distributed through one plan.
# PencilFFTs transforms dim 1 first (must be local), then transposes so dim 2
# is local, etc. So the FFT'd (frequency/m) dimension must be dim 1 of the
# global array, and the forward plan ITSELF performs the transpose that lands
# data in the θ-local / m-distributed layout.
#
# ----------------------------- 2D WORKING CONFIG -----------------------------
#   nlat, nlon = 24, 48                     # θ, φ
#   proc = (nproc,)                         # 1D process grid (M = N-1 = 1)
#   transforms = (Transforms.RFFT(), Transforms.NoTransform())  # RFFT on dim1=φ
#   plan = PencilFFTPlan((nlon, nlat), transforms, proc, comm)  # global dims (φ, θ)
#   f = allocate_input(plan)               # REAL space: decomp=(2,) -> θ DIST, φ LOCAL
#   F = allocate_output(plan)              # SPECTRAL : decomp=(1,) perm=(2,1) -> m DIST, θ LOCAL
#   mul!(F, plan, f)                       # forward rFFT(φ) + internal transpose to θ-local
#   ldiv!(f, plan, F)                      # inverse: θ-local/m-dist -> θ-dist/φ-local + irFFT
#
#   # EXPLICIT transpose! between sibling pencils (for the Legendre stage, which
#   # needs to move data between m-distributed and θ-distributed layouts WITHOUT
#   # an FFT). Build the sibling from an existing Pencil so they share topology:
#   pen_F     = pencil(F)                                  # m-DIST, θ-LOCAL (perm (2,1))
#   pen_theta = Pencil(pen_F; decomp_dims=(2,), permute=Permutation(2,1)) # θ-DIST, m-LOCAL
#   Gt = PencilArray{eltype(F)}(undef, pen_theta)
#   transpose!(Gt, F)        # m-dist/θ-local  ->  θ-dist/m-local
#   transpose!(F,  Gt)       # and back
#
# --------------------------- BATCHED (3D) CONFIG -----------------------------
#   nlev, nlat, nlon = 3, 24, 48
#   # lev is a LEADING NON-TRANSFORMED, NON-DECOMPOSED dim via extra_dims:
#   transforms3 = (Transforms.RFFT(), Transforms.NoTransform())  # over (φ, θ)
#   plan3 = PencilFFTPlan((nlon, nlat), transforms3, proc, comm; extra_dims=(nlev,))
#   f3 = allocate_input(plan3)   # parent dims (φ_local, θ_dist..., nlev)  lev & θ local-batched
#   F3 = allocate_output(plan3)  # m DIST, θ LOCAL, lev LOCAL
#   mul!(F3, plan3, f3); ldiv!(f3, plan3, F3)
#   # NOTE: extra_dims appear as TRAILING dims of parent(); logical SHT dims are (φ/m, θ).
# =============================================================================

using MPI; MPI.Init()
using PencilArrays, PencilFFTs, MPI, LinearAlgebra, Random
using PencilFFTs: Transforms
comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm); nproc = MPI.Comm_size(comm)

# ----------------------------------------------------------------------------
# 2D round trip
# ----------------------------------------------------------------------------
nlat, nlon = 24, 48                                  # θ, φ
proc = (nproc,)                                      # 1D process grid
tr = (Transforms.RFFT(), Transforms.NoTransform())   # rFFT along dim1 = φ
plan = PencilFFTPlan((nlon, nlat), tr, proc, comm)   # global (φ, θ)

f = allocate_input(plan); rand!(parent(f))           # real space: θ DIST, φ LOCAL
fcopy = copy(f)
F = allocate_output(plan)                            # spectral: m DIST, θ LOCAL
mul!(F, plan, f)                                     # rFFT(φ) + internal transpose

if rank == 0
    println("[2D] input  pencil decomp=", decomposition(pencil(f)),
            " perm=", permutation(pencil(f)), " global=", size_global(pencil(f)))
    println("[2D] output pencil decomp=", decomposition(pencil(F)),
            " perm=", permutation(pencil(F)), " global=", size_global(pencil(F)))
    println("[2D] spectral parent size=", size(parent(F)),
            " logical size_local=", size_local(F), " (logical dims = (m, θ))")
end

# θ must be LOCAL in the spectral layout (logical dim 2 == nlat, fully local).
@assert size_local(F)[2] == nlat "theta not local in spectral layout"
# m is the distributed dim (logical dim 1).
@assert decomposition(pencil(F)) == (1,) "m (dim1) not the distributed dim"

# EXPLICIT transpose! between sibling pencils (no FFT): m-dist/θ-local <-> θ-dist/m-local
pen_F     = pencil(F)
pen_theta = Pencil(pen_F; decomp_dims=(2,), permute=Permutation(2, 1))
Gt = PencilArray{eltype(F)}(undef, pen_theta)
transpose!(Gt, F)                                    # -> θ DIST, m LOCAL
if rank == 0
    println("[2D] after transpose! Gt decomp=", decomposition(pencil(Gt)),
            " parent size=", size(parent(Gt)), " logical size_local=", size_local(Gt))
end
@assert size_local(Gt)[1] == 25 "m not local after transpose to theta-dist"  # nlon/2+1 = 25
Fback = allocate_output(plan)
transpose!(Fback, Gt)                                # back to m DIST, θ LOCAL
terr = MPI.Allreduce(maximum(abs.(parent(Fback) .- parent(F))), MPI.MAX, comm)
rank == 0 && println("[2D] transpose! round-trip err=", terr)
@assert terr == 0.0 "explicit transpose! round-trip not exact"

# Full FFT round trip back to real space
f2 = allocate_input(plan)
ldiv!(f2, plan, F)
gerr = MPI.Allreduce(maximum(abs.(parent(f2) .- parent(fcopy))), MPI.MAX, comm)
rank == 0 && println("[2D] FFT round-trip err=", gerr)
@assert gerr < 1e-12
rank == 0 && println("SPIKE-2D OK")

# ----------------------------------------------------------------------------
# Batched (3D) round trip — leading radial-level dimension
# ----------------------------------------------------------------------------
nlev = 3
tr3  = (Transforms.RFFT(), Transforms.NoTransform())                  # over (φ, θ)
plan3 = PencilFFTPlan((nlon, nlat), tr3, proc, comm; extra_dims=(nlev,))
f3 = allocate_input(plan3); rand!(parent(f3))
f3copy = copy(f3)
F3 = allocate_output(plan3)
mul!(F3, plan3, f3)

if rank == 0
    println("[3D] input  parent size=", size(parent(f3)), " extra_dims=", extra_dims(f3))
    println("[3D] output pencil decomp=", decomposition(pencil(F3)),
            " perm=", permutation(pencil(F3)), " global=", size_global(pencil(F3)))
    println("[3D] spectral parent size=", size(parent(F3)),
            " logical size_local=", size_local(F3), " extra_dims=", extra_dims(F3))
end

# θ LOCAL, lev LOCAL, m DISTRIBUTED in spectral layout.
@assert size_local(F3)[2] == nlat "theta not local in batched spectral layout"
@assert extra_dims(F3) == (nlev,) "lev not carried as a local extra dim"
@assert decomposition(pencil(F3)) == (1,) "m (dim1) not distributed in batched layout"

# explicit transpose! also works on batched arrays (extra dims ride along)
pen_F3    = pencil(F3)
pen_th3   = Pencil(pen_F3; decomp_dims=(2,), permute=Permutation(2, 1))
Gt3 = PencilArray{eltype(F3)}(undef, pen_th3, extra_dims(F3)...)
transpose!(Gt3, F3)
F3back = allocate_output(plan3)
transpose!(F3back, Gt3)
t3err = MPI.Allreduce(maximum(abs.(parent(F3back) .- parent(F3))), MPI.MAX, comm)
rank == 0 && println("[3D] transpose! round-trip err=", t3err)
@assert t3err == 0.0

f3b = allocate_input(plan3)
ldiv!(f3b, plan3, F3)
g3err = MPI.Allreduce(maximum(abs.(parent(f3b) .- parent(f3copy))), MPI.MAX, comm)
rank == 0 && println("[3D] FFT round-trip err=", g3err)
@assert g3err < 1e-12
rank == 0 && println("SPIKE-3D OK")

MPI.Barrier(comm)
MPI.Finalize()
