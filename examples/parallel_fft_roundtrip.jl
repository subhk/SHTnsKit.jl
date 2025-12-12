#!/usr/bin/env julia

# Distributed FFT roundtrip with PencilArrays + PencilFFTs
#
# Run:
#   mpiexec -n 2 julia --project=. examples/parallel_fft_roundtrip.jl
#
# Notes:
# - Uses safe PencilArrays allocation (no zeros(pencil; eltype=...) calls).
# - Attempts real-to-complex (rfft/irfft) along φ; falls back to complex FFT if needed.

using Random

try
    using MPI
    using PencilArrays
    using PencilFFTs
catch e
    @error "This example requires MPI, PencilArrays, and PencilFFTs" exception=(e, catch_backtrace())
    exit(1)
end

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const SIZE = MPI.Comm_size(COMM)

if RANK == 0
    println("Distributed FFT roundtrip with PencilFFTs ($SIZE ranks)")
end

# Choose a modest 2D grid (θ × φ)
lmax = 24
nlat = lmax + 8
nlon = 2*lmax + 2

# Build a balanced 2D processor grid (pθ × pφ)
function procgrid(p)
    best = (1, p); diff = p - 1
    for d in 1:p
        p % d == 0 || continue
        d2 = div(p, d)
        if abs(d - d2) < diff
            best = (d, d2); diff = abs(d - d2)
        end
    end
    return best
end

pθ, pφ = procgrid(SIZE)
topo = PencilArrays.Pencil((nlat, nlon), (pθ, pφ), COMM)

# Safe allocation helpers for PencilArrays v0.19+
function pa_zeros(::Type{T}, pen::Pencil) where {T}
    # For PencilArrays v0.19+, use PencilArray constructor with local array
    local_dims = PencilArrays.size_local(pen)
    local_data = zeros(T, local_dims...)
    return PencilArray(pen, local_data)
end

# Build a real-valued distributed field f(θ,φ)
Random.seed!(1234)
fθφ = pa_zeros(Float64, topo)
fill!(fθφ, 0)
# Fill with a smooth, separable pattern
for iθ in axes(fθφ, 1), iφ in axes(fθφ, 2)
    fθφ[iθ, iφ] = sin(0.3 * (iθ + 1)) * cos(0.2 * (iφ + 1))
end

# Try real-to-complex roundtrip first, then fall back to complex FFTs
function r2c_roundtrip!(fθφ)
    # Plan and execute R2C FFT along φ (dimension 2)
    pr = PencilFFTs.plan_rfft(fθφ; dims=2)
    Fθk = PencilFFTs.rfft(fθφ, pr)
    # Inverse transform back to real grid
    pi = PencilFFTs.plan_irfft(Fθk; dims=2)
    fθφ2 = PencilFFTs.irfft(Fθk, pi)
    # Normalize by nlon to match FFTW conventions (ifft is unnormalized)
    fθφ2 ./= nlon
    return fθφ2
end

function c2c_roundtrip!(fθφ)
    # Promote to complex and perform complex FFT → iFFT along φ
    g = pa_zeros(ComplexF64, topo)
    for iθ in axes(fθφ, 1), iφ in axes(fθφ, 2)
        g[iθ, iφ] = complex(fθφ[iθ, iφ])
    end
    pf = PencilFFTs.plan_fft(g; dims=2)
    G = PencilFFTs.fft(g, pf)
    pb = PencilFFTs.plan_fft(G; dims=2)
    g2 = PencilFFTs.ifft(G, pb)
    g2 ./= nlon
    # Return real part to compare with original
    out = pa_zeros(Float64, topo)
    for iθ in axes(out, 1), iφ in axes(out, 2)
        out[iθ, iφ] = real(g2[iθ, iφ])
    end
    return out
end

ok_r2c = true
fθφ_rt = nothing
try
    fθφ_rt = r2c_roundtrip!(fθφ)
catch e
    ok_r2c = false
    if RANK == 0
        @warn "R2C path unavailable; falling back to complex FFT" exception=(e, catch_backtrace())
    end
end

if !ok_r2c
    fθφ_rt = c2c_roundtrip!(fθφ)
end

# Compute max error and reduce across ranks
local_err = maximum(abs.(fθφ_rt .- fθφ))
global_err = Ref(0.0)
MPI.Allreduce!(Ref(local_err), global_err, MPI.MAX, COMM)

if RANK == 0
    mode = ok_r2c ? "R2C/IR2C" : "C2C"
    println("[$mode] FFT roundtrip max error: $(global_err[]) (expected ~1e-12 to 1e-14)")
end

MPI.Barrier(COMM)
if RANK == 0
    println("Done.")
end
MPI.Finalize()

