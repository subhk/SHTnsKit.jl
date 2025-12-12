#!/usr/bin/env julia

# Distributed FFT roundtrip with PencilArrays + FFTW
#
# Run:
#   mpiexec -n 2 julia --project=. examples/parallel_fft_roundtrip.jl
#
# Notes:
# - Uses PencilArrays for distributed array management
# - Uses FFTW for 1D FFTs along the longitude dimension (local operations)
# - This matches the approach used in SHTnsKit's parallel extension

using Random

try
    using MPI
    using PencilArrays
    using FFTW
catch e
    @error "This example requires MPI, PencilArrays, and FFTW" exception=(e, catch_backtrace())
    exit(1)
end

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const SIZE = MPI.Comm_size(COMM)

if RANK == 0
    println("Distributed FFT roundtrip with PencilArrays + FFTW ($SIZE ranks)")
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

# Safe allocation helper for PencilArrays v0.19+
function pa_zeros(::Type{T}, pen::Pencil) where {T}
    local_dims = PencilArrays.size_local(pen)
    local_data = zeros(T, local_dims...)
    return PencilArray(pen, local_data)
end

# Build a real-valued distributed field f(θ,φ)
Random.seed!(1234)
fθφ = pa_zeros(Float64, topo)

# Get local data and fill with a smooth, separable pattern
local_data = parent(fθφ)
for iθ in axes(local_data, 1), iφ in axes(local_data, 2)
    local_data[iθ, iφ] = sin(0.3 * (iθ + 1)) * cos(0.2 * (iφ + 1))
end

# Store original for comparison
fθφ_orig = copy(local_data)

# Perform FFT → IFFT roundtrip along φ (dimension 2) using FFTW on local data
function fft_roundtrip(data::Matrix{Float64})
    nlat_loc, nlon_loc = size(data)

    # Forward FFT along dimension 2 (longitude) for each row
    fft_result = Matrix{ComplexF64}(undef, nlat_loc, nlon_loc)
    for i in 1:nlat_loc
        row = data[i, :]
        fft_result[i, :] = FFTW.fft(row)
    end

    # Inverse FFT to recover original
    recovered = Matrix{Float64}(undef, nlat_loc, nlon_loc)
    for i in 1:nlat_loc
        row = fft_result[i, :]
        recovered[i, :] = real.(FFTW.ifft(row))
    end

    return recovered
end

# Real-to-complex FFT roundtrip (more efficient for real data)
function rfft_roundtrip(data::Matrix{Float64})
    nlat_loc, nlon_loc = size(data)
    nk = nlon_loc ÷ 2 + 1

    # Forward RFFT along dimension 2 (longitude) for each row
    rfft_result = Matrix{ComplexF64}(undef, nlat_loc, nk)
    for i in 1:nlat_loc
        row = data[i, :]
        rfft_result[i, :] = FFTW.rfft(row)
    end

    # Inverse RFFT to recover original
    recovered = Matrix{Float64}(undef, nlat_loc, nlon_loc)
    for i in 1:nlat_loc
        row = rfft_result[i, :]
        recovered[i, :] = FFTW.irfft(row, nlon_loc)
    end

    return recovered
end

# Test complex FFT roundtrip
fθφ_recovered_c2c = fft_roundtrip(local_data)
local_err_c2c = maximum(abs.(fθφ_recovered_c2c .- fθφ_orig))

# Test real FFT roundtrip
fθφ_recovered_r2c = rfft_roundtrip(local_data)
local_err_r2c = maximum(abs.(fθφ_recovered_r2c .- fθφ_orig))

# Reduce across ranks to get global max error
global_err_c2c = MPI.Allreduce(local_err_c2c, MPI.MAX, COMM)
global_err_r2c = MPI.Allreduce(local_err_r2c, MPI.MAX, COMM)

if RANK == 0
    println("[C2C] FFT roundtrip max error: $global_err_c2c (expected ~1e-14 to 1e-15)")
    println("[R2C] FFT roundtrip max error: $global_err_r2c (expected ~1e-14 to 1e-15)")

    if global_err_c2c < 1e-10 && global_err_r2c < 1e-10
        println("✓ FFT roundtrip test PASSED")
    else
        println("✗ FFT roundtrip test FAILED")
        exit(1)
    end
end

MPI.Barrier(COMM)
if RANK == 0
    println("Done.")
end
MPI.Finalize()
