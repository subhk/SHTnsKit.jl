using Test
using MPI
using SHTnsKit
using CUDA
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using PencilArrays
using PencilFFTs

MPI.Initialized() || MPI.Init()
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "test", "parity", "mpi_gpu.jl"))

@testset "MPI + CUDA strict storage" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    @test extension !== nothing
    test_mpi_gpu_policy(extension)
    compound = Base.get_extension(SHTnsKit, :SHTnsKitParallelCUDAExt)
    test_mpi_gpu_source_contract(ROOT, :cuda, compound)

    if CUDA.functional() && MPI.Comm_size(MPI.COMM_WORLD) == 2
        pen = Pencil(CuArray, (8, 8), (1,), MPI.COMM_WORLD)
        field = PencilArray{Float32}(undef, pen)
        @test parent(field) isa CUDA.AnyCuArray
        extension.allreduce!(parent(field), +, MPI.COMM_WORLD)
        @test parent(field) isa CUDA.AnyCuArray
    else
        @test_skip CUDA.functional() && MPI.Comm_size(MPI.COMM_WORLD) == 2
    end
end

MPI.Barrier(MPI.COMM_WORLD)
