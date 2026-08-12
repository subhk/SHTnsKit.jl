using Test
using MPI
using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using PencilArrays
using PencilFFTs

MPI.Initialized() || MPI.Init()
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "test", "parity", "mpi_gpu.jl"))

@testset "MPI + AMDGPU strict storage" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    @test extension !== nothing
    test_mpi_gpu_policy(extension)
    compound = Base.get_extension(SHTnsKit, :SHTnsKitParallelAMDGPUExt)
    test_mpi_gpu_source_contract(ROOT, :amdgpu, compound)

    if AMDGPU.functional() && MPI.Comm_size(MPI.COMM_WORLD) == 2
        pen = Pencil(ROCArray, (8, 8), (1,), MPI.COMM_WORLD)
        field = PencilArray{Float32}(undef, pen)
        @test parent(field) isa AMDGPU.AnyROCArray
        extension.allreduce!(parent(field), +, MPI.COMM_WORLD)
        @test parent(field) isa AMDGPU.AnyROCArray
    else
        @test_skip AMDGPU.functional() && MPI.Comm_size(MPI.COMM_WORLD) == 2
    end
end

MPI.Barrier(MPI.COMM_WORLD)
