using Test
using SHTnsKit
using CUDA
using GPUArrays
using GPUArraysCore
using KernelAbstractions

@testset "CUDA backend routing" begin
    @test Base.get_extension(SHTnsKit, :SHTnsKitGPUExt) !== nothing

    if !CUDA.functional()
        @test_skip CUDA.functional()
        @test get_device() isa SHTnsKit.CPU
        @test_throws SHTnsKit.BackendUnavailableError to_device(SHTnsKit.GPU(), zeros(Float32, 2, 2))
    else
        cfg = create_gauss_config(3, 6; nlon=8)
        host = Float64[sin(i / 3) + cos(j / 4) for i in 1:cfg.nlat, j in 1:cfg.nlon]
        device = to_device(SHTnsKit.GPU(), host)

        @test device isa CuArray
        @test on_device(device) isa SHTnsKit.GPU
        @test to_device(SHTnsKit.GPU(), host, device) isa CuArray
        @test to_device(host, SHTnsKit.GPU(), device) isa CuArray

        coefficients = analysis(SHTnsKit.GPU(), cfg, device)
        @test coefficients isa CuArray
        @test analysis(cfg, device) isa CuArray
        @test synthesis(SHTnsKit.GPU(), cfg, coefficients) isa CuArray
        @test synthesis(cfg, coefficients) isa CuArray
    end
end
