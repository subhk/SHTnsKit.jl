using Test
using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
using KernelAbstractions

@testset "AMDGPU backend routing" begin
    @test Base.get_extension(SHTnsKit, :SHTnsKitAMDGPUExt) !== nothing

    if !AMDGPU.functional()
        @test_skip AMDGPU.functional()
        @test get_device() isa SHTnsKit.CPU
        @test_throws SHTnsKit.BackendUnavailableError to_device(SHTnsKit.GPU(), zeros(Float32, 2, 2))
    else
        host = reshape(Float32.(1:12), 3, 4)
        device = to_device(SHTnsKit.GPU(), host)

        @test device isa ROCArray
        @test on_device(device) isa SHTnsKit.GPU
        @test to_device(SHTnsKit.GPU(), host, device) isa ROCArray
        @test to_device(host, SHTnsKit.GPU(), device) isa ROCArray

        cfg = create_gauss_config(2, 3; nlon=4)
        @test_throws SHTnsKit.BackendUnavailableError analysis(cfg, device)
        @test_throws SHTnsKit.BackendUnavailableError analysis(SHTnsKit.GPU(), cfg, device)
    end
end
