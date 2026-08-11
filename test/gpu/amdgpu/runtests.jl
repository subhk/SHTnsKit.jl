using Test
using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
using KernelAbstractions

@testset "AMDGPU backend routing" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitAMDGPUExt)
    @test extension !== nothing

    @test which(on_device, Tuple{AMDGPU.AnyROCArray}).module === extension
    @test which(
        SHTnsKit._gpu_adapter_matches,
        Tuple{typeof(extension.AMDGPU_ADAPTER),AMDGPU.AnyROCArray},
    ).module === extension
    host_view = @view zeros(Float32, 3, 4)[:, 1:2]
    @test on_device(host_view) isa SHTnsKit.CPU
    @test !SHTnsKit._gpu_adapter_matches(extension.AMDGPU_ADAPTER, host_view)

    if !AMDGPU.functional()
        @test_skip AMDGPU.functional()
        @test get_device() isa SHTnsKit.CPU
        @test_throws SHTnsKit.BackendUnavailableError to_device(SHTnsKit.GPU(), zeros(Float32, 2, 2))
    else
        cfg = create_gauss_config(2, 3; nlon=6)
        host = reshape(Float32.(1:(cfg.nlat * cfg.nlon)), cfg.nlat, cfg.nlon)
        device = to_device(SHTnsKit.GPU(), host)

        @test device isa ROCArray
        @test on_device(device) isa SHTnsKit.GPU
        device_view = @view device[:, :]
        @test device_view isa AMDGPU.AnyROCArray
        @test on_device(device_view) isa SHTnsKit.GPU
        @test SHTnsKit._gpu_adapter_matches(extension.AMDGPU_ADAPTER, device_view)
        @test to_device(SHTnsKit.GPU(), device_view) === device_view
        @test to_device(SHTnsKit.GPU(), host, device_view) isa AMDGPU.AnyROCArray
        @test_throws ArgumentError analysis(SHTnsKit.CPU(), cfg, device_view)
        @test to_device(SHTnsKit.GPU(), host, device) isa ROCArray
        @test to_device(host, SHTnsKit.GPU(), device) isa ROCArray

        @test_throws SHTnsKit.BackendUnavailableError analysis(cfg, device_view)
        @test_throws SHTnsKit.BackendUnavailableError analysis(SHTnsKit.GPU(), cfg, device_view)
    end
end
