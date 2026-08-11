using Test
using SHTnsKit

@testset "Device Utilities" begin
    @test CPU() isa ComputeDevice
    @test GPU() isa ComputeDevice
    @test CPU() == CPU()
    @test GPU() == GPU()
    @test CPU() != GPU()

    old_device = SHTnsKit._DEVICE_STATE[]
    old_checked = SHTnsKit._CUDA_CHECKED[]
    old_available = SHTnsKit._CUDA_AVAILABLE[]
    try
        SHTnsKit._CUDA_CHECKED[] = true
        SHTnsKit._CUDA_AVAILABLE[] = false
        SHTnsKit._DEVICE_STATE[] = nothing

        @test get_device() == CPU()
        @test set_device!(CPU()) == CPU()
        @test get_device() == CPU()
        @test_throws MethodError set_device!(:cpu)

        @test_logs (:warn, r"CUDA requested but not available") begin
            @test set_device!(GPU()) == CPU()
        end
        @test SHTnsKit._DEVICE_STATE[] == GPU()

        SHTnsKit._DEVICE_STATE[] = nothing
        SHTnsKit._CUDA_AVAILABLE[] = true
        @test get_device() == GPU()
    finally
        SHTnsKit._DEVICE_STATE[] = old_device
        SHTnsKit._CUDA_CHECKED[] = old_checked
        SHTnsKit._CUDA_AVAILABLE[] = old_available
    end

    arr = rand(5, 5)
    @test to_device(arr, CPU()) === arr
    @test on_device(arr) == CPU()
    @test_throws MethodError to_device(arr, :cpu)
    @test_throws ErrorException to_device(arr, GPU())
end
