using Test
using SHTnsKit

@testset "Device Utilities" begin
    @test CPU() isa ComputeDevice
    @test GPU() isa ComputeDevice
    @test CPU() == CPU()
    @test GPU() == GPU()
    @test CPU() != GPU()

    old_device = lock(SHTnsKit._DEVICE_STATE_LOCK) do
        SHTnsKit._DEVICE_STATE[]
    end
    try
        lock(SHTnsKit._DEVICE_STATE_LOCK) do
            SHTnsKit._DEVICE_STATE[] = nothing
        end

        @test get_device() == CPU()
        @test set_device!(CPU()) == CPU()
        @test get_device() == CPU()
        @test_throws MethodError set_device!(:cpu)

        @test_throws BackendUnavailableError set_device!(GPU())
        @test lock(SHTnsKit._DEVICE_STATE_LOCK) do
            SHTnsKit._DEVICE_STATE[] == CPU()
        end
    finally
        lock(SHTnsKit._DEVICE_STATE_LOCK) do
            SHTnsKit._DEVICE_STATE[] = old_device
        end
    end

    arr = rand(5, 5)
    @test to_device(arr, CPU()) === arr
    @test on_device(arr) == CPU()
    @test_throws MethodError to_device(arr, :cpu)
    @test_throws BackendUnavailableError to_device(arr, GPU())
end
