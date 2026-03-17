# SHTnsKit.jl - Device Utilities Tests
# Tests for backend management, device selection, array transfer

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Device Utilities" begin
    @testset "available_backends" begin
        backends = SHTnsKit.available_backends()
        @test :cpu in backends
        @test backends isa Vector{Symbol}
    end

    @testset "current_backend defaults to CPU" begin
        # Without CUDA, should return :cpu
        backend = SHTnsKit.current_backend()
        @test backend == :cpu
    end

    @testset "set_backend!" begin
        old = SHTnsKit._BACKEND_STATE[]

        # Set to CPU
        result = SHTnsKit.set_backend!(:cpu)
        @test result == :cpu
        @test SHTnsKit.current_backend() == :cpu

        # Set to auto (without CUDA, resolves to CPU)
        SHTnsKit.set_backend!(:auto)
        @test SHTnsKit.current_backend() == :cpu

        # Invalid backend should throw
        @test_throws ArgumentError SHTnsKit.set_backend!(:invalid)

        # Restore
        SHTnsKit._BACKEND_STATE[] = old
    end

    @testset "use_gpu" begin
        old = SHTnsKit._BACKEND_STATE[]
        SHTnsKit.set_backend!(:cpu)
        @test SHTnsKit.use_gpu() == false
        SHTnsKit._BACKEND_STATE[] = old
    end

    @testset "with_backend" begin
        old = SHTnsKit._BACKEND_STATE[]
        SHTnsKit.set_backend!(:auto)

        result = SHTnsKit.with_backend(:cpu) do
            @test SHTnsKit._BACKEND_STATE[] == :cpu
            42
        end
        @test result == 42
        # Should restore original state
        @test SHTnsKit._BACKEND_STATE[] == :auto

        SHTnsKit._BACKEND_STATE[] = old
    end

    @testset "with_backend restores on error" begin
        old = SHTnsKit._BACKEND_STATE[]
        SHTnsKit.set_backend!(:auto)

        try
            SHTnsKit.with_backend(:cpu) do
                error("intentional")
            end
        catch
        end
        # Should still restore
        @test SHTnsKit._BACKEND_STATE[] == :auto

        SHTnsKit._BACKEND_STATE[] = old
    end

    @testset "reset_backend!" begin
        old = SHTnsKit._BACKEND_STATE[]
        SHTnsKit.set_backend!(:cpu)
        SHTnsKit.reset_backend!()
        @test SHTnsKit._BACKEND_STATE[] == :auto
        SHTnsKit._BACKEND_STATE[] = old
    end

    @testset "select_compute_device" begin
        device, gpu_ok = SHTnsKit.select_compute_device()
        @test device == :cpu  # Without CUDA
        @test gpu_ok == false

        device2, gpu2 = SHTnsKit.select_compute_device([:cpu])
        @test device2 == :cpu
        @test gpu2 == false
    end

    @testset "to_device CPU" begin
        arr = rand(5, 5)
        result = SHTnsKit.to_device(arr, :cpu)
        @test result === arr  # Same object (no copy for Array)

        @test_throws ArgumentError SHTnsKit.to_device(arr, :invalid)
    end

    @testset "on_device" begin
        arr = rand(3)
        @test SHTnsKit.on_device(arr) == :cpu
    end

    @testset "device_transfer_arrays" begin
        a = rand(3)
        b = rand(4)
        result = SHTnsKit.device_transfer_arrays(:cpu, a, b)
        @test length(result) == 2
        @test result[1] === a
        @test result[2] === b
    end

    @testset "device_info" begin
        info = SHTnsKit.device_info()
        @test info.backend == :cpu
        @test :cpu in info.available_backends
        @test info.gpu_available == false
        @test info.details.device_type == :cpu
        @test info.details.threads == Threads.nthreads()
    end

    @testset "dispatch_to_backend" begin
        old = SHTnsKit._BACKEND_STATE[]
        SHTnsKit.set_backend!(:cpu)

        result = SHTnsKit.dispatch_to_backend(
            () -> "cpu_result",
            () -> "gpu_result"
        )
        @test result == "cpu_result"

        SHTnsKit._BACKEND_STATE[] = old
    end

    @testset "ensure_backend_initialized" begin
        backend = SHTnsKit.ensure_backend_initialized()
        @test backend == :cpu
    end
end
