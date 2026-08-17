using Test
using SHTnsKit

import SHTnsKit: on_device, _gpu_adapter_functional, _gpu_adapter_matches,
                  _gpu_adapter_adapt

mutable struct RoutingTestAdapter
    vendor::Symbol
end

mutable struct FailingRoutingTestAdapter
    message::String
end

struct RoutingTestArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    vendor::Symbol
end

Base.size(array::RoutingTestArray) = size(array.parent)
Base.getindex(array::RoutingTestArray, indices...) = getindex(array.parent, indices...)
on_device(::RoutingTestArray) = GPU()
_gpu_adapter_functional(::RoutingTestAdapter) = true
_gpu_adapter_functional(adapter::FailingRoutingTestAdapter) = error(adapter.message)
_gpu_adapter_matches(adapter::RoutingTestAdapter, array::RoutingTestArray) =
    adapter.vendor == array.vendor
_gpu_adapter_adapt(adapter::RoutingTestAdapter, array::AbstractArray) =
    RoutingTestArray(array, adapter.vendor)

_test_adapter_state() = lock(SHTnsKit._GPU_ADAPTERS_LOCK) do
    copy(SHTnsKit._GPU_ADAPTERS)
end

function _test_restore_adapters!(saved)
    lock(SHTnsKit._GPU_ADAPTERS_LOCK) do
        empty!(SHTnsKit._GPU_ADAPTERS)
        merge!(SHTnsKit._GPU_ADAPTERS, saved)
    end
    return nothing
end

_test_device_state() = lock(SHTnsKit._DEVICE_STATE_LOCK) do
    SHTnsKit._DEVICE_STATE[]
end

function _test_set_device_state!(device)
    lock(SHTnsKit._DEVICE_STATE_LOCK) do
        SHTnsKit._DEVICE_STATE[] = device
    end
    return device
end

@testset "Strict typed backend routing" begin
    cfg = create_gauss_config(3, 6; nlon=8)
    field = [sin(i / 3) + cos(j / 4) for i in 1:cfg.nlat, j in 1:cfg.nlon]
    coefficients = analysis(cfg, field)

    @test analysis(CPU(), cfg, field) ≈ coefficients
    @test synthesis(CPU(), cfg, coefficients) ≈ synthesis(cfg, coefficients)
    @test to_device(CPU(), field) === field
    @test to_device(field, CPU()) === field
    @test on_device(field) isa CPU

    old_device = _test_device_state()
    try
        _test_set_device_state!(nothing)
        @test get_device() isa CPU
        @test_throws BackendUnavailableError set_device!(GPU())
        @test_throws BackendUnavailableError to_device(GPU(), field)
        @test_throws BackendUnavailableError analysis(GPU(), cfg, field)
        @test_throws BackendUnavailableError synthesis(GPU(), cfg, coefficients)
        @test get_device() isa CPU
    finally
        _test_set_device_state!(old_device)
    end
end

@testset "GPU adapter selection is deterministic" begin
    saved_adapters = _test_adapter_state()
    saved_device = _test_device_state()
    adapter_a = RoutingTestAdapter(:a)
    adapter_b = RoutingTestAdapter(:b)
    host = reshape(collect(1:6), 2, 3)
    try
        _test_restore_adapters!(Dict{Symbol,WeakRef}())
        SHTnsKit._register_gpu_adapter!(:b, adapter_b)
        SHTnsKit._register_gpu_adapter!(:a, adapter_a)

        @test set_device!(GPU()) isa GPU
        @test get_device() isa GPU
        ambiguity = try
            to_device(GPU(), host)
            nothing
        catch err
            err
        end
        @test ambiguity isa ArgumentError
        @test occursin("a, b", sprint(showerror, ambiguity))
        @test occursin("prototype", sprint(showerror, ambiguity))

        prototype = RoutingTestArray(host, :a)
        placed = to_device(GPU(), host, prototype)
        @test placed isa RoutingTestArray
        @test placed.vendor == :a
        @test to_device(host, GPU(), prototype).vendor == :a

        cfg = create_gauss_config(1, 2; nlon=4)
        @test_throws ArgumentError analysis(CPU(), cfg, prototype)
    finally
        _test_restore_adapters!(saved_adapters)
        _test_set_device_state!(saved_device)
    end
end

@testset "GPU selection validates prototypes and reports probe failures" begin
    saved_adapters = _test_adapter_state()
    adapter = RoutingTestAdapter(:only)
    broken = FailingRoutingTestAdapter("runtime probe exploded")
    host = reshape(collect(1:6), 2, 3)
    cfg = create_gauss_config(1, 2; nlon=4)
    try
        _test_restore_adapters!(Dict{Symbol,WeakRef}())
        SHTnsKit._register_gpu_adapter!(:only, adapter)
        placed = to_device(GPU(), host)
        @test placed isa RoutingTestArray
        @test placed.vendor == :only
        @test_throws ArgumentError analysis(GPU(), cfg, host; prototype=host)
        @test_throws ArgumentError synthesis(GPU(), cfg, complex.(host); prototype=host)
        @test_throws ArgumentError analysis(GPU(), cfg, host; prototype=1)
        @test_throws ArgumentError to_device(GPU(), host, 1)

        _test_restore_adapters!(Dict{Symbol,WeakRef}())
        SHTnsKit._register_gpu_adapter!(:broken, broken)
        err = try
            to_device(GPU(), host)
            nothing
        catch caught
            caught
        end
        @test err isa BackendUnavailableError
        if err isa BackendUnavailableError
            message = sprint(showerror, err)
            @test occursin("broken", message)
            @test occursin("runtime probe exploded", message)
        end

        SHTnsKit._register_gpu_adapter!(:only, adapter)
        @test to_device(GPU(), host).vendor == :only
    finally
        _test_restore_adapters!(saved_adapters)
    end
end

@testset "Backend registry and device state are synchronized" begin
    locks_defined = isdefined(SHTnsKit, :_GPU_ADAPTERS_LOCK) &&
                    isdefined(SHTnsKit, :_DEVICE_STATE_LOCK)
    @test locks_defined
    if locks_defined
        adapters = [RoutingTestAdapter(Symbol(:stress_, i)) for i in 1:32]
        saved_adapters = lock(SHTnsKit._GPU_ADAPTERS_LOCK) do
            copy(SHTnsKit._GPU_ADAPTERS)
        end
        saved_device = lock(SHTnsKit._DEVICE_STATE_LOCK) do
            SHTnsKit._DEVICE_STATE[]
        end
        failures = Channel{Any}(256)
        try
            lock(SHTnsKit._GPU_ADAPTERS_LOCK) do
                empty!(SHTnsKit._GPU_ADAPTERS)
            end
            @sync for (i, adapter) in enumerate(adapters)
                Threads.@spawn begin
                    try
                        for iteration in 1:100
                            SHTnsKit._register_gpu_adapter!(Symbol(:stress_, i), adapter)
                            snapshot = SHTnsKit._registered_gpu_adapters()
                            issorted(first.(snapshot)) || error("adapter snapshot is not sorted")
                            iteration % 10 == 0 && yield()
                        end
                    catch err
                        put!(failures, err)
                    end
                end
            end
            @test isempty(failures)
            @test length(SHTnsKit._registered_gpu_adapters()) == length(adapters)

            @sync for i in 1:max(4, Threads.nthreads())
                Threads.@spawn begin
                    try
                        for iteration in 1:200
                            set_device!(isodd(i + iteration) ? CPU() : GPU())
                            get_device() isa ComputeDevice || error("invalid device snapshot")
                            iteration % 10 == 0 && yield()
                        end
                    catch err
                        put!(failures, err)
                    end
                end
            end
            @test isempty(failures)
        finally
            lock(SHTnsKit._GPU_ADAPTERS_LOCK) do
                empty!(SHTnsKit._GPU_ADAPTERS)
                merge!(SHTnsKit._GPU_ADAPTERS, saved_adapters)
            end
            lock(SHTnsKit._DEVICE_STATE_LOCK) do
                SHTnsKit._DEVICE_STATE[] = saved_device
            end
        end
    end
end

@testset "Device argument orders are unambiguous" begin
    ambiguities = Test.detect_ambiguities(SHTnsKit; recursive=true)
    device_ambiguities = filter(ambiguities) do methods
        any(method -> method.name == :to_device, methods)
    end
    @test isempty(device_ambiguities)
end
