using Test
using SHTnsKit

import SHTnsKit: on_device, _gpu_adapter_functional, _gpu_adapter_matches,
                  _gpu_adapter_adapt

mutable struct RoutingTestAdapter
    vendor::Symbol
end

struct RoutingTestArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    vendor::Symbol
end

Base.size(array::RoutingTestArray) = size(array.parent)
Base.getindex(array::RoutingTestArray, indices...) = getindex(array.parent, indices...)
on_device(::RoutingTestArray) = GPU()
_gpu_adapter_functional(::RoutingTestAdapter) = true
_gpu_adapter_matches(adapter::RoutingTestAdapter, array::RoutingTestArray) =
    adapter.vendor == array.vendor
_gpu_adapter_adapt(adapter::RoutingTestAdapter, array::AbstractArray) =
    RoutingTestArray(array, adapter.vendor)

@testset "Strict typed backend routing" begin
    cfg = create_gauss_config(3, 6; nlon=8)
    field = [sin(i / 3) + cos(j / 4) for i in 1:cfg.nlat, j in 1:cfg.nlon]
    coefficients = analysis(cfg, field)

    @test analysis(CPU(), cfg, field) ≈ coefficients
    @test synthesis(CPU(), cfg, coefficients) ≈ synthesis(cfg, coefficients)
    @test to_device(CPU(), field) === field
    @test to_device(field, CPU()) === field
    @test on_device(field) isa CPU

    old_device = SHTnsKit._DEVICE_STATE[]
    try
        SHTnsKit._DEVICE_STATE[] = nothing
        @test get_device() isa CPU
        @test_throws BackendUnavailableError set_device!(GPU())
        @test_throws BackendUnavailableError to_device(GPU(), field)
        @test_throws BackendUnavailableError analysis(GPU(), cfg, field)
        @test_throws BackendUnavailableError synthesis(GPU(), cfg, coefficients)
        @test get_device() isa CPU
    finally
        SHTnsKit._DEVICE_STATE[] = old_device
    end
end

@testset "GPU adapter selection is deterministic" begin
    saved_adapters = copy(SHTnsKit._GPU_ADAPTERS)
    saved_device = SHTnsKit._DEVICE_STATE[]
    adapter_a = RoutingTestAdapter(:a)
    adapter_b = RoutingTestAdapter(:b)
    host = reshape(collect(1:6), 2, 3)
    try
        empty!(SHTnsKit._GPU_ADAPTERS)
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
        empty!(SHTnsKit._GPU_ADAPTERS)
        merge!(SHTnsKit._GPU_ADAPTERS, saved_adapters)
        SHTnsKit._DEVICE_STATE[] = saved_device
    end
end

@testset "Device argument orders are unambiguous" begin
    ambiguities = Test.detect_ambiguities(SHTnsKit; recursive=true)
    device_ambiguities = filter(ambiguities) do methods
        any(method -> method.name == :to_device, methods)
    end
    @test isempty(device_ambiguities)
end
