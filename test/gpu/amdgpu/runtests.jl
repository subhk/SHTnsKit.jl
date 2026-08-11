using Test
using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
using KernelAbstractions

include("../../parity/scalar_full.jl")

struct AMDGPUScalarAdapter <: ScalarParityAdapter end
function place(::AMDGPUScalarAdapter, ::SHTConfig, value, ::Symbol)
    if value isa PermutedDimsArray
        padded = zeros(eltype(value), size(value, 1), 2size(value, 2))
        @views padded[:, 1:2:end] .= value
        storage = ROCArray(padded)
        return @view storage[:, 1:2:end]
    end
    return ROCArray(value)
end
collect_result(::AMDGPUScalarAdapter, value, ::SHTConfig) = Array(value)
analysis_call(::AMDGPUScalarAdapter, cfg, field) = analysis(GPU(), cfg, field)
synthesis_call(::AMDGPUScalarAdapter, cfg, coefficients, _prototype; real_output) =
    synthesis(GPU(), cfg, coefficients; real_output)
assert_resident(::AMDGPUScalarAdapter, value) = @test value isa AMDGPU.AnyROCArray

@testset "AMDGPU backend routing" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitAMDGPUExt)
    @test extension !== nothing
    @test isdefined(extension.GPUCommon, :scalar_analysis_kernel!)
    @test isdefined(extension.GPUCommon, :scalar_synthesis_kernel!)
    @test isdefined(extension.GPUCommon, :coefficient_conversion_kernel!)
    @test isdefined(extension, :_amdgpu_scalar_analysis)
    @test isdefined(extension, :_amdgpu_scalar_synthesis)
    run_shared_scalar_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    source = read(joinpath(@__DIR__, "../../../ext/SHTnsKitAMDGPUExt.jl"), String)
    ordinary_pipeline = split(
        split(source, "function _amdgpu_scalar_analysis"; limit=2)[2],
        "function _gpu_adapter_analysis"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", ordinary_pipeline)
    @test !occursin(r"\bcollect\s*\(", ordinary_pipeline)

    @test which(on_device, Tuple{AMDGPU.AnyROCArray}).module === extension
    @test which(
        SHTnsKit._gpu_adapter_matches,
        Tuple{typeof(extension.AMDGPU_ADAPTER),AMDGPU.AnyROCArray},
    ).module === extension
    host_view = @view zeros(Float32, 3, 4)[:, 1:2]
    @test on_device(host_view) isa SHTnsKit.CPU
    @test !SHTnsKit._gpu_adapter_matches(extension.AMDGPU_ADAPTER, host_view)

    if !AMDGPU.functional() || !AMDGPU.functional(:rocfft)
        @test_skip AMDGPU.functional() && AMDGPU.functional(:rocfft)
        if !AMDGPU.functional()
            @test get_device() isa SHTnsKit.CPU
            @test_throws SHTnsKit.BackendUnavailableError to_device(
                SHTnsKit.GPU(), zeros(Float32, 2, 2),
            )
        else
            cfg = create_gauss_config(2, 3; nlon=6)
            field = ROCArray(zeros(Float32, cfg.nlat, cfg.nlon))
            @test_throws SHTnsKit.BackendUnavailableError analysis(cfg, field)
        end
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

        @test analysis(cfg, device_view) isa AMDGPU.AnyROCArray
        @test analysis(SHTnsKit.GPU(), cfg, device_view) isa AMDGPU.AnyROCArray

        AMDGPU.allowscalar(false)
        run_scalar_full_parity(
            AMDGPUScalarAdapter();
            grid_kinds=_SCALAR_GRID_KINDS,
            precisions=(Float32, Float64),
            mres_values=(1, 2),
            norms=(:orthonormal, :fourpi, :schmidt),
            real_norm_values=(false, true),
            cs_phase_values=(false, true),
            pole_orders=(false, true),
        )
    end
end
