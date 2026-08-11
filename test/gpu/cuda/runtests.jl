using Test
using SHTnsKit
using CUDA
using GPUArrays
using GPUArraysCore
using KernelAbstractions

include("../../parity/scalar_full.jl")
include("../../parity/scalar_variants.jl")

struct CUDAScalarAdapter <: ScalarParityAdapter end
function place(::CUDAScalarAdapter, ::SHTConfig, value, ::Symbol)
    if value isa PermutedDimsArray
        padded = zeros(eltype(value), size(value, 1), 2size(value, 2))
        @views padded[:, 1:2:end] .= value
        storage = CuArray(padded)
        return @view storage[:, 1:2:end]
    end
    return CuArray(value)
end
collect_result(::CUDAScalarAdapter, value, ::SHTConfig) = Array(value)
analysis_call(::CUDAScalarAdapter, cfg, field; use_rfft=false) =
    analysis(GPU(), cfg, field; use_rfft)
synthesis_call(::CUDAScalarAdapter, cfg, coefficients, _prototype;
               real_output, use_rfft=false) =
    synthesis(GPU(), cfg, coefficients; real_output, use_rfft)
synthesis_cplx_call(::CUDAScalarAdapter, cfg, coefficients, _prototype) =
    synthesis_cplx(cfg, coefficients)
assert_resident(::CUDAScalarAdapter, value) = @test value isa CUDA.AnyCuArray
function assert_warm_device_noalloc(::CUDAScalarAdapter, call)
    call()
    CUDA.synchronize()
    @test CUDA.@allocated(begin
        call()
        CUDA.synchronize()
    end) == 0
    return nothing
end

struct SafeFallbackRedispatchError <: Exception end

struct SafeFallbackArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
end

Base.size(array::SafeFallbackArray) = size(array.parent)
Base.getindex(array::SafeFallbackArray, indices...) = getindex(array.parent, indices...)
SHTnsKit.on_device(::SafeFallbackArray) = SHTnsKit.GPU()
SHTnsKit.analysis(::SHTConfig, ::SafeFallbackArray; kwargs...) =
    throw(SafeFallbackRedispatchError())
SHTnsKit.synthesis(::SHTConfig, ::SafeFallbackArray; kwargs...) =
    throw(SafeFallbackRedispatchError())

@testset "CUDA backend routing" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitGPUExt)
    @test extension !== nothing
    @test isdefined(extension.GPUCommon, :scalar_analysis_kernel!)
    @test isdefined(extension.GPUCommon, :scalar_synthesis_kernel!)
    @test isdefined(extension.GPUCommon, :coefficient_conversion_kernel!)
    @test isdefined(extension.GPUCommon, :coefficient_batch_conversion_kernel!)
    @test isdefined(extension.GPUCommon, :ScalarWorkspaceCache)
    @test isdefined(extension, :_cuda_scalar_analysis)
    @test isdefined(extension, :_cuda_scalar_synthesis)
    @test isdefined(extension, :_cuda_clear_scalar_cache!)
    @test isdefined(extension, :_cuda_scalar_analysis_direct!)
    @test isdefined(extension, :_cuda_batch_analysis_direct!)
    for (function_name, signature) in (
        (:analysis_packed, Tuple{SHTConfig,CuArray{Float32,1}}),
        (:synthesis_packed, Tuple{SHTConfig,CuArray{ComplexF32,1}}),
        (:analysis_packed_l, Tuple{SHTConfig,CuArray{Float32,1},Int}),
        (:synthesis_packed_l, Tuple{SHTConfig,CuArray{ComplexF32,1},Int}),
        (:analysis_axisym, Tuple{SHTConfig,CuArray{Float32,1}}),
        (:synthesis_axisym, Tuple{SHTConfig,CuArray{ComplexF32,1}}),
        (:analysis_axisym_l, Tuple{SHTConfig,CuArray{Float32,1},Int}),
        (:synthesis_axisym_l, Tuple{SHTConfig,CuArray{ComplexF32,1},Int}),
        (:analysis_packed_ml, Tuple{SHTConfig,Int,CuArray{ComplexF32,1},Int}),
        (:synthesis_packed_ml, Tuple{SHTConfig,Int,CuArray{ComplexF32,1},Int}),
        (:analysis_packed_cplx, Tuple{SHTConfig,CuArray{ComplexF32,2}}),
        (:synthesis_packed_cplx, Tuple{SHTConfig,CuArray{ComplexF32,1}}),
        (:analysis_packed_cplx_l, Tuple{SHTConfig,CuArray{ComplexF32,2},Int}),
        (:synthesis_packed_cplx_l, Tuple{SHTConfig,CuArray{ComplexF32,1},Int}),
        (:analysis_batch, Tuple{SHTConfig,CuArray{Float32,3}}),
        (:analysis_batch!, Tuple{SHTConfig,CuArray{ComplexF32,3},CuArray{Float32,3}}),
        (:synthesis_batch, Tuple{SHTConfig,CuArray{ComplexF32,3}}),
        (:synthesis_batch!, Tuple{SHTConfig,CuArray{Float32,3},CuArray{ComplexF32,3}}),
        (:synthesis_batch_cplx, Tuple{SHTConfig,CuArray{ComplexF32,3}}),
    )
        @test which(getproperty(SHTnsKit, function_name), signature).module === extension
    end
    @test which(
        analysis!, Tuple{SHTPlan,CuArray{ComplexF32,2},CuArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis!, Tuple{SHTPlan,CuArray{Float32,2},CuArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis!,
        Tuple{SHTnsKit.GPU,SHTPlan,CuArray{ComplexF32,2},CuArray{Float32,2}},
    ).module === extension
    @test which(gpu_clear_cache!, Tuple{SHTnsKit.GPU}).module === SHTnsKit
    @test hasmethod(
        SHTnsKit._gpu_adapter_clear_cache!, Tuple{typeof(extension.CUDA_ADAPTER)},
    )
    cache_device = typemax(Int)
    extension.GPUCommon.scalar_cache_insert!(
        extension._CUDA_SCALAR_CACHE, cache_device, UInt(1), Float32, UInt(1), :sentinel,
    )
    extension._cuda_clear_scalar_cache!(; device=cache_device)
    @test extension.GPUCommon.scalar_cache_size(
        extension._CUDA_SCALAR_CACHE; device=cache_device,
    ) == 0
    workspace_cache = extension.GPUCommon.ScalarWorkspaceCache(2)
    workspace_owner = Ref(:owner)
    builds = Ref(0)
    builder = () -> (builds[] += 1; :workspace)
    use_workspace = value -> value
    @test extension.GPUCommon.scalar_workspace_use!(
        use_workspace, builder, workspace_cache, :mock, workspace_owner,
        Float32, :scalar, (1,), UInt(1),
    ) === :workspace
    @test extension.GPUCommon.scalar_workspace_use!(
        use_workspace, builder, workspace_cache, :mock, workspace_owner,
        Float32, :scalar, (1,), UInt(1),
    ) === :workspace
    @test builds[] == 1
    extension.GPUCommon.scalar_workspace_clear!(workspace_cache; device=:mock)
    @test extension.GPUCommon.scalar_workspace_size(workspace_cache) == 0
    @test which(
        synthesis_cplx, Tuple{SHTConfig,CuArray{ComplexF32,2}},
    ).module === extension
    @test which(
        synthesis_cplx, Tuple{SHTnsKit.GPU,SHTConfig,CuArray{ComplexF32,2}},
    ).module === SHTnsKit
    run_shared_scalar_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_scalar_variant_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    source = read(joinpath(@__DIR__, "../../../ext/SHTnsKitGPUExt.jl"), String)
    ordinary_pipeline = split(
        split(source, "function _cuda_scalar_analysis"; limit=2)[2],
        "function _gpu_adapter_analysis"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", ordinary_pipeline)
    @test !occursin(r"\bcollect\s*\(", ordinary_pipeline)
    @test occursin("use_rfft=true requires a real-valued input", ordinary_pipeline)
    @test occursin("use_rfft=true implies real_output", ordinary_pipeline)
    variant_pipeline = split(
        split(source, "@inline function _cuda_lcap"; limit=2)[2],
        "\"\"\"\n    _to_gpu_impl"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", variant_pipeline)
    @test !occursin(r"\bcollect\s*\(", variant_pipeline)
    @test occursin("ScalarWorkspaceCache(8)", source)
    @test occursin("_cuda_batch_analysis_direct!", source)
    @test occursin("_cuda_batch_synthesis_direct!", source)
    @test occursin("CUFFT.plan_rfft", source)
    @test !occursin(
        r"result\s*=\s*_cuda_(?:scalar_analysis|scalar_synthesis|batch_analysis|batch_synthesis)\(",
        source,
    )

    @test which(on_device, Tuple{CUDA.AnyCuArray}).module === extension
    @test which(
        SHTnsKit._gpu_adapter_matches,
        Tuple{typeof(extension.CUDA_ADAPTER),CUDA.AnyCuArray},
    ).module === extension
    host_view = @view zeros(Float32, 3, 4)[:, 1:2]
    @test on_device(host_view) isa SHTnsKit.CPU
    @test !SHTnsKit._gpu_adapter_matches(extension.CUDA_ADAPTER, host_view)

    if !CUDA.functional()
        @test_skip CUDA.functional()
        @test get_device() isa SHTnsKit.CPU
        @test_throws SHTnsKit.BackendUnavailableError to_device(SHTnsKit.GPU(), zeros(Float32, 2, 2))

        cfg = create_gauss_config(2, 3; nlon=6)
        field = Float64[sin(i / 3) + cos(j / 4) for i in 1:cfg.nlat, j in 1:cfg.nlon]
        coefficients = analysis(cfg, field)
        for (operation, call) in (
            (:gpu_analysis, () -> gpu_analysis(cfg, field)),
            (:gpu_analysis, () -> gpu_analysis(cfg, field; device=SHTnsKit.CPU())),
            (:gpu_synthesis, () -> gpu_synthesis(cfg, coefficients)),
            (:gpu_analysis_sphtor, () -> gpu_analysis_sphtor(cfg, field, field)),
            (:gpu_synthesis_sphtor, () -> gpu_synthesis_sphtor(cfg, coefficients, coefficients)),
            (:gpu_apply_laplacian!, () -> gpu_apply_laplacian!(cfg, copy(coefficients))),
            (:gpu_memory_info, () -> gpu_memory_info()),
            (:gpu_clear_cache!, () -> gpu_clear_cache!()),
        )
            err = try
                call()
                nothing
            catch caught
                caught
            end
            if err isa SHTnsKit.BackendUnavailableError
                @test err.operation == operation
                @test occursin("CUDA.functional()", sprint(showerror, err))
            else
                @test err isa SHTnsKit.BackendUnavailableError
            end
        end

        @test gpu_analysis_safe(cfg, field) ≈ coefficients
        @test gpu_synthesis_safe(cfg, coefficients) ≈ synthesis(cfg, coefficients)
        wrapped_field = SafeFallbackArray(field)
        wrapped_coefficients = SafeFallbackArray(coefficients)
        wrapped_analysis = try
            gpu_analysis_safe(cfg, wrapped_field; device=SHTnsKit.GPU())
        catch err
            err
        end
        wrapped_synthesis = try
            gpu_synthesis_safe(cfg, wrapped_coefficients; device=SHTnsKit.GPU())
        catch err
            err
        end
        @test wrapped_analysis isa Matrix
        @test wrapped_synthesis isa Matrix
        if wrapped_analysis isa Matrix
            @test wrapped_analysis ≈ coefficients
            @test on_device(wrapped_analysis) isa SHTnsKit.CPU
        end
        if wrapped_synthesis isa Matrix
            @test wrapped_synthesis ≈ synthesis(cfg, coefficients)
            @test on_device(wrapped_synthesis) isa SHTnsKit.CPU
        end

        oom_helper_defined = isdefined(extension, :_with_cuda_oom_fallback)
        @test oom_helper_defined
        if oom_helper_defined
            injected = SafeFallbackRedispatchError()
            @test_throws SafeFallbackRedispatchError extension._with_cuda_oom_fallback(
                () -> throw(injected),
                () -> :cpu,
            )
            @test_throws ErrorException extension._with_cuda_oom_fallback(
                () -> error("illegal memory access"),
                () -> :cpu,
            )
            @test_logs (:warn, r"GPU out of memory") begin
                @test extension._with_cuda_oom_fallback(
                    () -> throw(CUDA.OutOfGPUMemoryError()),
                    () -> :cpu,
                ) == :cpu
            end
        end
        safe_analysis = try
            gpu_analysis_safe(cfg, field; device=SHTnsKit.GPU())
        catch err
            err
        end
        safe_synthesis = try
            gpu_synthesis_safe(cfg, coefficients; device=SHTnsKit.GPU())
        catch err
            err
        end
        if safe_analysis isa AbstractMatrix
            @test safe_analysis ≈ coefficients
        else
            @test safe_analysis isa AbstractMatrix
        end
        if safe_synthesis isa AbstractMatrix
            @test safe_synthesis ≈ synthesis(cfg, coefficients)
        else
            @test safe_synthesis isa AbstractMatrix
        end
        @test !check_gpu_memory(1)
        @test isempty(get_available_gpus())
        @test !set_gpu_device(0)
    else
        cfg = create_gauss_config(3, 6; nlon=8)
        host = Float64[sin(i / 3) + cos(j / 4) for i in 1:cfg.nlat, j in 1:cfg.nlon]
        device = to_device(SHTnsKit.GPU(), host)

        @test device isa CuArray
        @test on_device(device) isa SHTnsKit.GPU
        device_view = @view device[:, :]
        @test device_view isa CUDA.AnyCuArray
        @test on_device(device_view) isa SHTnsKit.GPU
        @test SHTnsKit._gpu_adapter_matches(extension.CUDA_ADAPTER, device_view)
        @test to_device(SHTnsKit.GPU(), device_view) === device_view
        @test to_device(SHTnsKit.GPU(), host, device_view) isa CUDA.AnyCuArray
        @test_throws ArgumentError analysis(SHTnsKit.CPU(), cfg, device_view)
        @test to_device(SHTnsKit.GPU(), host, device) isa CuArray
        @test to_device(host, SHTnsKit.GPU(), device) isa CuArray

        coefficients = analysis(SHTnsKit.GPU(), cfg, device_view)
        @test coefficients isa CUDA.AnyCuArray
        @test analysis(cfg, device_view) isa CUDA.AnyCuArray
        coefficient_view = @view coefficients[:, :]
        @test synthesis(SHTnsKit.GPU(), cfg, coefficients) isa CuArray
        @test synthesis(cfg, coefficient_view) isa CUDA.AnyCuArray
        @test synthesis_cplx(SHTnsKit.GPU(), cfg, coefficient_view) isa CUDA.AnyCuArray
        @test synthesis_cplx(cfg, coefficient_view) isa CUDA.AnyCuArray
        legacy_coefficients = gpu_analysis(cfg, host)
        @test legacy_coefficients isa Matrix
        @test gpu_synthesis(cfg, legacy_coefficients) isa Matrix

        CUDA.allowscalar(false)
        variant_cfg = create_gauss_config(
            5, 8; nlon=14, mres=2, norm=:schmidt,
            real_norm=true, cs_phase=false,
        )
        variant_dense = _variant_coefficients(variant_cfg, Float32)
        variant_packed = SHTnsKit.pack_lm(variant_cfg, variant_dense)
        variant_field = synthesis_packed(variant_cfg, variant_packed)
        device_field = CuArray(variant_field)
        device_packed = CuArray(variant_packed)
        @test analysis_packed(variant_cfg, device_field) isa CUDA.AnyCuArray
        @test Array(analysis_packed(SHTnsKit.GPU(), variant_cfg, device_field)) ≈
              variant_packed atol=2f-4 rtol=2f-4
        @test Array(synthesis_packed(variant_cfg, device_packed)) ≈
              variant_field atol=2f-4 rtol=2f-4
        ltr = 3
        @test Array(analysis_packed_l(variant_cfg, device_field, ltr)) ≈
              analysis_packed_l(variant_cfg, variant_field, ltr) atol=2f-4 rtol=2f-4
        @test Array(synthesis_packed_l(variant_cfg, device_packed, ltr)) ≈
              synthesis_packed_l(variant_cfg, variant_packed, ltr) atol=2f-4 rtol=2f-4

        axis_coefficients = ComplexF32.(variant_dense[:, 1])
        axis_field = synthesis_axisym(variant_cfg, axis_coefficients)
        @test Array(analysis_axisym(variant_cfg, CuArray(axis_field))) ≈
              axis_coefficients atol=2f-4 rtol=2f-4
        @test Array(synthesis_axisym(variant_cfg, CuArray(axis_coefficients))) ≈
              axis_field atol=2f-4 rtol=2f-4

        im, physical_m = 2, 4
        mode_coefficients = ComplexF32.(variant_dense[(physical_m + 1):end, physical_m + 1])
        mode = synthesis_packed_ml(variant_cfg, im, mode_coefficients, variant_cfg.lmax)
        @test Array(analysis_packed_ml(
            variant_cfg, im, CuArray(mode), variant_cfg.lmax,
        )) ≈ mode_coefficients atol=2f-4 rtol=2f-4
        @test Array(synthesis_packed_ml(
            variant_cfg, im, CuArray(mode_coefficients), variant_cfg.lmax,
        )) ≈ mode atol=2f-4 rtol=2f-4

        batch_fields = repeat(
            reshape(Float32.(reshape(variant_field, variant_cfg.nlat, variant_cfg.nlon)),
                    variant_cfg.nlat, variant_cfg.nlon, 1), 1, 1, 2,
        )
        device_batch = CuArray(batch_fields)
        batch_coefficients = analysis_batch(variant_cfg, device_batch)
        @test batch_coefficients isa CUDA.AnyCuArray
        @test Array(batch_coefficients) ≈ analysis_batch(variant_cfg, batch_fields) atol=2f-4 rtol=2f-4
        @test Array(synthesis_batch(variant_cfg, batch_coefficients)) ≈
              batch_fields atol=2f-4 rtol=2f-4
        analysis_output = similar(batch_coefficients)
        @test analysis_batch!(variant_cfg, analysis_output, device_batch) === analysis_output
        synthesis_output = similar(device_batch)
        @test synthesis_batch!(variant_cfg, synthesis_output, batch_coefficients) === synthesis_output

        complex_cfg = create_gauss_config(3, 6; nlon=10)
        complex_coefficients = zeros(ComplexF32, nlm_cplx_calc(3, 3, 1))
        complex_coefficients[LM_cplx_index(3, 3, 2, -1) + 1] = 0.2f0 - 0.1f0im
        complex_field = synthesis_packed_cplx(complex_cfg, complex_coefficients)
        device_complex = CuArray(complex_coefficients)
        @test Array(synthesis_packed_cplx(complex_cfg, device_complex)) ≈
              complex_field atol=2f-4 rtol=2f-4
        @test Array(analysis_packed_cplx(complex_cfg, CuArray(complex_field))) ≈
              complex_coefficients atol=2f-4 rtol=2f-4
        run_gpu_scalar_variant_matrix(CUDAScalarAdapter())
        run_scalar_full_parity(
            CUDAScalarAdapter();
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
