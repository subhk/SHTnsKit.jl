using Test
using SHTnsKit
using CUDA
using GPUArrays
using GPUArraysCore
using KernelAbstractions

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
    end
end
