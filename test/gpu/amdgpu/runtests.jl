using Test
using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
using KernelAbstractions

include("../../parity/scalar_full.jl")
include("../../parity/scalar_variants.jl")
include("../../parity/sphtor_full.jl")
include("../../parity/qst_full.jl")

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
analysis_call(::AMDGPUScalarAdapter, cfg, field; use_rfft=false) =
    analysis(GPU(), cfg, field; use_rfft)
synthesis_call(::AMDGPUScalarAdapter, cfg, coefficients, _prototype;
               real_output, use_rfft=false) =
    synthesis(GPU(), cfg, coefficients; real_output, use_rfft)
synthesis_cplx_call(::AMDGPUScalarAdapter, cfg, coefficients, _prototype) =
    synthesis_cplx(cfg, coefficients)
assert_resident(::AMDGPUScalarAdapter, value) = @test value isa AMDGPU.AnyROCArray

struct AMDGPUVectorAdapter <: VectorParityAdapter end
vector_place(::AMDGPUVectorAdapter, ::SHTConfig, value, ::Symbol) = ROCArray(value)
vector_collect(::AMDGPUVectorAdapter, value, ::SHTConfig) = Array(value)
vector_resident(::AMDGPUVectorAdapter, value) = @test value isa AMDGPU.AnyROCArray
vector_analysis(::AMDGPUVectorAdapter, cfg, Vt, Vp; use_rfft=false) =
    analysis_sphtor(GPU(), cfg, Vt, Vp; use_rfft)
vector_analysis_cplx(::AMDGPUVectorAdapter, cfg, Vt, Vp) =
    analysis_sphtor_cplx(GPU(), cfg, Vt, Vp)
vector_synthesis(::AMDGPUVectorAdapter, cfg, S, T, _prototype;
                 real_output=true, use_rfft=false) =
    synthesis_sphtor(GPU(), cfg, S, T; real_output, use_rfft)
vector_synthesis_cplx(::AMDGPUVectorAdapter, cfg, S, T, _prototype) =
    synthesis_sphtor_cplx(GPU(), cfg, S, T)
vector_sph(::AMDGPUVectorAdapter, cfg, S, _prototype; real_output=true) =
    synthesis_sph(GPU(), cfg, S; real_output)
vector_sph_cplx(::AMDGPUVectorAdapter, cfg, S, _prototype) =
    synthesis_sph_cplx(GPU(), cfg, S)
vector_tor(::AMDGPUVectorAdapter, cfg, T, _prototype; real_output=true) =
    synthesis_tor(GPU(), cfg, T; real_output)
vector_tor_cplx(::AMDGPUVectorAdapter, cfg, T, _prototype) =
    synthesis_tor_cplx(GPU(), cfg, T)

struct AMDGPUQSTAdapter <: QSTParityAdapter end
qst_place(::AMDGPUQSTAdapter, ::SHTConfig, value, ::Symbol) = ROCArray(value)
qst_collect(::AMDGPUQSTAdapter, value, ::SHTConfig) = Array(value)
qst_resident(::AMDGPUQSTAdapter, value) = @test value isa AMDGPU.AnyROCArray
qst_analysis(::AMDGPUQSTAdapter, cfg, Vr, Vt, Vp; use_rfft=false) =
    analysis_qst(GPU(), cfg, Vr, Vt, Vp; use_rfft)
qst_analysis_cplx(::AMDGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst_cplx(GPU(), cfg, Vr, Vt, Vp)
qst_synthesis(::AMDGPUQSTAdapter, cfg, Q, S, Tlm, prototype;
              real_output=true, use_rfft=false) =
    synthesis_qst(GPU(), cfg, Q, S, Tlm; prototype, real_output, use_rfft)
qst_synthesis_cplx(::AMDGPUQSTAdapter, cfg, Q, S, Tlm, prototype) =
    synthesis_qst_cplx(GPU(), cfg, Q, S, Tlm; prototype)
function assert_warm_device_noalloc(::AMDGPUScalarAdapter, call)
    call()
    AMDGPU.synchronize()
    @test @allocated(begin
        call()
        AMDGPU.synchronize()
    end) <= 65_536
    return nothing
end

@testset "AMDGPU backend routing" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitAMDGPUExt)
    @test extension !== nothing
    @test isdefined(extension.GPUCommon, :scalar_analysis_kernel!)
    @test isdefined(extension.GPUCommon, :scalar_synthesis_kernel!)
    @test isdefined(extension.GPUCommon, :coefficient_conversion_kernel!)
    @test isdefined(extension.GPUCommon, :coefficient_batch_conversion_kernel!)
    @test isdefined(extension.GPUCommon, :ScalarWorkspaceCache)
    @test isdefined(extension.GPUCommon, :vector_derivative_table_kernel!)
    @test isdefined(extension.GPUCommon, :vector_analysis_kernel!)
    @test isdefined(extension.GPUCommon, :vector_synthesis_kernel!)
    @test isdefined(extension.GPUCommon, :vector_diagonal_kernel!)
    @test isdefined(extension, :_amdgpu_scalar_analysis)
    @test isdefined(extension, :_amdgpu_scalar_synthesis)
    @test isdefined(extension, :_amdgpu_clear_scalar_cache!)
    @test isdefined(extension, :_amdgpu_scalar_analysis_direct!)
    @test isdefined(extension, :_amdgpu_batch_analysis_direct!)
    @test isdefined(extension, :_amdgpu_vector_analysis)
    @test isdefined(extension, :_amdgpu_vector_synthesis)
    @test isdefined(extension, :_amdgpu_vector_analysis_direct!)
    @test isdefined(extension, :_amdgpu_vector_synthesis_direct!)
    @test fieldnames(extension.AMDGPUScalarTables) == (:x, :weights, :Plm, :scales)
    @test isdefined(extension, :AMDGPUVectorTables)
    @test isdefined(extension, :_AMDGPU_VECTOR_CACHE)
    @test isdefined(extension, :_amdgpu_vector_tables)
    @test extension._AMDGPU_SCALAR_CACHE.max_per_device == 8
    @test extension._AMDGPU_VECTOR_CACHE.max_per_device == 8
    @test which(
        analysis_sphtor,
        Tuple{SHTConfig,ROCArray{Float32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_sphtor,
        Tuple{SHTConfig,ROCArray{ComplexF32,2},ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_qst,
        Tuple{SHTConfig,ROCArray{Float32,2},ROCArray{Float32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_qst,
        Tuple{SHTConfig,ROCArray{ComplexF32,2},ROCArray{ComplexF32,2},ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_sphtor,
        Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{Float32,2},ROCArray{Float32,2}},
    ).module === SHTnsKit
    for (function_name, signature) in (
        (:analysis_packed, Tuple{SHTConfig,ROCArray{Float32,1}}),
        (:synthesis_packed, Tuple{SHTConfig,ROCArray{ComplexF32,1}}),
        (:analysis_packed_l, Tuple{SHTConfig,ROCArray{Float32,1},Int}),
        (:synthesis_packed_l, Tuple{SHTConfig,ROCArray{ComplexF32,1},Int}),
        (:analysis_axisym, Tuple{SHTConfig,ROCArray{Float32,1}}),
        (:synthesis_axisym, Tuple{SHTConfig,ROCArray{ComplexF32,1}}),
        (:analysis_axisym_l, Tuple{SHTConfig,ROCArray{Float32,1},Int}),
        (:synthesis_axisym_l, Tuple{SHTConfig,ROCArray{ComplexF32,1},Int}),
        (:analysis_packed_ml, Tuple{SHTConfig,Int,ROCArray{ComplexF32,1},Int}),
        (:synthesis_packed_ml, Tuple{SHTConfig,Int,ROCArray{ComplexF32,1},Int}),
        (:analysis_packed_cplx, Tuple{SHTConfig,ROCArray{ComplexF32,2}}),
        (:synthesis_packed_cplx, Tuple{SHTConfig,ROCArray{ComplexF32,1}}),
        (:analysis_packed_cplx_l, Tuple{SHTConfig,ROCArray{ComplexF32,2},Int}),
        (:synthesis_packed_cplx_l, Tuple{SHTConfig,ROCArray{ComplexF32,1},Int}),
        (:analysis_batch, Tuple{SHTConfig,ROCArray{Float32,3}}),
        (:analysis_batch!, Tuple{SHTConfig,ROCArray{ComplexF32,3},ROCArray{Float32,3}}),
        (:synthesis_batch, Tuple{SHTConfig,ROCArray{ComplexF32,3}}),
        (:synthesis_batch!, Tuple{SHTConfig,ROCArray{Float32,3},ROCArray{ComplexF32,3}}),
        (:synthesis_batch_cplx, Tuple{SHTConfig,ROCArray{ComplexF32,3}}),
    )
        @test which(getproperty(SHTnsKit, function_name), signature).module === extension
    end
    @test which(
        analysis!, Tuple{SHTPlan,ROCArray{ComplexF32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis!, Tuple{SHTPlan,ROCArray{Float32,2},ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_sphtor!,
        Tuple{SHTPlan,ROCArray{ComplexF32,2},ROCArray{ComplexF32,2},
              ROCArray{Float32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_sphtor!,
        Tuple{SHTPlan,ROCArray{Float32,2},ROCArray{Float32,2},
              ROCArray{ComplexF32,2},ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis!,
        Tuple{SHTnsKit.GPU,SHTPlan,ROCArray{ComplexF32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(gpu_clear_cache!, Tuple{SHTnsKit.GPU}).module === SHTnsKit
    @test hasmethod(
        SHTnsKit._gpu_adapter_clear_cache!, Tuple{typeof(extension.AMDGPU_ADAPTER)},
    )
    cache_device = typemax(Int)
    extension.GPUCommon.scalar_cache_insert!(
        extension._AMDGPU_SCALAR_CACHE, cache_device, UInt(1), Float32, UInt(1), :sentinel,
    )
    if isdefined(extension, :_AMDGPU_VECTOR_CACHE)
        extension.GPUCommon.scalar_cache_insert!(
            extension._AMDGPU_VECTOR_CACHE, cache_device,
            UInt(1), Float32, UInt(1), :vector_sentinel,
        )
    end
    extension._amdgpu_clear_scalar_cache!(; device=cache_device)
    @test extension.GPUCommon.scalar_cache_size(
        extension._AMDGPU_SCALAR_CACHE; device=cache_device,
    ) == 0
    if isdefined(extension, :_AMDGPU_VECTOR_CACHE)
        @test extension.GPUCommon.scalar_cache_size(
            extension._AMDGPU_VECTOR_CACHE; device=cache_device,
        ) == 0
    end
    workspace_cache = extension.GPUCommon.ScalarWorkspaceCache(2)
    workspace_owner = Ref(:owner)
    builds = Ref(0)
    builder = () -> (builds[] += 1; :workspace)
    @test extension.GPUCommon.scalar_workspace_use!(
        identity, builder, workspace_cache, :mock, workspace_owner,
        Float32, :scalar, (1,), UInt(1),
    ) === :workspace
    @test extension.GPUCommon.scalar_workspace_use!(
        identity, builder, workspace_cache, :mock, workspace_owner,
        Float32, :scalar, (1,), UInt(1),
    ) === :workspace
    @test builds[] == 1
    extension.GPUCommon.scalar_workspace_clear!(workspace_cache; device=:mock)
    @test extension.GPUCommon.scalar_workspace_size(workspace_cache) == 0
    @test which(
        synthesis_cplx, Tuple{SHTConfig,ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        synthesis_cplx, Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{ComplexF32,2}},
    ).module === SHTnsKit
    run_shared_scalar_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_scalar_variant_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_vector_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_scalar_workspace_cache_reference(extension.GPUCommon)
    source = read(joinpath(@__DIR__, "../../../ext/SHTnsKitAMDGPUExt.jl"), String)
    @test occursin("function _amdgpu_vector_tables", source)
    if occursin("function _amdgpu_vector_tables", source)
        scalar_table_builder = split(
            split(source, "function _amdgpu_scalar_tables"; limit=2)[2],
            "function _amdgpu_vector_tables"; limit=2,
        )[1]
        @test occursin("scalar_host_tables", scalar_table_builder)
        @test !occursin("vector_host_tables", scalar_table_builder)
        @test !occursin("vector_derivative_table_kernel!", scalar_table_builder)
        @test !occursin("dtheta", scalar_table_builder)
    end
    vector_pipeline = split(
        split(source, "function _amdgpu_vector_analysis"; limit=2)[2],
        "@inline function _amdgpu_lcap"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", vector_pipeline)
    @test !occursin(r"\bcollect\s*\(", vector_pipeline)
    ordinary_pipeline = split(
        split(source, "function _amdgpu_scalar_analysis"; limit=2)[2],
        "function _gpu_adapter_analysis"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", ordinary_pipeline)
    @test !occursin(r"\bcollect\s*\(", ordinary_pipeline)
    @test occursin("use_rfft=true requires a real-valued input", ordinary_pipeline)
    @test occursin("use_rfft=true implies real_output", ordinary_pipeline)
    variant_pipeline = split(
        split(source, "@inline function _amdgpu_lcap"; limit=2)[2],
        "end # module SHTnsKitAMDGPUExt"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", variant_pipeline)
    @test !occursin(r"\bcollect\s*\(", variant_pipeline)
    @test occursin("ScalarWorkspaceCache(8)", source)
    @test occursin("_amdgpu_batch_analysis_direct!", source)
    @test occursin("_amdgpu_batch_synthesis_direct!", source)
    @test occursin("_amdgpu_vector_analysis_direct!", source)
    @test occursin("_amdgpu_vector_synthesis_direct!", source)
    @test occursin("tables = _amdgpu_vector_tables(cfg, RT)", source)
    @test occursin("FFTW.plan_rfft", source)
    @test !occursin(
        r"result\s*=\s*_amdgpu_(?:scalar_analysis|scalar_synthesis|batch_analysis|batch_synthesis)\(",
        source,
    )

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
        coefficients = analysis(SHTnsKit.GPU(), cfg, device_view)
        @test coefficients isa AMDGPU.AnyROCArray
        @test synthesis_cplx(SHTnsKit.GPU(), cfg, coefficients) isa AMDGPU.AnyROCArray
        @test synthesis_cplx(cfg, coefficients) isa AMDGPU.AnyROCArray

        AMDGPU.allowscalar(false)
        vector_cfg = _vector_config(:regular_poles, 3, 8; norm=:schmidt,
                                    real_norm=true, cs_phase=false)
        vector_S, vector_T = _vector_modes(vector_cfg, Float32)
        vector_Vt, vector_Vp = synthesis_sphtor(
            vector_cfg, ROCArray(vector_S), ROCArray(vector_T),
        )
        @test vector_Vt isa AMDGPU.AnyROCArray
        @test vector_Vp isa AMDGPU.AnyROCArray
        @test analysis_sphtor(vector_cfg, vector_Vt, vector_Vp)[1] isa AMDGPU.AnyROCArray
        legacy_S, legacy_T = gpu_analysis_sphtor(
            vector_cfg, Array(vector_Vt), Array(vector_Vp),
        )
        @test legacy_S isa Matrix{ComplexF32}
        @test legacy_T isa Matrix{ComplexF32}
        legacy_Vt, legacy_Vp = gpu_synthesis_sphtor(
            vector_cfg, legacy_S, legacy_T,
        )
        @test legacy_Vt isa Matrix{Float32}
        @test legacy_Vp isa Matrix{Float32}

        # Ordinary and direct-plan vector paths accept supported strided device
        # matrix views without staging through host memory.
        Vt_storage = AMDGPU.zeros(Float32, vector_cfg.nlat, 2vector_cfg.nlon)
        Vp_storage = similar(Vt_storage)
        Vt_view = @view Vt_storage[:, 1:2:end]
        Vp_view = @view Vp_storage[:, 1:2:end]
        copyto!(Vt_view, vector_Vt)
        copyto!(Vp_view, vector_Vp)
        view_S, view_T = analysis_sphtor(vector_cfg, Vt_view, Vp_view)
        @test view_S isa AMDGPU.AnyROCArray
        @test view_T isa AMDGPU.AnyROCArray
        @test Array(view_S) ≈ vector_S atol=4f-4 rtol=4f-4
        @test Array(view_T) ≈ vector_T atol=4f-4 rtol=4f-4

        S_storage = AMDGPU.zeros(ComplexF32, size(vector_S, 1), 2size(vector_S, 2))
        T_storage = similar(S_storage)
        S_view = @view S_storage[:, 1:2:end]
        T_view = @view T_storage[:, 1:2:end]
        copyto!(S_view, ROCArray(vector_S))
        copyto!(T_view, ROCArray(vector_T))
        view_Vt, view_Vp = synthesis_sphtor(vector_cfg, S_view, T_view)
        @test view_Vt isa AMDGPU.AnyROCArray
        @test view_Vp isa AMDGPU.AnyROCArray
        @test Array(view_Vt) ≈ Array(vector_Vt) atol=4f-4 rtol=4f-4
        @test Array(view_Vp) ≈ Array(vector_Vp) atol=4f-4 rtol=4f-4

        vector_plan = SHTPlan(vector_cfg)
        plan_S_store = similar(S_storage); plan_T_store = similar(S_storage)
        plan_S = @view plan_S_store[:, 1:2:end]
        plan_T = @view plan_T_store[:, 1:2:end]
        analysis_sphtor!(vector_plan, plan_S, plan_T, vector_Vt, vector_Vp)
        plan_Vt_store = similar(Vt_storage); plan_Vp_store = similar(Vp_storage)
        plan_Vt = @view plan_Vt_store[:, 1:2:end]
        plan_Vp = @view plan_Vp_store[:, 1:2:end]
        synthesis_sphtor!(vector_plan, plan_Vt, plan_Vp, plan_S, plan_T)
        @test Array(plan_S) ≈ vector_S atol=4f-4 rtol=4f-4
        @test Array(plan_T) ≈ vector_T atol=4f-4 rtol=4f-4
        @test Array(plan_Vt) ≈ Array(vector_Vt) atol=4f-4 rtol=4f-4
        @test Array(plan_Vp) ≈ Array(vector_Vp) atol=4f-4 rtol=4f-4
        assert_warm_device_noalloc(AMDGPUScalarAdapter()) do
            analysis_sphtor!(vector_plan, plan_S, plan_T, vector_Vt, vector_Vp)
        end
        assert_warm_device_noalloc(AMDGPUScalarAdapter()) do
            synthesis_sphtor!(vector_plan, plan_Vt, plan_Vp, plan_S, plan_T)
        end
        variant_cfg = create_gauss_config(
            5, 8; nlon=14, mres=2, norm=:schmidt,
            real_norm=true, cs_phase=false,
        )
        variant_dense = _variant_coefficients(variant_cfg, Float32)
        variant_packed = SHTnsKit.pack_lm(variant_cfg, variant_dense)
        variant_field = synthesis_packed(variant_cfg, variant_packed)
        device_field = ROCArray(variant_field)
        device_packed = ROCArray(variant_packed)
        @test analysis_packed(variant_cfg, device_field) isa AMDGPU.AnyROCArray
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
        @test Array(analysis_axisym(variant_cfg, ROCArray(axis_field))) ≈
              axis_coefficients atol=2f-4 rtol=2f-4
        @test Array(synthesis_axisym(variant_cfg, ROCArray(axis_coefficients))) ≈
              axis_field atol=2f-4 rtol=2f-4

        im, physical_m = 2, 4
        mode_coefficients = ComplexF32.(variant_dense[(physical_m + 1):end, physical_m + 1])
        mode = synthesis_packed_ml(variant_cfg, im, mode_coefficients, variant_cfg.lmax)
        @test Array(analysis_packed_ml(
            variant_cfg, im, ROCArray(mode), variant_cfg.lmax,
        )) ≈ mode_coefficients atol=2f-4 rtol=2f-4
        @test Array(synthesis_packed_ml(
            variant_cfg, im, ROCArray(mode_coefficients), variant_cfg.lmax,
        )) ≈ mode atol=2f-4 rtol=2f-4

        batch_fields = repeat(
            reshape(Float32.(reshape(variant_field, variant_cfg.nlat, variant_cfg.nlon)),
                    variant_cfg.nlat, variant_cfg.nlon, 1), 1, 1, 2,
        )
        device_batch = ROCArray(batch_fields)
        batch_coefficients = analysis_batch(variant_cfg, device_batch)
        @test batch_coefficients isa AMDGPU.AnyROCArray
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
        device_complex = ROCArray(complex_coefficients)
        @test Array(synthesis_packed_cplx(complex_cfg, device_complex)) ≈
              complex_field atol=2f-4 rtol=2f-4
        @test Array(analysis_packed_cplx(complex_cfg, ROCArray(complex_field))) ≈
              complex_coefficients atol=2f-4 rtol=2f-4
        run_gpu_scalar_variant_matrix(AMDGPUScalarAdapter())
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
        run_sphtor_full_parity(AMDGPUVectorAdapter())
        run_qst_full_parity(AMDGPUQSTAdapter())
    end
end
