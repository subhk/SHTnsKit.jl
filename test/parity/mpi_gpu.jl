using Test
using LinearAlgebra

# Reuse the same mathematical oracle suites as CPU and standalone GPU parity.
isdefined(@__MODULE__, :ScalarParityAdapter) || include("scalar_full.jl")
isdefined(@__MODULE__, :VectorParityAdapter) || include("sphtor_full.jl")
isdefined(@__MODULE__, :QSTParityAdapter) || include("qst_full.jl")

"""A CPU-backed stand-in that exercises the MPI/GPU policy without GPU hardware."""
mutable struct MockMPIArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.IndexStyle(::Type{<:MockMPIArray}) = IndexLinear()
Base.size(array::MockMPIArray) = size(array.data)
Base.getindex(array::MockMPIArray, indices...) = array.data[indices...]
Base.setindex!(array::MockMPIArray, value, indices...) =
    (array.data[indices...] = value)
Base.copyto!(destination::MockMPIArray, source::AbstractArray) =
    (copyto!(destination.data, source); destination)
Base.copyto!(destination::AbstractArray, source::MockMPIArray) =
    copyto!(destination, source.data)
MockMPIArray{T}(::UndefInitializer, dims::Tuple) where {T} =
    MockMPIArray(Array{T}(undef, dims))
MockMPIArray{T}(::UndefInitializer, n::Integer) where {T} =
    MockMPIArray(Vector{T}(undef, n))
Base.similar(array::MockMPIArray, ::Type{T}, dims::Dims) where {T} =
    MockMPIArray(Array{T}(undef, dims))

"""A fake allocation whose device may differ from the task's current device."""
mutable struct MockMultiDeviceArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    device::Int
end

const MOCK_CURRENT_DEVICE = Ref(99)
MockMultiDeviceArray{T}(::UndefInitializer, dims::Dims) where {T} =
    MockMultiDeviceArray(Array{T}(undef, dims), MOCK_CURRENT_DEVICE[])

Base.IndexStyle(::Type{<:MockMultiDeviceArray}) = IndexLinear()
Base.size(array::MockMultiDeviceArray) = size(array.data)
Base.getindex(array::MockMultiDeviceArray, indices...) = array.data[indices...]
Base.setindex!(array::MockMultiDeviceArray, value, indices...) =
    (array.data[indices...] = value)
Base.copyto!(destination::MockMultiDeviceArray, source::AbstractArray) =
    (copyto!(destination.data, source); destination)
Base.copyto!(destination::AbstractArray, source::MockMultiDeviceArray) =
    copyto!(destination, source.data)
Base.copyto!(destination::MockMultiDeviceArray,
             source::MockMultiDeviceArray) =
    (copyto!(destination.data, source.data); destination)

function _cache_temporary_view!(extension, adapter, comm)
    root = MockMultiDeviceArray(reshape(Float32.(1:4), 2, 2), 1)
    extension._staging_entry(adapter, comm, view(root, 1, :), 2)
    return WeakRef(root)
end

function _mpi_gpu_place(array_type, values::AbstractArray, decomposition, comm)
    pen = PencilArrays.Pencil(array_type, size(values), decomposition, comm)
    result = PencilArrays.PencilArray{eltype(values)}(undef, pen)
    ranges = PencilArrays.range_local(pen)
    local_values = Array{eltype(values)}(undef, size(parent(result)))
    @inbounds for index in CartesianIndices(local_values)
        global_index = ntuple(d -> ranges[d][index[d]], ndims(values))
        local_values[index] = values[global_index...]
    end
    copyto!(parent(result), local_values)
    return result
end

function _mpi_gpu_collect(value::PencilArrays.PencilArray, comm)
    local_values = Array(parent(value))
    global_value = zeros(eltype(value), PencilArrays.size_global(value))
    ranges = PencilArrays.range_local(PencilArrays.pencil(value))
    @inbounds for index in CartesianIndices(local_values)
        global_index = ntuple(d -> ranges[d][index[d]], ndims(local_values))
        global_value[global_index...] = local_values[index]
    end
    MPI.Allreduce!(global_value, +, comm)
    return global_value
end

_mpi_gpu_collect_any(value::PencilArrays.PencilArray, comm) =
    _mpi_gpu_collect(value, comm)
_mpi_gpu_collect_any(value::AbstractArray, _comm) = Array(value)

function _mpi_gpu_fill_native_spatial!(destination, fields)
    ranges = PencilArrays.range_local(PencilArrays.pencil(destination))
    host = Array{eltype(destination)}(undef, size(parent(destination)))
    @inbounds for lev in axes(host, 3), local_θ in axes(host, 2),
                  local_φ in axes(host, 1)
        host[local_φ, local_θ, lev] =
            fields[lev][ranges[2][local_θ], ranges[1][local_φ]]
    end
    copyto!(parent(destination), host)
    return destination
end

function _mpi_gpu_fill_native_spectral!(destination, coefficients, m_local)
    host = zeros(eltype(destination), size(parent(destination)))
    @inbounds for lev in axes(host, 3), (local_m, m) in enumerate(m_local),
                  l in m:(size(host, 1) - 1)
        host[l + 1, local_m, lev] = coefficients[lev][l + 1, m + 1]
    end
    copyto!(parent(destination), host)
    return destination
end

function _mpi_gpu_native_spectral_error(value, references, m_local)
    host = Array(parent(value))
    error = zero(typeof(abs(zero(eltype(host)))))
    @inbounds for lev in axes(host, 3), (local_m, m) in enumerate(m_local),
                  l in m:(size(host, 1) - 1)
        error = max(error, abs(
            host[l + 1, local_m, lev] - references[lev][l + 1, m + 1],
        ))
    end
    return error
end

function _mpi_gpu_native_spatial_error(value, references)
    ranges = PencilArrays.range_local(PencilArrays.pencil(value))
    host = Array(parent(value))
    error = zero(eltype(host))
    @inbounds for lev in axes(host, 3), local_θ in axes(host, 2),
                  local_φ in axes(host, 1)
        error = max(error, abs(
            host[local_φ, local_θ, lev] -
            references[lev][ranges[2][local_θ], ranges[1][local_φ]],
        ))
    end
    return error
end

function _mpi_gpu_native_references(cfg, ::Type{RT}, nlev) where {RT}
    CT = Complex{RT}
    Q = [zeros(CT, cfg.lmax + 1, cfg.mmax + 1) for _ in 1:nlev]
    S = [zeros(CT, cfg.lmax + 1, cfg.mmax + 1) for _ in 1:nlev]
    T = [zeros(CT, cfg.lmax + 1, cfg.mmax + 1) for _ in 1:nlev]
    for lev in 1:nlev, m in 0:cfg.mmax, l in m:cfg.lmax
        scale = RT(0.025 * (lev + 1) / (l + 1)^2)
        imag_scale = m == 0 ? zero(RT) : RT(0.35) * scale
        Q[lev][l + 1, m + 1] = CT(scale, imag_scale)
        if l > 0
            S[lev][l + 1, m + 1] = CT(RT(0.7) * scale, -imag_scale)
            T[lev][l + 1, m + 1] = CT(-RT(0.4) * scale, RT(0.2) * imag_scale)
        end
    end
    Vr = [SHTnsKit.synthesis(cfg, Q[lev]; real_output=true) for lev in 1:nlev]
    vector = [SHTnsKit.synthesis_sphtor(
        cfg, S[lev], T[lev]; real_output=true,
    ) for lev in 1:nlev]
    Vt = [vector[lev][1] for lev in 1:nlev]
    Vp = [vector[lev][2] for lev in 1:nlev]
    Qanalysis = [SHTnsKit.analysis(cfg, Vr[lev]) for lev in 1:nlev]
    STanalysis = [SHTnsKit.analysis_sphtor(
        cfg, Vt[lev], Vp[lev],
    ) for lev in 1:nlev]
    Sanalysis = [STanalysis[lev][1] for lev in 1:nlev]
    Tanalysis = [STanalysis[lev][2] for lev in 1:nlev]
    return (; Q, S, T, Vr, Vt, Vp, Qanalysis, Sanalysis, Tanalysis)
end

@inline function _mpi_gpu_assert_resident(value, is_vendor)
    if value isa PencilArrays.PencilArray
        @test is_vendor(parent(value))
    elseif value isa AbstractArray
        @test is_vendor(value)
    elseif value isa Tuple
        foreach(item -> _mpi_gpu_assert_resident(item, is_vendor), value)
    end
    return value
end

struct MPIGPUScalarAdapter{A,F} <: ScalarParityAdapter
    array_type::A
    is_vendor::F
    comm
end
struct MPIGPUVectorAdapter{A,F} <: VectorParityAdapter
    array_type::A
    is_vendor::F
    comm
end
struct MPIGPUQSTAdapter{A,F} <: QSTParityAdapter
    array_type::A
    is_vendor::F
    comm
end

place(adapter::MPIGPUScalarAdapter, cfg, value, kind::Symbol) =
    _mpi_gpu_place(adapter.array_type, value,
                   kind === :spectral ? (2,) : (1,), adapter.comm)
collect_result(adapter::MPIGPUScalarAdapter, value::PencilArrays.PencilArray,
               _cfg) = _mpi_gpu_collect(value, adapter.comm)
analysis_call(::MPIGPUScalarAdapter, cfg, field; use_rfft=false) =
    SHTnsKit.analysis(cfg, field; use_rfft)
synthesis_call(::MPIGPUScalarAdapter, cfg, coefficients, prototype;
               real_output, use_rfft=false) = SHTnsKit.synthesis(
    cfg, coefficients; prototype_θφ=prototype, real_output, use_rfft,
)
synthesis_cplx_call(::MPIGPUScalarAdapter, cfg, coefficients, prototype) =
    SHTnsKit.synthesis_cplx(cfg, coefficients; prototype_θφ=prototype)
assert_resident(adapter::MPIGPUScalarAdapter, value) =
    _mpi_gpu_assert_resident(value, adapter.is_vendor)

vector_place(adapter::MPIGPUVectorAdapter, cfg, value, kind::Symbol) =
    _mpi_gpu_place(adapter.array_type, value,
                   kind === :spectral ? (2,) : (1,), adapter.comm)
vector_collect(adapter::MPIGPUVectorAdapter,
               value::PencilArrays.PencilArray, _cfg) =
    _mpi_gpu_collect(value, adapter.comm)
vector_resident(adapter::MPIGPUVectorAdapter, value) =
    _mpi_gpu_assert_resident(value, adapter.is_vendor)
vector_analysis(::MPIGPUVectorAdapter, cfg, Vt, Vp; use_rfft=false) =
    SHTnsKit.analysis_sphtor(cfg, Vt, Vp; use_rfft)
vector_analysis_cplx(::MPIGPUVectorAdapter, cfg, Vt, Vp) =
    SHTnsKit.analysis_sphtor_cplx(cfg, Vt, Vp)
vector_synthesis(::MPIGPUVectorAdapter, cfg, S, T, prototype;
                 real_output=true, use_rfft=false) = SHTnsKit.synthesis_sphtor(
    cfg, S, T; prototype_θφ=prototype, real_output, use_rfft,
)
vector_synthesis_cplx(::MPIGPUVectorAdapter, cfg, S, T, prototype) =
    SHTnsKit.synthesis_sphtor_cplx(cfg, S, T; prototype_θφ=prototype)
vector_sph(::MPIGPUVectorAdapter, cfg, S, prototype; real_output=true) =
    SHTnsKit.synthesis_sph(cfg, S; prototype_θφ=prototype, real_output)
vector_sph_cplx(::MPIGPUVectorAdapter, cfg, S, prototype) =
    SHTnsKit.synthesis_sph_cplx(cfg, S; prototype_θφ=prototype)
vector_tor(::MPIGPUVectorAdapter, cfg, T, prototype; real_output=true) =
    SHTnsKit.synthesis_tor(cfg, T; prototype_θφ=prototype, real_output)
vector_tor_cplx(::MPIGPUVectorAdapter, cfg, T, prototype) =
    SHTnsKit.synthesis_tor_cplx(cfg, T; prototype_θφ=prototype)

qst_place(adapter::MPIGPUQSTAdapter, cfg, value, kind::Symbol) =
    _mpi_gpu_place(adapter.array_type, value,
                   kind === :spectral ? (2,) : (1,), adapter.comm)
qst_collect(adapter::MPIGPUQSTAdapter, value::PencilArrays.PencilArray, _cfg) =
    _mpi_gpu_collect(value, adapter.comm)
qst_resident(adapter::MPIGPUQSTAdapter, value) =
    _mpi_gpu_assert_resident(value, adapter.is_vendor)
qst_analysis(::MPIGPUQSTAdapter, cfg, Vr, Vt, Vp; use_rfft=false) =
    SHTnsKit.analysis_qst(cfg, Vr, Vt, Vp; use_rfft)
qst_analysis_cplx(::MPIGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    SHTnsKit.analysis_qst_cplx(cfg, Vr, Vt, Vp)
qst_analysis_inferred(::MPIGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    SHTnsKit.analysis_qst(cfg, Vr, Vt, Vp)
qst_analysis_cplx_inferred(::MPIGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    SHTnsKit.analysis_qst_cplx(cfg, Vr, Vt, Vp)
qst_synthesis(::MPIGPUQSTAdapter, cfg, Q, S, T, prototype;
              real_output=true, use_rfft=false) = SHTnsKit.synthesis_qst(
    cfg, Q, S, T; prototype_θφ=prototype, real_output, use_rfft,
)
qst_synthesis_cplx(::MPIGPUQSTAdapter, cfg, Q, S, T, prototype) =
    SHTnsKit.synthesis_qst_cplx(cfg, Q, S, T; prototype_θφ=prototype)
qst_synthesis_inferred(::MPIGPUQSTAdapter, cfg, Q, S, T, prototype) =
    SHTnsKit.synthesis_qst(cfg, Q, S, T; prototype_θφ=prototype)
qst_synthesis_cplx_inferred(::MPIGPUQSTAdapter, cfg, Q, S, T, prototype) =
    SHTnsKit.synthesis_qst_cplx(cfg, Q, S, T; prototype_θφ=prototype)

"""
Run the shared two-rank hardware matrix. This function is always parsed and its
invocation is source-checked; on machines without a matching device per rank it
emits exactly one honest skip and performs no device-math claim.
"""
function run_mpi_gpu_full_parity(vendor::Symbol, array_type::Type,
                                 is_vendor, functional::Bool, devices,
                                 activate_device!, device_of)
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    local_ready = functional && !isempty(devices)
    ready_ranks = MPI.Allreduce(local_ready ? 1 : 0, +, comm)
    if nranks != 2 || ready_ranks != nranks
        @test_skip nranks == 2 && ready_ranks == nranks
        return nothing
    end

    assigned = devices[mod(rank, length(devices)) + 1]
    activate_device!(assigned)
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    adapter = extension._parallel_gpu_adapter(array_type{Float32}(undef, 1))
    @test adapter !== nothing

    # Shared full-grid mathematical oracles (same suite used by CPU/standalone
    # GPU) validate values as well as residency for scalar, vector and QST.
    shared_axes = (
        grid_kinds=(:gauss,), precisions=(Float32, Float64), mres_values=(1,),
        norms=(:orthonormal, :fourpi, :schmidt),
        real_norm_values=(false, true),
        cs_phase_values=(false, true), pole_orders=(false,),
    )
    run_scalar_full_parity(MPIGPUScalarAdapter(array_type, is_vendor, comm);
                           shared_axes...)
    run_sphtor_full_parity(MPIGPUVectorAdapter(array_type, is_vendor, comm);
        shared_axes..., robert_values=(false, true))
    run_qst_full_parity(MPIGPUQSTAdapter(array_type, is_vendor, comm);
        shared_axes..., robert_values=(false, true))

    @testset "actual allocation device cache key" begin
        # A buffer allocated on device 1 remains keyed to device 1 even while
        # device 2 is current. Distinct real allocations produce two entries.
        if length(devices) >= 2
            activate_device!(devices[1])
            first_buffer = array_type{Float32}(undef, 1)
            activate_device!(devices[2])
            second_buffer = array_type{Float32}(undef, 1)
            @test device_of(first_buffer) == devices[1]
            @test device_of(second_buffer) == devices[2]
            extension.parallel_gpu_clear_caches!()
            extension._gpu_awareness(adapter, MPI.COMM_SELF, first_buffer)
            extension._gpu_awareness(adapter, MPI.COMM_SELF, second_buffer)
            @test extension.parallel_gpu_cache_sizes().awareness == 2
            prototype_result = extension._device_result(
                adapter, first_buffer, Float32[1], (), (),
            )
            @test device_of(prototype_result) == devices[1]
            current_probe = array_type{Float32}(undef, 1)
            @test device_of(current_probe) == devices[2]

            # hardware multi-device context and rejection
            # Staged copies and direct device collectives select the buffer's
            # real device and restore the caller's different current device.
            staged_result = extension._staged_gpu_call(
                adapter, :hardware_owner_device, MPI.COMM_SELF,
                host -> copy(host), first_buffer; validate_storage=false,
            )
            @test device_of(staged_result) == devices[1]
            @test device_of(array_type{Float32}(undef, 1)) == devices[2]
            forced_direct_adapter = extension.ParallelGPUAdapter(
                Symbol(vendor, :_forced_direct), adapter.matches,
                adapter.array_type, adapter.device, adapter.with_device,
                _ -> true, adapter.synchronize, adapter.allocate_pinned,
                adapter.device_to_host!, adapter.host_to_device!,
            )
            direct_callback_device = Ref{Any}()
            extension.allreduce!(
                first_buffer, +, MPI.COMM_SELF;
                adapter=forced_direct_adapter,
                collective=(buffer, _op, _comm) -> begin
                    direct_callback_device[] = device_of(
                        array_type{Float32}(undef, 1),
                    )
                    buffer
                end,
            )
            @test direct_callback_device[] == devices[1]
            @test device_of(array_type{Float32}(undef, 1)) == devices[2]

            staged_callback_device = Ref{Any}()
            forced_staged_adapter = extension.ParallelGPUAdapter(
                Symbol(vendor, :_forced_staged), adapter.matches,
                adapter.array_type, adapter.device, adapter.with_device,
                _ -> false, adapter.synchronize, adapter.allocate_pinned,
                adapter.device_to_host!, adapter.host_to_device!,
            )
            @test_throws ErrorException extension.allreduce!(
                first_buffer, +, MPI.COMM_SELF;
                adapter=forced_staged_adapter,
                collective=(_host, _op, _comm) -> begin
                    staged_callback_device[] = device_of(
                        array_type{Float32}(undef, 1),
                    )
                    error("staged MPI failed")
                end,
            )
            @test staged_callback_device[] == devices[2]
            @test device_of(array_type{Float32}(undef, 1)) == devices[2]

            direct_failure_device = Ref{Any}()
            @test_throws ErrorException extension.allreduce!(
                first_buffer, +, MPI.COMM_SELF;
                adapter=forced_direct_adapter,
                collective=(_buffer, _op, _comm) -> begin
                    direct_failure_device[] = device_of(
                        array_type{Float32}(undef, 1),
                    )
                    error("direct MPI failed")
                end,
            )
            @test direct_failure_device[] == devices[1]
            @test device_of(array_type{Float32}(undef, 1)) == devices[2]

            copy_failure_adapter = extension.ParallelGPUAdapter(
                Symbol(vendor, :_copy_failure), adapter.matches,
                adapter.array_type, adapter.device, adapter.with_device,
                _ -> false, adapter.synchronize, adapter.allocate_pinned,
                (_host, _device) -> error("device copy failed"),
                adapter.host_to_device!,
            )
            @test_throws ErrorException extension.allreduce!(
                first_buffer, +, MPI.COMM_SELF; adapter=copy_failure_adapter,
                collective=(host, _op, _comm) -> host,
            )
            @test device_of(array_type{Float32}(undef, 1)) == devices[2]

            cross_device_sentinel = Array(second_buffer)
            cross_device_calls = Ref(0)
            cross_device_stats = extension.parallel_gpu_stats()
            @test_throws ArgumentError extension.exchange!(
                first_buffer, second_buffer, MPI.COMM_SELF; adapter,
                collective=(_send, receive, _comm) -> begin
                    cross_device_calls[] += 1
                    fill!(receive, -1)
                end,
            )
            @test Array(second_buffer) == cross_device_sentinel
            @test cross_device_calls[] == 0
            @test extension.parallel_gpu_stats() == cross_device_stats
            @test device_of(array_type{Float32}(undef, 1)) == devices[2]
            MPI.Barrier(MPI.COMM_SELF)

            # Native batched vector/QST validation includes both FFT
            # workspaces and rejects a single workspace moved to device 2.
            activate_device!(devices[1])
            plan = SHTnsKit.DistTransposePlan(
                SHTnsKit.create_gauss_config(2, 4; nlon=6);
                comm=MPI.COMM_SELF, nlev=2, array_type, real_type=Float32,
                with_vector=true,
            )
            Vt = SHTnsKit.allocate_spatial(plan)
            Vp = SHTnsKit.allocate_spatial(plan)
            S = SHTnsKit.allocate_spectral(plan)
            T = SHTnsKit.allocate_spectral(plan)
            activate_device!(devices[2])
            bad_workspace = PencilArrays.PencilArray{eltype(plan.F_buf2)}(
                undef, PencilArrays.pencil(plan.F_buf2),
                PencilArrays.extra_dims(plan.F_buf2)...,
            )
            bad_plan = typeof(plan)(
                plan.cfg, plan.nlat, plan.nlon, plan.lmax, plan.mmax,
                plan.nlev, plan.comm, plan.fft_plan, plan.F_buf, bad_workspace,
                plan.spectral_pencil, plan.m_local, plan.NP, plan.dP, plan.Pos,
                plan.with_vector,
            )
            native_calls = extension.parallel_gpu_stats()
            @test_throws ArgumentError SHTnsKit.dist_analysis_sphtor!(
                bad_plan, S, T, Vt, Vp,
            )
            @test extension.parallel_gpu_stats() == native_calls
            @test device_of(array_type{Float32}(undef, 1)) == devices[2]
            MPI.Barrier(MPI.COMM_SELF)
        end

        # Distinct equal-size logical views of one vendor allocation retain
        # independent host snapshots in a multi-input staged callback.
        extension.parallel_gpu_clear_caches!()
        view_root = array_type(reshape(Float32.(1:4), 2, 2))
        first_view = view(view_root, 1, :)
        second_view = view(view_root, 2, :)
        @test extension._staged_gpu_call(
            adapter, :hardware_distinct_views, MPI.COMM_SELF,
            (first_host, second_host) -> sum(10 .* first_host .+ second_host),
            first_view, second_view; validate_storage=false,
        ) ≈ 46
        @test extension.parallel_gpu_cache_sizes().staging == 2
        activate_device!(assigned)
    end

    for RT in (Float32, Float64)
        CT = Complex{RT}
        tol = RT === Float32 ? RT(8e-4) : RT(8e-11)
        cfg = SHTnsKit.create_gauss_config(3, 8; nlon=8)
        scalar = fill(RT(0.25), cfg.nlat, cfg.nlon)
        spatial = _mpi_gpu_place(array_type, scalar, (1,), comm)

        @testset "scalar/vector/QST cfg parity" begin
            analyzed = SHTnsKit.analysis(cfg, spatial)
            _mpi_gpu_assert_resident(analyzed, is_vendor)
            synthesized = SHTnsKit.synthesis(
                cfg, analyzed; prototype_θφ=spatial,
            )
            _mpi_gpu_assert_resident(synthesized, is_vendor)
            @test _mpi_gpu_collect(synthesized, comm) ≈ scalar atol=tol rtol=tol

            packed = SHTnsKit.analysis_packed(cfg, spatial)
            _mpi_gpu_assert_resident(packed, is_vendor)
            packed_field = SHTnsKit.synthesis_packed(
                cfg, packed; prototype_θφ=spatial,
            )
            _mpi_gpu_assert_resident(packed_field, is_vendor)
            @test _mpi_gpu_collect(packed_field, comm) ≈ scalar atol=tol rtol=tol

            S = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
            T = zero(S)
            Q = zero(S)
            S[2, 1] = RT(0.12)
            T[3, 2] = CT(RT(-0.04), RT(0.02))
            Q[1, 1] = RT(0.18)
            Sd = _mpi_gpu_place(array_type, S, (2,), comm)
            Td = _mpi_gpu_place(array_type, T, (2,), comm)
            Qd = _mpi_gpu_place(array_type, Q, (2,), comm)
            vector = SHTnsKit.synthesis_sphtor(
                cfg, Sd, Td; prototype_θφ=spatial,
            )
            qst = SHTnsKit.synthesis_qst(
                cfg, Qd, Sd, Td; prototype_θφ=spatial,
            )
            _mpi_gpu_assert_resident(vector, is_vendor)
            _mpi_gpu_assert_resident(qst, is_vendor)
            _mpi_gpu_assert_resident(SHTnsKit.analysis_sphtor(cfg, vector...), is_vendor)
            _mpi_gpu_assert_resident(SHTnsKit.analysis_qst(cfg, qst...), is_vendor)
        end

        @testset "scalar packed complex axisym _l _ml compatibility" begin
            Q = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
            Q[1, 1] = RT(0.18)
            Q[2, 1] = RT(-0.07)
            Q[3, 2] = CT(RT(0.04), RT(-0.03))
            field = SHTnsKit.synthesis(cfg, Q; real_output=true)
            field_device = _mpi_gpu_place(array_type, field, (1,), comm)
            for ltr in (2, cfg.lmax)
                packed = SHTnsKit.analysis_packed_l(cfg, field_device, ltr)
                _mpi_gpu_assert_resident(packed, is_vendor)
                @test vec(_mpi_gpu_collect(packed, comm)) ≈
                      SHTnsKit.analysis_packed_l(cfg, vec(field), ltr) atol=tol rtol=tol
                rebuilt = SHTnsKit.synthesis_packed_l(
                    cfg, packed, ltr; prototype_θφ=field_device,
                )
                @test _mpi_gpu_collect(rebuilt, comm) ≈
                      reshape(SHTnsKit.synthesis_packed_l(
                          cfg, SHTnsKit.pack_lm(cfg, Q), ltr,
                      ), cfg.nlat, cfg.nlon) atol=tol rtol=tol
            end

            complex_field = complex.(field, RT(0.2) .* field)
            complex_device = _mpi_gpu_place(
                array_type, complex_field, (1,), comm,
            )
            complex_packed = SHTnsKit.analysis_packed_cplx_l(
                cfg, complex_device, 2,
            )
            _mpi_gpu_assert_resident(complex_packed, is_vendor)
            @test vec(_mpi_gpu_collect(complex_packed, comm)) ≈
                  SHTnsKit.analysis_packed_cplx_l(
                      cfg, vec(complex_field), 2,
                  ) atol=4tol rtol=4tol
            complex_rebuilt = SHTnsKit.synthesis_packed_cplx_l(
                cfg, complex_packed, 2; prototype_θφ=complex_device,
            )
            @test _mpi_gpu_collect(complex_rebuilt, comm) ≈
                  reshape(SHTnsKit.synthesis_packed_cplx_l(
                      cfg, vec(_mpi_gpu_collect(complex_packed, comm)), 2,
                  ), cfg.nlat, cfg.nlon) atol=4tol rtol=4tol

            axisym = CT[RT(0.2), RT(-0.08), RT(0.04), RT(-0.01)]
            axisym_field = SHTnsKit.synthesis_axisym(cfg, axisym)
            axisym_device = _mpi_gpu_place(
                array_type, axisym_field, (1,), comm,
            )
            axisym_back = SHTnsKit.analysis_axisym_l(
                cfg, axisym_device, 2,
            )
            _mpi_gpu_assert_resident(axisym_back, is_vendor)
            @test vec(_mpi_gpu_collect(axisym_back, comm)) ≈
                  SHTnsKit.analysis_axisym_l(cfg, axisym_field, 2) atol=tol rtol=tol
            @test vec(_mpi_gpu_collect(
                SHTnsKit.synthesis_axisym_l(cfg, axisym_back, 2), comm,
            )) ≈ SHTnsKit.synthesis_axisym_l(cfg, axisym, 2) atol=tol rtol=tol

            stored_im = 1
            mode = SHTnsKit.synthesis_packed_ml(
                cfg, stored_im, CT[RT(0.1), CT(RT(-0.03), RT(0.02)), RT(0.01)],
                cfg.lmax,
            )
            mode_device = _mpi_gpu_place(array_type, mode, (1,), comm)
            mode_back = SHTnsKit.analysis_packed_ml(
                cfg, stored_im, mode_device, cfg.lmax,
            )
            _mpi_gpu_assert_resident(mode_back, is_vendor)
            @test vec(_mpi_gpu_collect(mode_back, comm)) ≈
                  SHTnsKit.analysis_packed_ml(
                      cfg, stored_im, mode, cfg.lmax,
                  ) atol=4tol rtol=4tol
            mode_synthesized = SHTnsKit.synthesis_packed_ml(
                cfg, stored_im, mode_back, cfg.lmax,
            )
            _mpi_gpu_assert_resident(mode_synthesized, is_vendor)
            @test vec(_mpi_gpu_collect(mode_synthesized, comm)) ≈
                  mode atol=4tol rtol=4tol

            dist_packed = SHTnsKit.dist_analysis_packed(cfg, field_device)
            _mpi_gpu_assert_resident(dist_packed, is_vendor)
            @test Array(dist_packed) ≈ SHTnsKit.analysis_packed(
                cfg, vec(field),
            ) atol=tol rtol=tol
            dist_field = SHTnsKit.dist_synthesis_packed(
                cfg, dist_packed; prototype_θφ=field_device,
            )
            @test _mpi_gpu_collect_any(dist_field, comm) ≈ field atol=tol rtol=tol
        end

        @testset "batch sizes 1/2/5 and bang identity" begin
            for nfields in (1, 2, 5)
                host_fields = cat(
                    (scalar .* RT(1 + 0.1field) for field in 1:nfields)...;
                    dims=3,
                )
                fields = _mpi_gpu_place(
                    array_type, host_fields, (1,), comm,
                )
                coefficients = SHTnsKit.analysis_batch(cfg, fields)
                _mpi_gpu_assert_resident(coefficients, is_vendor)
                @test _mpi_gpu_collect(coefficients, comm) ≈
                      SHTnsKit.analysis_batch(cfg, host_fields) atol=tol rtol=tol
                coefficients_bang = similar(coefficients)
                @test SHTnsKit.analysis_batch!(
                    cfg, coefficients_bang, fields,
                ) === coefficients_bang
                @test _mpi_gpu_collect(coefficients_bang, comm) ≈
                      SHTnsKit.analysis_batch(cfg, host_fields) atol=tol rtol=tol
                reconstructed = SHTnsKit.synthesis_batch(
                    cfg, coefficients; prototype_θφ=fields,
                )
                reconstructed_bang = similar(reconstructed)
                @test SHTnsKit.synthesis_batch!(
                    cfg, reconstructed_bang, coefficients;
                    prototype_θφ=fields,
                ) === reconstructed_bang
                _mpi_gpu_assert_resident(reconstructed_bang, is_vendor)
                @test _mpi_gpu_collect(reconstructed_bang, comm) ≈
                      host_fields atol=tol rtol=tol
                complex_scalar = SHTnsKit.synthesis_batch_cplx(
                    cfg, coefficients; prototype_θφ=fields,
                )
                _mpi_gpu_assert_resident(complex_scalar, is_vendor)
                @test _mpi_gpu_collect(complex_scalar, comm) ≈
                      SHTnsKit.synthesis_batch_cplx(
                          cfg, SHTnsKit.analysis_batch(cfg, host_fields),
                      ) atol=4tol rtol=4tol

                Qbatch = _mpi_gpu_collect(coefficients, comm)
                Sbatch = RT(0.35) .* Qbatch
                Tbatch = RT(-0.2) .* Qbatch
                Sdevice = _mpi_gpu_place(array_type, Sbatch, (2,), comm)
                Tdevice = _mpi_gpu_place(array_type, Tbatch, (2,), comm)
                host_vector = SHTnsKit.synthesis_sphtor_batch(
                    cfg, Sbatch, Tbatch,
                )
                Vt = _mpi_gpu_place(array_type, host_vector[1], (1,), comm)
                Vp = _mpi_gpu_place(array_type, host_vector[2], (1,), comm)
                analyzed_vector = SHTnsKit.analysis_sphtor_batch(cfg, Vt, Vp)
                _mpi_gpu_assert_resident(analyzed_vector, is_vendor)
                @test _mpi_gpu_collect(analyzed_vector[1], comm) ≈
                      SHTnsKit.analysis_sphtor_batch(
                          cfg, host_vector[1], host_vector[2],
                      )[1] atol=4tol rtol=4tol
                synthesized_vector = SHTnsKit.synthesis_sphtor_batch(
                    cfg, Sdevice, Tdevice,
                )
                @test _mpi_gpu_collect(synthesized_vector[1], comm) ≈
                      host_vector[1] atol=4tol rtol=4tol
                complex_vector = SHTnsKit.synthesis_sphtor_batch_cplx(
                    cfg, Sdevice, Tdevice,
                )
                _mpi_gpu_assert_resident(complex_vector, is_vendor)
                host_complex_vector = SHTnsKit.synthesis_sphtor_batch_cplx(
                    cfg, Sbatch, Tbatch,
                )
                @test _mpi_gpu_collect(complex_vector[1], comm) ≈
                      host_complex_vector[1] atol=4tol rtol=4tol

                Qdevice = _mpi_gpu_place(array_type, Qbatch, (2,), comm)
                Vr = _mpi_gpu_place(array_type, host_fields, (1,), comm)
                analyzed_qst = SHTnsKit.analysis_qst_batch(cfg, Vr, Vt, Vp)
                _mpi_gpu_assert_resident(analyzed_qst, is_vendor)
                host_qst_analysis = SHTnsKit.analysis_qst_batch(
                    cfg, host_fields, host_vector[1], host_vector[2],
                )
                @test _mpi_gpu_collect(analyzed_qst[1], comm) ≈
                      host_qst_analysis[1] atol=4tol rtol=4tol
                synthesized_qst = SHTnsKit.synthesis_qst_batch(
                    cfg, Qdevice, Sdevice, Tdevice,
                )
                _mpi_gpu_assert_resident(synthesized_qst, is_vendor)
                @test _mpi_gpu_collect(synthesized_qst[1], comm) ≈
                      host_fields atol=4tol rtol=4tol
                complex_qst = SHTnsKit.synthesis_qst_batch_cplx(
                    cfg, Qdevice, Sdevice, Tdevice,
                )
                _mpi_gpu_assert_resident(complex_qst, is_vendor)
                host_complex_qst = SHTnsKit.synthesis_qst_batch_cplx(
                    cfg, Qbatch, Sbatch, Tbatch,
                )
                @test _mpi_gpu_collect(complex_qst[1], comm) ≈
                      host_complex_qst[1] atol=4tol rtol=4tol
            end
        end

        @testset "fixed/local/operator/rotation staged parity" begin
            S = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
            T = zero(S); Q = zero(S)
            S[2, 1] = RT(0.1); Q[1, 1] = RT(0.2)
            Sd = _mpi_gpu_place(array_type, S, (2,), comm)
            Td = _mpi_gpu_place(array_type, T, (2,), comm)
            Qd = _mpi_gpu_place(array_type, Q, (2,), comm)
            extension._reset_pencil_scalar_stats!()
            fixed = SHTnsKit.synthesis_sphtor_l(
                cfg, Sd, Td, 2; prototype_θφ=spatial,
            )
            _mpi_gpu_assert_resident(fixed, is_vendor)
            active = extension._pencil_scalar_stats()
            @test active.vector_synthesis_max_message_elements <=
                  cfg.nlat * (2length(0:cfg.mres:2) - 1)

            point = SHTnsKit.dist_SH_to_point(cfg, Qd, RT(0.2), RT(0.4))
            @test isfinite(point)
            latitude = SHTnsKit.dist_SH_to_lat(cfg, Qd, RT(0.2); nphi=5)
            _mpi_gpu_assert_resident(latitude, is_vendor)

            lap = copy(Qd)
            @test SHTnsKit.dist_apply_laplacian!(cfg, lap) === lap
            rotated = similar(Qd)
            @test SHTnsKit.dist_SH_Zrotate(
                cfg, Qd, RT(0.17), rotated,
            ) === rotated
            _mpi_gpu_assert_resident((lap, rotated), is_vendor)
        end

        @testset "vector QST _l _ml local gradient all operators" begin
            Q = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
            S = zero(Q); T = zero(Q)
            Q[1, 1] = RT(0.2); Q[3, 2] = CT(RT(0.03), RT(-0.02))
            S[2, 1] = RT(0.1); S[3, 2] = CT(RT(-0.04), RT(0.01))
            T[2, 1] = RT(-0.05); T[4, 2] = CT(RT(0.02), RT(0.03))
            Qd = _mpi_gpu_place(array_type, Q, (2,), comm)
            Sd = _mpi_gpu_place(array_type, S, (2,), comm)
            Td = _mpi_gpu_place(array_type, T, (2,), comm)
            Vr_host = SHTnsKit.synthesis(cfg, Q; real_output=true)
            Vt_host, Vp_host = SHTnsKit.synthesis_sphtor(
                cfg, S, T; real_output=true,
            )
            Vr = _mpi_gpu_place(array_type, Vr_host, (1,), comm)
            Vt = _mpi_gpu_place(array_type, Vt_host, (1,), comm)
            Vp = _mpi_gpu_place(array_type, Vp_host, (1,), comm)
            ltr = 2
            vector_l = SHTnsKit.analysis_sphtor_l(cfg, Vt, Vp, ltr)
            qst_l = SHTnsKit.analysis_qst_l(cfg, Vr, Vt, Vp, ltr)
            _mpi_gpu_assert_resident((vector_l, qst_l), is_vendor)
            vector_l_ref = SHTnsKit.analysis_sphtor_l(
                cfg, Vt_host, Vp_host, ltr,
            )
            qst_l_ref = SHTnsKit.analysis_qst_l(
                cfg, Vr_host, Vt_host, Vp_host, ltr,
            )
            @test _mpi_gpu_collect(vector_l[1], comm) ≈ vector_l_ref[1] atol=4tol rtol=4tol
            @test _mpi_gpu_collect(qst_l[1], comm) ≈ qst_l_ref[1] atol=4tol rtol=4tol
            synthesized_l = SHTnsKit.synthesis_qst_l(
                cfg, Qd, Sd, Td, ltr; prototype_θφ=Vr,
            )
            qst_synthesis_l_ref = SHTnsKit.synthesis_qst_l(
                cfg, Q, S, T, ltr; real_output=true,
            )
            @test _mpi_gpu_collect(synthesized_l[1], comm) ≈
                  qst_synthesis_l_ref[1] atol=4tol rtol=4tol
            complex_vector_l = SHTnsKit.synthesis_sphtor_l_cplx(
                cfg, Sd, Td, ltr; prototype_θφ=Vr,
            )
            complex_qst_l = SHTnsKit.synthesis_qst_l_cplx(
                cfg, Qd, Sd, Td, ltr; prototype_θφ=Vr,
            )
            _mpi_gpu_assert_resident((complex_vector_l, complex_qst_l), is_vendor)
            @test _mpi_gpu_collect(complex_vector_l[1], comm) ≈
                  SHTnsKit.synthesis_sphtor_l_cplx(
                      cfg, S, T, ltr,
                  )[1] atol=4tol rtol=4tol
            @test _mpi_gpu_collect(complex_qst_l[1], comm) ≈
                  SHTnsKit.synthesis_qst_l_cplx(
                      cfg, Q, S, T, ltr,
                  )[1] atol=4tol rtol=4tol
            for (operation, reference) in (
                (SHTnsKit.synthesis_sph_l,
                 SHTnsKit.synthesis_sph_l(cfg, S, ltr)),
                (SHTnsKit.synthesis_tor_l,
                 SHTnsKit.synthesis_tor_l(cfg, T, ltr)),
                (SHTnsKit.synthesis_grad_l,
                 SHTnsKit.synthesis_grad_l(cfg, S, ltr)),
            )
                input = operation === SHTnsKit.synthesis_tor_l ? Td : Sd
                value = operation(
                    cfg, input, ltr; prototype_θφ=Vr,
                )
                _mpi_gpu_assert_resident(value, is_vendor)
                @test _mpi_gpu_collect(value[1], comm) ≈ reference[1] atol=4tol rtol=4tol
            end
            for (operation, reference, input) in (
                (SHTnsKit.synthesis_sph_l_cplx,
                 SHTnsKit.synthesis_sph_l_cplx(cfg, S, ltr), Sd),
                (SHTnsKit.synthesis_tor_l_cplx,
                 SHTnsKit.synthesis_tor_l_cplx(cfg, T, ltr), Td),
            )
                value = operation(
                    cfg, input, ltr; prototype_θφ=Vr,
                )
                _mpi_gpu_assert_resident(value, is_vendor)
                @test _mpi_gpu_collect(value[1], comm) ≈ reference[1] atol=4tol rtol=4tol
                @test _mpi_gpu_collect(value[2], comm) ≈ reference[2] atol=4tol rtol=4tol
            end

            stored_im = 1
            m = stored_im * cfg.mres
            Qm = CT.(Q[(m + 1):end, m + 1])
            Sm = CT.(S[(m + 1):end, m + 1])
            Tm = CT.(T[(m + 1):end, m + 1])
            mode_qst = SHTnsKit.synthesis_qst_ml(
                cfg, stored_im, Qm, Sm, Tm, cfg.lmax,
            )
            mode_devices = map(mode_qst) do value
                _mpi_gpu_place(array_type, value, (1,), comm)
            end
            mode_back = SHTnsKit.analysis_qst_ml(
                cfg, stored_im, mode_devices..., cfg.lmax,
            )
            _mpi_gpu_assert_resident(mode_back, is_vendor)
            mode_ref = SHTnsKit.analysis_qst_ml(
                cfg, stored_im, mode_qst..., cfg.lmax,
            )
            @test vec(_mpi_gpu_collect(mode_back[1], comm)) ≈
                  mode_ref[1] atol=4tol rtol=4tol
            mode_rebuilt = SHTnsKit.synthesis_qst_ml(
                cfg, stored_im, mode_back..., cfg.lmax,
            )
            _mpi_gpu_assert_resident(mode_rebuilt, is_vendor)
            for component in 1:3
                @test vec(_mpi_gpu_collect(mode_rebuilt[component], comm)) ≈
                      mode_qst[component] atol=4tol rtol=4tol
            end
            mode_vector = SHTnsKit.synthesis_sphtor_ml(
                cfg, stored_im, mode_back[2], mode_back[3], cfg.lmax,
            )
            _mpi_gpu_assert_resident(mode_vector, is_vendor)
            @test vec(_mpi_gpu_collect(mode_vector[1], comm)) ≈
                  mode_qst[2] atol=4tol rtol=4tol
            for (operation, coefficient, reference) in (
                (SHTnsKit.synthesis_sph_ml, mode_back[2],
                 SHTnsKit.synthesis_sph_ml(
                     cfg, stored_im, mode_ref[2], cfg.lmax,
                 )),
                (SHTnsKit.synthesis_tor_ml, mode_back[3],
                 SHTnsKit.synthesis_tor_ml(
                     cfg, stored_im, mode_ref[3], cfg.lmax,
                 )),
            )
                value = operation(cfg, stored_im, coefficient, cfg.lmax)
                _mpi_gpu_assert_resident(value, is_vendor)
                @test vec(_mpi_gpu_collect(value[1], comm)) ≈
                      reference[1] atol=4tol rtol=4tol
            end
            @test vec(_mpi_gpu_collect(SHTnsKit.synthesis_grad_ml(
                cfg, stored_im, mode_back[2], cfg.lmax,
            ), comm)) ≈ SHTnsKit.synthesis_grad_ml(
                cfg, stored_im, mode_ref[2], cfg.lmax,
            ) atol=4tol rtol=4tol

            cost = RT(0.31); phi = RT(-0.27); nphi = 7
            packed_Q = SHTnsKit.pack_lm(cfg, Q)
            packed_S = SHTnsKit.pack_lm(cfg, S)
            packed_T = SHTnsKit.pack_lm(cfg, T)
            @test SHTnsKit.synthesis_point(cfg, Qd, cost, phi) ≈
                  SHTnsKit.synthesis_point(cfg, Q, cost, phi) atol=4tol rtol=4tol
            @test Array(SHTnsKit.SH_to_lat(cfg, Qd, cost; nphi)) ≈
                  SHTnsKit.SH_to_lat(cfg, packed_Q, cost; nphi) atol=4tol rtol=4tol
            @test collect(SHTnsKit.SHqst_to_point(
                cfg, Qd, Sd, Td, cost, phi,
            )) ≈ collect(SHTnsKit.SHqst_to_point(
                cfg, packed_Q, packed_S, packed_T, cost, phi,
            )) atol=4tol rtol=4tol
            qst_lat = SHTnsKit.SHqst_to_lat(
                cfg, Qd, Sd, Td, cost; nphi,
            )
            qst_lat_ref = SHTnsKit.SHqst_to_lat(
                cfg, packed_Q, packed_S, packed_T, cost; nphi,
            )
            for component in 1:3
                @test Array(qst_lat[component]) ≈ qst_lat_ref[component] atol=4tol rtol=4tol
            end
            grad = SHTnsKit.SH_to_grad_point(cfg, Qd, Sd, cost, phi)
            @test collect(grad) ≈ collect(SHTnsKit.SH_to_grad_point(
                cfg, packed_Q, packed_S, cost, phi,
            )) atol=4tol rtol=4tol

            for (operation, reference) in (
                (SHTnsKit.divergence_from_spheroidal,
                 SHTnsKit.divergence_from_spheroidal(cfg, S)),
                (SHTnsKit.spheroidal_from_divergence,
                 SHTnsKit.spheroidal_from_divergence(cfg, S)),
                (SHTnsKit.vorticity_from_toroidal,
                 SHTnsKit.vorticity_from_toroidal(cfg, T)),
                (SHTnsKit.toroidal_from_vorticity,
                 SHTnsKit.toroidal_from_vorticity(cfg, T)),
            )
                input = operation in (
                    SHTnsKit.vorticity_from_toroidal,
                    SHTnsKit.toroidal_from_vorticity,
                ) ? Td : Sd
                result = operation(cfg, input)
                _mpi_gpu_assert_resident(result, is_vendor)
                @test _mpi_gpu_collect(result, comm) ≈ reference atol=tol rtol=tol
            end
            for (operation!, input, reference) in (
                (SHTnsKit.divergence_from_spheroidal!, Sd,
                 SHTnsKit.divergence_from_spheroidal(cfg, S)),
                (SHTnsKit.spheroidal_from_divergence!, Sd,
                 SHTnsKit.spheroidal_from_divergence(cfg, S)),
                (SHTnsKit.vorticity_from_toroidal!, Td,
                 SHTnsKit.vorticity_from_toroidal(cfg, T)),
                (SHTnsKit.toroidal_from_vorticity!, Td,
                 SHTnsKit.toroidal_from_vorticity(cfg, T)),
            )
                output = similar(input)
                @test operation!(cfg, output, input) === output
                _mpi_gpu_assert_resident(output, is_vendor)
                @test _mpi_gpu_collect(output, comm) ≈ reference atol=tol rtol=tol
            end
            mx = zeros(RT, 2cfg.nlm)
            SHTnsKit.mul_ct_matrix(SHTnsKit.CPU(), cfg, mx)
            neighbour = similar(Qd)
            SHTnsKit.SH_mul_mx(SHTnsKit.CPU(), cfg, mx, Qd, neighbour)
            dense_neighbour = zeros(CT, size(Q))
            SHTnsKit.dist_SH_mul_mx!(cfg, mx, Q, dense_neighbour)
            @test _mpi_gpu_collect(neighbour, comm) ≈ dense_neighbour atol=tol rtol=tol
            divergence_grid = SHTnsKit.dist_spatial_divergence(
                cfg, Sd, Td; prototype_θφ=Vr,
            )
            vorticity_grid = SHTnsKit.dist_spatial_vorticity(
                cfg, Sd, Td; prototype_θφ=Vr,
            )
            @test _mpi_gpu_collect_any(divergence_grid, comm) ≈
                  SHTnsKit.synthesis(
                      cfg, SHTnsKit.divergence_from_spheroidal(cfg, S),
                  ) atol=4tol rtol=4tol
            @test _mpi_gpu_collect_any(vorticity_grid, comm) ≈
                  SHTnsKit.synthesis(
                      cfg, SHTnsKit.vorticity_from_toroidal(cfg, T),
                  ) atol=4tol rtol=4tol
            laplacian_expected = copy(Q)
            SHTnsKit.dist_apply_laplacian!(cfg, laplacian_expected)
            laplacian_grid = SHTnsKit.dist_scalar_laplacian(
                cfg, Vr; prototype_θφ=Vr,
            )
            @test _mpi_gpu_collect_any(laplacian_grid, comm) ≈
                  SHTnsKit.synthesis(cfg, laplacian_expected) atol=4tol rtol=4tol
            laplacian_output = similar(Vr)
            @test SHTnsKit.dist_scalar_laplacian!(
                cfg, laplacian_output, Vr,
            ) === laplacian_output
            @test _mpi_gpu_collect(laplacian_output, comm) ≈
                  SHTnsKit.synthesis(cfg, laplacian_expected) atol=4tol rtol=4tol
        end

        @testset "general rotations diagnostics storage and compatibility" begin
            Q = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
            Q[1, 1] = RT(0.2); Q[2, 1] = RT(-0.05)
            Q[3, 2] = CT(RT(0.04), RT(-0.03))
            S = RT(0.6) .* Q; T = RT(-0.35) .* Q
            Qd = _mpi_gpu_place(array_type, Q, (2,), comm)
            Sd = _mpi_gpu_place(array_type, S, (2,), comm)
            Td = _mpi_gpu_place(array_type, T, (2,), comm)
            alpha, beta, gamma = RT(0.17), RT(-0.31), RT(0.23)
            first = zeros(CT, size(Q)); second = similar(first); expected = similar(first)
            SHTnsKit.dist_SH_Zrotate(cfg, Q, alpha, first)
            SHTnsKit.dist_SH_Yrotate(cfg, first, beta, second)
            SHTnsKit.dist_SH_Zrotate(cfg, second, gamma, expected)
            rotated = similar(Qd)
            @test SHTnsKit.dist_SH_rotate_euler(
                cfg, Qd, alpha, beta, gamma, rotated,
            ) === rotated
            @test _mpi_gpu_collect(rotated, comm) ≈ expected atol=4tol rtol=4tol
            y = similar(Qd); x90 = similar(Qd); y90 = similar(Qd)
            SHTnsKit.dist_SH_Yrotate(cfg, Qd, beta, y)
            SHTnsKit.dist_SH_Xrotate90(cfg, Qd, x90)
            SHTnsKit.dist_SH_Yrotate90(cfg, Qd, y90)
            for value in (y, x90, y90)
                _mpi_gpu_assert_resident(value, is_vendor)
            end
            expected_y = zeros(CT, size(Q))
            SHTnsKit.dist_SH_Yrotate(cfg, Q, beta, expected_y)
            for operation! in (
                SHTnsKit.dist_SH_Yrotate_allgatherm!,
                SHTnsKit.dist_SH_Yrotate_truncgatherm!,
            )
                output = similar(Qd)
                @test operation!(cfg, Qd, beta, output) === output
                @test _mpi_gpu_collect(output, comm) ≈ expected_y atol=4tol rtol=4tol
            end
            packed = array_type(SHTnsKit.pack_lm(cfg, Q))
            packed_host = SHTnsKit.pack_lm(cfg, Q)
            packed_z = SHTnsKit.dist_SH_Zrotate_packed(
                cfg, packed, alpha; prototype_lm=Qd,
            )
            packed_y = SHTnsKit.dist_SH_Yrotate_packed(
                cfg, packed, beta; prototype_lm=Qd,
            )
            packed_y90 = SHTnsKit.dist_SH_Yrotate90_packed(
                cfg, packed; prototype_lm=Qd,
            )
            packed_x90 = SHTnsKit.dist_SH_Xrotate90_packed(
                cfg, packed; prototype_lm=Qd,
            )
            expected_z_packed = similar(packed_host)
            SHTnsKit.SH_Zrotate(cfg, packed_host, alpha, expected_z_packed)
            @test Array(packed_z) ≈ expected_z_packed atol=4tol rtol=4tol
            @test Array(packed_y) ≈ SHTnsKit.SH_Yrotate(
                cfg, packed_host, beta, similar(packed_host),
            ) atol=4tol rtol=4tol
            @test Array(packed_y90) ≈ SHTnsKit.SH_Yrotate90(
                cfg, packed_host, similar(packed_host),
            ) atol=4tol rtol=4tol
            @test Array(packed_x90) ≈ SHTnsKit.SH_Xrotate90(
                cfg, packed_host, similar(packed_host),
            ) atol=4tol rtol=4tol
            _mpi_gpu_assert_resident(
                (packed_z, packed_y, packed_y90, packed_x90), is_vendor,
            )

            scalar_energy = SHTnsKit.energy_scalar(cfg, Qd)
            @test scalar_energy ≈ SHTnsKit.energy_scalar(cfg, Q) atol=tol rtol=tol
            @test SHTnsKit.energy_scalar_l_spectrum(cfg, Qd) ≈
                  SHTnsKit.energy_scalar_l_spectrum(cfg, Q) atol=tol rtol=tol
            @test SHTnsKit.energy_scalar_m_spectrum(cfg, Qd) ≈
                  SHTnsKit.energy_scalar_m_spectrum(cfg, Q) atol=tol rtol=tol
            @test SHTnsKit.energy_vector_l_spectrum(cfg, Sd, Td) ≈
                  SHTnsKit.energy_vector_l_spectrum(cfg, S, T) atol=tol rtol=tol
            @test SHTnsKit.energy_vector_m_spectrum(cfg, Sd, Td) ≈
                  SHTnsKit.energy_vector_m_spectrum(cfg, S, T) atol=tol rtol=tol
            @test SHTnsKit.enstrophy_l_spectrum(cfg, Td) ≈
                  SHTnsKit.enstrophy_l_spectrum(cfg, T) atol=tol rtol=tol
            @test SHTnsKit.enstrophy_m_spectrum(cfg, Td) ≈
                  SHTnsKit.enstrophy_m_spectrum(cfg, T) atol=tol rtol=tol
            field = SHTnsKit.synthesis(cfg, Q)
            field_device = _mpi_gpu_place(array_type, field, (1,), comm)
            @test SHTnsKit.grid_energy_scalar(cfg, field_device) ≈
                  SHTnsKit.grid_energy_scalar(cfg, field) atol=4tol rtol=4tol
            @test SHTnsKit.grid_enstrophy(cfg, field_device) ≈
                  SHTnsKit.grid_enstrophy(cfg, field) atol=4tol rtol=4tol
            Vt_host, Vp_host = SHTnsKit.synthesis_sphtor(cfg, S, T)
            Vt = _mpi_gpu_place(array_type, Vt_host, (1,), comm)
            Vp = _mpi_gpu_place(array_type, Vp_host, (1,), comm)
            @test SHTnsKit.grid_energy_vector(cfg, Vt, Vp) ≈
                  SHTnsKit.grid_energy_vector(cfg, Vt_host, Vp_host) atol=4tol rtol=4tol

            # Preserved dist_* compatibility paths are compared with the same
            # independent serial CPU transforms, not with a round-trip oracle.
            compat_Q = SHTnsKit.dist_analysis(cfg, field_device)
            @test _mpi_gpu_collect_any(compat_Q, comm) ≈
                  SHTnsKit.analysis(cfg, field) atol=4tol rtol=4tol
            compat_field = SHTnsKit.dist_synthesis(
                cfg, Qd; prototype_θφ=field_device,
            )
            @test _mpi_gpu_collect_any(compat_field, comm) ≈ field atol=4tol rtol=4tol
            compat_ST = SHTnsKit.dist_analysis_sphtor(cfg, Vt, Vp)
            @test _mpi_gpu_collect_any(compat_ST[1], comm) ≈
                  SHTnsKit.analysis_sphtor(cfg, Vt_host, Vp_host)[1] atol=4tol rtol=4tol
            compat_QST = SHTnsKit.dist_analysis_qst(
                cfg, field_device, Vt, Vp,
            )
            @test _mpi_gpu_collect_any(compat_QST[1], comm) ≈
                  SHTnsKit.analysis(cfg, field) atol=4tol rtol=4tol
            _mpi_gpu_assert_resident((Qd, Sd, Td, rotated), is_vendor)
            @test eltype(parent(Qd)) === CT
        end

        @testset "native scalar/vector/QST transpose nonzero numerics" begin
            # DistTransposePlan deliberately exposes the canonical
            # orthonormal+CS convention; other conventions are covered above
            # by cfg-form parity and converted at that public boundary.
            @test cfg.norm === :orthonormal
            @test cfg.cs_phase
            @test !cfg.real_norm
            staged_before = extension.parallel_gpu_stats().staged_calls
            for nlev in (1, 2, 5)
                plan = SHTnsKit.DistTransposePlan(
                    cfg; comm, nlev, array_type, real_type=RT,
                    with_vector=true,
                )
                refs = _mpi_gpu_native_references(cfg, RT, nlev)
                Vr = SHTnsKit.allocate_spatial(plan)
                Vt = SHTnsKit.allocate_spatial(plan)
                Vp = SHTnsKit.allocate_spatial(plan)
                _mpi_gpu_fill_native_spatial!(Vr, refs.Vr)
                _mpi_gpu_fill_native_spatial!(Vt, refs.Vt)
                _mpi_gpu_fill_native_spatial!(Vp, refs.Vp)
                Q = SHTnsKit.allocate_spectral(plan)
                S = SHTnsKit.allocate_spectral(plan)
                T = SHTnsKit.allocate_spectral(plan)

                # Analysis is checked against independent serial CPU analysis,
                # not against a distributed round-trip result.
                @test SHTnsKit.dist_analysis!(plan, Q, Vr) === Q
                scalar_error = _mpi_gpu_native_spectral_error(
                    Q, refs.Qanalysis, plan.m_local,
                )
                @test MPI.Allreduce(scalar_error, MPI.MAX, comm) <= tol
                @test SHTnsKit.dist_analysis_sphtor!(
                    plan, S, T, Vt, Vp,
                ) === (S, T)
                vector_error = max(
                    _mpi_gpu_native_spectral_error(
                        S, refs.Sanalysis, plan.m_local,
                    ),
                    _mpi_gpu_native_spectral_error(
                        T, refs.Tanalysis, plan.m_local,
                    ),
                )
                @test MPI.Allreduce(vector_error, MPI.MAX, comm) <= 4tol
                @test SHTnsKit.dist_analysis_qst!(
                    plan, Q, S, T, Vr, Vt, Vp,
                ) === (Q, S, T)
                qst_error = max(
                    _mpi_gpu_native_spectral_error(
                        Q, refs.Qanalysis, plan.m_local,
                    ),
                    _mpi_gpu_native_spectral_error(
                        S, refs.Sanalysis, plan.m_local,
                    ),
                    _mpi_gpu_native_spectral_error(
                        T, refs.Tanalysis, plan.m_local,
                    ),
                )
                @test MPI.Allreduce(qst_error, MPI.MAX, comm) <= 4tol

                # Synthesis starts from independent nonzero CPU coefficients.
                _mpi_gpu_fill_native_spectral!(Q, refs.Q, plan.m_local)
                _mpi_gpu_fill_native_spectral!(S, refs.S, plan.m_local)
                _mpi_gpu_fill_native_spectral!(T, refs.T, plan.m_local)
                @test SHTnsKit.dist_synthesis!(plan, Vr, Q) === Vr
                scalar_error = _mpi_gpu_native_spatial_error(Vr, refs.Vr)
                @test MPI.Allreduce(scalar_error, MPI.MAX, comm) <= tol
                @test SHTnsKit.dist_synthesis_sphtor!(
                    plan, Vt, Vp, S, T,
                ) === (Vt, Vp)
                vector_error = max(
                    _mpi_gpu_native_spatial_error(Vt, refs.Vt),
                    _mpi_gpu_native_spatial_error(Vp, refs.Vp),
                )
                @test MPI.Allreduce(vector_error, MPI.MAX, comm) <= 4tol
                @test SHTnsKit.dist_synthesis_qst!(
                    plan, Vr, Vt, Vp, Q, S, T,
                ) === (Vr, Vt, Vp)
                qst_error = max(
                    _mpi_gpu_native_spatial_error(Vr, refs.Vr),
                    _mpi_gpu_native_spatial_error(Vt, refs.Vt),
                    _mpi_gpu_native_spatial_error(Vp, refs.Vp),
                )
                @test MPI.Allreduce(qst_error, MPI.MAX, comm) <= 4tol
                _mpi_gpu_assert_resident((Vr, Vt, Vp, Q, S, T), is_vendor)
                @test extension.parallel_gpu_stats().staged_calls == staged_before

                if nlev == 2
                    # QST must reject the complete six-array payload before
                    # mutation, staging, FFT work, or communication side effects.
                    WrongRT = RT === Float32 ? Float64 : Float32
                    bad_vp = PencilArrays.PencilArray{WrongRT}(
                        undef, PencilArrays.pencil(Vp), nlev,
                    )
                    fill!(parent(bad_vp), WrongRT(0.125))
                    spectral_sentinel = CT(RT(73), RT(-19))
                    fill!(parent(Q), spectral_sentinel)
                    fill!(parent(S), spectral_sentinel)
                    fill!(parent(T), spectral_sentinel)
                    before = extension.parallel_gpu_stats()
                    caught = false
                    try
                        SHTnsKit.dist_analysis_qst!(
                            plan, Q, S, T, Vr, Vt, bad_vp,
                        )
                    catch error
                        caught = error isa ArgumentError
                    end
                    @test MPI.Allreduce(caught ? 1 : 0, min, comm) == 1
                    @test all(==(spectral_sentinel), Array(parent(Q)))
                    @test all(==(spectral_sentinel), Array(parent(S)))
                    @test all(==(spectral_sentinel), Array(parent(T)))
                    @test extension.parallel_gpu_stats() == before
                    MPI.Barrier(comm)

                    bad_t = PencilArrays.PencilArray{Complex{WrongRT}}(
                        undef, PencilArrays.pencil(T), nlev,
                    )
                    fill!(parent(bad_t), Complex{WrongRT}(0.1, -0.2))
                    spatial_sentinel = RT(91)
                    fill!(parent(Vr), spatial_sentinel)
                    fill!(parent(Vt), spatial_sentinel)
                    fill!(parent(Vp), spatial_sentinel)
                    before = extension.parallel_gpu_stats()
                    caught = false
                    try
                        SHTnsKit.dist_synthesis_qst!(
                            plan, Vr, Vt, Vp, Q, S, bad_t,
                        )
                    catch error
                        caught = error isa ArgumentError
                    end
                    @test MPI.Allreduce(caught ? 1 : 0, min, comm) == 1
                    @test all(==(spatial_sentinel), Array(parent(Vr)))
                    @test all(==(spatial_sentinel), Array(parent(Vt)))
                    @test all(==(spatial_sentinel), Array(parent(Vp)))
                    @test extension.parallel_gpu_stats() == before
                    MPI.Barrier(comm)
                end
            end
            @test extension.parallel_gpu_stats().staged_calls == staged_before
        end

        @testset "repeated-plan cache and residency" begin
            plan = SHTnsKit.DistTransposePlan(
                cfg; comm, nlev=2, array_type, real_type=RT,
            )
            input = SHTnsKit.allocate_spatial(plan)
            output = SHTnsKit.allocate_spectral(plan)
            fill!(parent(input), RT(0.125))
            for _ in 1:3
                @test SHTnsKit.dist_analysis!(plan, output, input) === output
            end
            _mpi_gpu_assert_resident((input, output), is_vendor)
        end
    end
    return nothing
end

function test_mpi_gpu_policy(extension)
    @test isdefined(extension, :ParallelGPUAdapter)
    @test isdefined(extension, :exchange!)
    @test isdefined(extension, :allreduce!)
    @test isdefined(extension, :parallel_gpu_stats)
    @test isdefined(extension, :parallel_gpu_clear_caches!)
    @test isdefined(extension, :parallel_gpu_cache_sizes)

    aware_calls = Ref(0)
    sync_calls = Ref(0)
    host_allocations = Ref(0)
    host_to_device = Ref(0)
    device_to_host = Ref(0)
    direct_collectives = Ref(0)
    staged_collectives = Ref(0)

    adapter = extension.ParallelGPUAdapter(
        :mock,
        value -> value isa MockMPIArray,
        _ -> MockMPIArray,
        _ -> 7,
        _ -> (aware_calls[] += 1; false),
        _ -> (sync_calls[] += 1),
        (T, n) -> (host_allocations[] += 1; Vector{T}(undef, n)),
        (host, device) -> (device_to_host[] += 1; copyto!(host, device)),
        (device, host) -> (host_to_device[] += 1; copyto!(device, host)),
    )

    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)
    buffer = MockMPIArray(Float32[1, 2, 3])
    comm = Ref(:subgroup_a)
    collective = function(host, _op, _comm)
        staged_collectives[] += 1
        host .*= 2
        return host
    end
    extension.allreduce!(buffer, +, comm; adapter, collective)
    extension.allreduce!(buffer, +, comm; adapter, collective)
    @test buffer.data == Float32[4, 8, 12]
    @test aware_calls[] == 1
    @test host_allocations[] == 1
    @test device_to_host[] == 2
    @test host_to_device[] == 2
    @test sync_calls[] == 6
    @test staged_collectives[] == 2
    @test extension.parallel_gpu_cache_sizes().staging == 1

    # Device-aware cache keys follow the allocation behind views, not the
    # current task device.  Reusing an allocation after its fake device changes
    # must create independent awareness and pinned-staging entries.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(8)
    multi = MockMultiDeviceArray(reshape(Float32.(1:4), 2, 2), 1)
    multi_view = view(multi, :, :)
    current_device = Ref(99)
    multi_host_allocations = Ref(0)
    multi_adapter = extension.ParallelGPUAdapter(
        :mock_multidevice,
        value -> extension._parallel_root_buffer(value) isa MockMultiDeviceArray,
        _ -> MockMultiDeviceArray,
        value -> extension._parallel_root_buffer(value).device,
        (f, device) -> begin
            previous = MOCK_CURRENT_DEVICE[]
            try
                MOCK_CURRENT_DEVICE[] = device
                f()
            finally
                MOCK_CURRENT_DEVICE[] = previous
            end
        end,
        _ -> false,
        _ -> nothing,
        (T, n) -> (multi_host_allocations[] += 1; Vector{T}(undef, n)),
        copyto!, copyto!,
    )
    multi_comm = Ref(:multidevice_subgroup)

    # Every sync/copy must run on the physical device owning that buffer while
    # host MPI/CPU callbacks run on the caller's original current device.
    context_events = Any[]
    context_aware = Ref(false)
    context_copy_error = Ref(false)
    context_adapter = extension.ParallelGPUAdapter(
        :mock_device_context,
        multi_adapter.matches, multi_adapter.array_type, multi_adapter.device,
        (f, device) -> begin
            previous = MOCK_CURRENT_DEVICE[]
            push!(context_events, (:enter, device, previous))
            try
                MOCK_CURRENT_DEVICE[] = device
                f()
            finally
                MOCK_CURRENT_DEVICE[] = previous
                push!(context_events, (:exit, device, previous))
            end
        end,
        _ -> context_aware[],
        value -> push!(context_events, (
            :sync, MOCK_CURRENT_DEVICE[], multi_adapter.device(value),
        )),
        (T, n) -> Vector{T}(undef, n),
        (host, value) -> begin
            push!(context_events, (
                :device_to_host, MOCK_CURRENT_DEVICE[],
                multi_adapter.device(value),
            ))
            context_copy_error[] && error("device copy failed")
            copyto!(host, value)
        end,
        (value, host) -> begin
            push!(context_events, (
                :host_to_device, MOCK_CURRENT_DEVICE[],
                multi_adapter.device(value),
            ))
            context_copy_error[] && error("device copy failed")
            copyto!(value, host)
        end,
    )
    context_value = MockMultiDeviceArray(Float32[1, 2], 1)
    context_receive = MockMultiDeviceArray(zeros(Float32, 2), 1)
    MOCK_CURRENT_DEVICE[] = 2

    function assert_context_events(events, callback_device)
        device_events = filter(
            event -> first(event) in (:sync, :device_to_host, :host_to_device),
            events,
        )
        @test !isempty(device_events)
        @test all(event -> event[2] == event[3] == 1, device_events)
        @test all(event -> event[2] == callback_device,
                  filter(event -> first(event) == :host, events))
        @test MOCK_CURRENT_DEVICE[] == 2
    end

    extension._staged_gpu_call(
        context_adapter, :device_context_staged_math, MPI.COMM_SELF,
        host -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            host .+= 1
            host
        end,
        context_value; mutated=(1,), validate_storage=false,
    )
    assert_context_events(context_events, 2)

    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(host, _op, _comm) -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            host
        end,
    )
    assert_context_events(context_events, 2)

    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    extension.exchange!(
        context_value, context_receive, MPI.COMM_SELF; adapter=context_adapter,
        collective=(send, receive, _comm) -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            copyto!(receive, send)
        end,
    )
    assert_context_events(context_events, 2)

    context_aware[] = true
    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(device, _op, _comm) -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            device
        end,
    )
    assert_context_events(context_events, 1)

    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    extension.exchange!(
        context_value, context_receive, MPI.COMM_SELF; adapter=context_adapter,
        collective=(send, receive, _comm) -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            copyto!(receive, send)
        end,
    )
    assert_context_events(context_events, 1)

    # Device scopes restore caller state after copy and MPI exceptions.
    context_aware[] = false
    context_copy_error[] = true
    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    @test_throws ErrorException extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(host, _op, _comm) -> host,
    )
    @test MOCK_CURRENT_DEVICE[] == 2
    context_copy_error[] = false
    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    @test_throws ErrorException extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(_host, _op, _comm) -> error("MPI failed"),
    )
    @test MOCK_CURRENT_DEVICE[] == 2

    context_aware[] = true
    extension.parallel_gpu_clear_caches!()
    @test_throws ErrorException extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(_device, _op, _comm) -> error("direct MPI failed"),
    )
    @test MOCK_CURRENT_DEVICE[] == 2
    extension.parallel_gpu_clear_caches!()
    @test_throws ErrorException extension.exchange!(
        context_value, context_receive, MPI.COMM_SELF;
        adapter=context_adapter,
        collective=(_send, _receive, _comm) -> error("direct MPI failed"),
    )
    @test MOCK_CURRENT_DEVICE[] == 2
    context_aware[] = false

    # Same-vendor buffers on different local devices are a collective error
    # before callbacks, communication counters, or output mutation.
    cross_device = MockMultiDeviceArray(Float32[9, 10], 2)
    @test_throws ArgumentError extension._validate_parallel_storage!(
        MPI.COMM_SELF, :cross_device_fake, context_value, cross_device;
        adapter=context_adapter,
    )
    cross_device_sentinel = copy(cross_device.data)
    cross_device_calls = Ref(0)
    before_cross_device = extension.parallel_gpu_stats()
    @test_throws ArgumentError extension.exchange!(
        context_value, cross_device, MPI.COMM_SELF; adapter=context_adapter,
        collective=(_send, receive, _comm) -> begin
            cross_device_calls[] += 1
            fill!(receive, -1)
        end,
    )
    @test cross_device.data == cross_device_sentinel
    @test cross_device_calls[] == 0
    @test extension.parallel_gpu_stats() == before_cross_device
    MPI.Barrier(MPI.COMM_SELF)

    extension.parallel_gpu_clear_caches!()
    multi_host_allocations[] = 0
    # Equal-size logical views of one allocation need independent staging
    # snapshots. Sharing a root-keyed entry overwrites the first input before
    # the CPU callback observes it.
    first_view = view(multi, 1, :)
    second_view = view(multi, 2, :)
    distinct_view_result = extension._staged_gpu_call(
        multi_adapter, :distinct_views, multi_comm,
        (first_host, second_host) -> sum(10 .* first_host .+ second_host),
        first_view, second_view; validate_storage=false,
    )
    @test distinct_view_result == 46
    @test multi_host_allocations[] == 2
    @test extension.parallel_gpu_cache_sizes().staging == 2
    @test extension._staged_gpu_call(
        multi_adapter, :distinct_views, multi_comm,
        (first_host, second_host) -> sum(10 .* first_host .+ second_host),
        first_view, second_view; validate_storage=false,
    ) == 46
    @test multi_host_allocations[] == 2

    # Out-of-place restoration must allocate on the prototype's physical
    # device, not whichever device happens to be current in this task.
    MOCK_CURRENT_DEVICE[] = 2
    prototype_device_result = extension._device_result(
        multi_adapter, multi, Float32[7, 8], (), (),
    )
    @test prototype_device_result isa MockMultiDeviceArray
    @test prototype_device_result.device == multi.device
    @test MOCK_CURRENT_DEVICE[] == 2
    failing_device_adapter = extension.ParallelGPUAdapter(
        :mock_multidevice_failure, multi_adapter.matches,
        multi_adapter.array_type, multi_adapter.device,
        multi_adapter.with_device, multi_adapter.gpu_aware,
        multi_adapter.synchronize, multi_adapter.allocate_pinned,
        multi_adapter.device_to_host!,
        (_device, _host) -> error("copy to device failed"),
    )
    @test_throws ErrorException extension._device_result(
        failing_device_adapter, multi, Float32[7, 8], (), (),
    )
    @test MOCK_CURRENT_DEVICE[] == 2

    # Every standard array wrapper admitted by a vendor adapter must resolve
    # to the allocation that owns its physical device and cache lifetime.
    mask_root = MockMultiDeviceArray(Bool[true, false, true, false], 1)
    wrappers = (
        PermutedDimsArray(multi, (2, 1)), adjoint(multi), transpose(multi),
        Symmetric(multi), Hermitian(multi), UpperTriangular(multi),
        LowerTriangular(multi), UnitUpperTriangular(multi),
        UnitLowerTriangular(multi), Diagonal(view(multi, 1:2, 1)),
        Bidiagonal(view(multi, 1:2, 1), view(multi, 1:1, 2), :U),
        Tridiagonal(
            view(multi, 1:1, 1), view(multi, 1:2, 1), view(multi, 1:1, 2),
        ),
        Base.LogicalIndex(mask_root),
    )
    expected_roots = (ntuple(_ -> multi, length(wrappers) - 1)..., mask_root)
    for (wrapper, expected_root) in zip(wrappers, expected_roots)
        @test extension._parallel_root_buffer(wrapper) === expected_root
        @test multi_adapter.matches(wrapper)
        @test multi_adapter.device(wrapper) == expected_root.device
    end
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(length(wrappers) + 1)
    for wrapper in wrappers
        extension._staging_entry(
            multi_adapter, multi_comm, wrapper, length(wrapper),
        )
    end
    @test extension.parallel_gpu_cache_sizes().staging == length(wrappers)

    # Live logical wrappers cannot make the registry exceed its cap, and a
    # cached view does not keep its physical allocation alive.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)
    for wrapper in wrappers
        extension._staging_entry(
            multi_adapter, multi_comm, wrapper, length(wrapper),
        )
        @test extension.parallel_gpu_cache_sizes().staging <= 2
    end
    extension.parallel_gpu_clear_caches!()
    weak_root = _cache_temporary_view!(extension, multi_adapter, multi_comm)
    @test extension.parallel_gpu_cache_sizes().staging == 1
    GC.gc(true)
    @test weak_root.value === nothing
    @test extension.parallel_gpu_cache_sizes().staging == 0

    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(8)
    extension.allreduce!(multi_view, +, multi_comm; adapter=multi_adapter,
                         collective=(host, _op, _comm) -> host)
    multi.device = 2
    extension.allreduce!(multi_view, +, multi_comm; adapter=multi_adapter,
                         collective=(host, _op, _comm) -> host)
    @test current_device[] == 99
    @test extension.parallel_gpu_cache_sizes() == (awareness=2, staging=2)

    # A staged collective cannot return until its host-to-device copy has been
    # synchronized. Per-entry locks must also be released on MPI/copy errors.
    extension.parallel_gpu_clear_caches!()
    events = Symbol[]
    throw_copy = Ref(false)
    event_adapter = extension.ParallelGPUAdapter(
        :mock_events,
        value -> value isa MockMPIArray,
        _ -> MockMPIArray,
        _ -> 1,
        _ -> false,
        _ -> push!(events, :sync),
        (T, n) -> Vector{T}(undef, n),
        (host, device) -> (push!(events, :device_to_host); copyto!(host, device)),
        (device, host) -> begin
            push!(events, :host_to_device)
            throw_copy[] && error("copy-back failed")
            copyto!(device, host)
        end,
    )
    extension._register_parallel_gpu_adapter!(event_adapter)
    event_buffer = MockMPIArray(Float32[1, 2])
    event_comm = Ref(:event_subgroup)
    extension.allreduce!(
        event_buffer, +, event_comm; adapter=event_adapter,
        collective=(host, _op, _comm) -> (push!(events, :collective); host),
    )
    @test events == [
        :sync, :device_to_host, :sync, :collective,
        :host_to_device, :sync,
    ]

    empty!(events)
    event_receive = MockMPIArray(zeros(Float32, 2))
    extension.exchange!(
        event_buffer, event_receive, MPI.COMM_SELF;
        adapter=event_adapter,
        collective=(send, receive, _comm) -> begin
            push!(events, :collective)
            copyto!(receive, send)
        end,
    )
    @test events == [
        :sync, :device_to_host, :sync, :collective,
        :host_to_device, :sync,
    ]

    empty!(events)
    @test_throws ErrorException extension.allreduce!(
        event_buffer, +, Ref(:event_mpi_throw); adapter=event_adapter,
        collective=(_host, _op, _comm) -> begin
            push!(events, :collective_throw)
            error("MPI failed")
        end,
    )
    @test events == [:sync, :device_to_host, :sync, :collective_throw]
    empty!(events)
    extension.allreduce!(event_buffer, +, Ref(:event_mpi_throw);
                         adapter=event_adapter,
                         collective=(host, _op, _comm) -> host)
    @test last(events) == :sync

    empty!(events)
    throw_copy[] = true
    @test_throws ErrorException extension.allreduce!(
        event_buffer, +, Ref(:event_copy_throw); adapter=event_adapter,
        collective=(host, _op, _comm) -> host,
    )
    @test last(events) == :host_to_device
    throw_copy[] = false
    empty!(events)
    extension.allreduce!(event_buffer, +, Ref(:event_copy_throw);
                         adapter=event_adapter,
                         collective=(host, _op, _comm) -> host)
    @test last(events) == :sync
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)

    # Staged mathematical calls preserve `!` output identity and recursively
    # restore tuple results to the same vendor without leaking host arrays.
    staged_output = MockMPIArray(Float32[0, 0, 0])
    staged_input = MockMPIArray(Float32[2, 3, 4])
    staged_result = extension._staged_gpu_call(
        adapter, :mock_bang, comm,
        (host_output, host_input) -> begin
            host_output .= host_input .+ 1
            (host_output, copy(host_input))
        end,
        staged_output, staged_input; mutated=(1,), validate_storage=false,
    )
    @test first(staged_result) === staged_output
    @test staged_output.data == Float32[3, 4, 5]
    @test last(staged_result) isa MockMPIArray
    @test last(staged_result).data == staged_input.data

    # Pencil metadata and subgroup communicators survive staging intact.
    pencil = PencilArrays.Pencil(
        MockMPIArray, (3, 2), (1,), MPI.COMM_SELF,
    )
    device_pencil = PencilArrays.PencilArray{Float32}(undef, pencil)
    parent(device_pencil).data .= reshape(Float32.(1:6), 3, 2)
    pencil_result = extension._staged_gpu_call(
        adapter, :mock_pencil, MPI.COMM_SELF,
        host -> begin
            @test PencilArrays.size_global(host) == (3, 2)
            parent(host) .*= 3
            host
        end,
        device_pencil; mutated=(1,),
    )
    @test pencil_result === device_pencil
    @test parent(device_pencil).data == 3 .* reshape(Float32.(1:6), 3, 2)

    # Execute real staged mathematical callbacks on COMM_SELF while this test
    # normally runs under a two-rank WORLD. Before subgroup propagation was
    # fixed, the generic validators entered WORLD and mismatched/hung here.
    subgroup_cfg = SHTnsKit.create_gauss_config(2, 4; nlon=6)
    subgroup_dense = zeros(ComplexF64, 3, 3)
    subgroup_dense[1, 1] = 0.2
    subgroup_dense[2, 1] = -0.05
    subgroup_dense[3, 2] = 0.03 - 0.02im
    subgroup_pen = PencilArrays.Pencil(
        MockMPIArray, size(subgroup_dense), (2,), MPI.COMM_SELF,
    )
    subgroup_spectral = PencilArrays.PencilArray{ComplexF64}(
        undef, subgroup_pen,
    )
    parent(subgroup_spectral).data .= subgroup_dense
    subgroup_point = extension._staged_gpu_call(
        adapter, :mock_subgroup_point, MPI.COMM_SELF,
        host -> SHTnsKit.synthesis_point(
            subgroup_cfg, host, 0.31, -0.27,
        ), subgroup_spectral,
    )
    @test subgroup_point ≈ SHTnsKit.synthesis_point(
        subgroup_cfg, subgroup_dense, 0.31, -0.27,
    ) atol=3e-12
    subgroup_diagonal = extension._staged_gpu_call(
        adapter, :mock_subgroup_diagonal, MPI.COMM_SELF,
        host -> SHTnsKit.divergence_from_spheroidal(
            subgroup_cfg, host,
        ), subgroup_spectral,
    )
    @test subgroup_diagonal isa PencilArrays.PencilArray
    @test parent(subgroup_diagonal).data ≈
          SHTnsKit.divergence_from_spheroidal(
              subgroup_cfg, subgroup_dense,
          ) atol=3e-12
    MPI.Barrier(MPI.COMM_SELF)

    concurrent_adapter = extension.ParallelGPUAdapter(
        :mock_concurrent,
        value -> value isa MockMPIArray,
        _ -> MockMPIArray,
        _ -> 9,
        _ -> false,
        _ -> nothing,
        (T, n) -> Vector{T}(undef, n),
        copyto!, copyto!,
    )
    concurrent = MockMPIArray(Float32[0])
    tasks = [Threads.@spawn extension.allreduce!(
        concurrent, +, comm; adapter=concurrent_adapter,
        collective=(host, _op, _comm) -> (host .+= 1; host),
    ) for _ in 1:8]
    fetch.(tasks)
    @test concurrent.data == Float32[8]
    @test extension.parallel_gpu_cache_sizes().staging <= 2

    direct_adapter = extension.ParallelGPUAdapter(
        :mock_direct,
        value -> value isa MockMPIArray,
        _ -> Vector,
        _ -> 8,
        _ -> true,
        _ -> (sync_calls[] += 1),
        (T, n) -> Vector{T}(undef, n),
        copyto!,
        copyto!,
    )
    direct = function(device, _op, _comm)
        direct_collectives[] += 1
        device.data .+= 1
        return device
    end
    extension.allreduce!(buffer, +, Ref(:subgroup_b);
                         adapter=direct_adapter, collective=direct)
    @test buffer.data == Float32[5, 9, 13]
    @test direct_collectives[] == 1

    # The bounded cache may flush old entries, but must never grow past its cap.
    for n in 1:8
        value = MockMPIArray(fill(Float32(n), n))
        extension.allreduce!(value, +, Ref(Symbol(:comm_, n)); adapter,
                             collective=(host, _op, _comm) -> host)
        @test extension.parallel_gpu_cache_sizes().staging <= 2
    end

    stats = extension.parallel_gpu_stats()
    @test stats.direct_calls >= 1
    @test stats.staged_calls >= 10
    @test stats.direct_bytes >= sizeof(Float32) * 3
    @test stats.staged_bytes >= sizeof(Float32) * (3 + 3)
    extension.parallel_gpu_clear_caches!()
    @test extension.parallel_gpu_cache_sizes() == (awareness=0, staging=0)
end

const MPI_GPU_FIREWALL_GROUPS = (
    :analysis, :synthesis, :synthesis_cplx,
    :analysis_sphtor, :analysis_sphtor_cplx, :synthesis_sphtor,
    :synthesis_sphtor_cplx, :synthesis_sph, :synthesis_sph_cplx,
    :synthesis_tor, :synthesis_tor_cplx, :analysis_sphtor_l,
    :analysis_sphtor_ml, :synthesis_sphtor_l, :synthesis_sphtor_l_cplx,
    :synthesis_sphtor_ml, :synthesis_sph_l, :synthesis_sph_l_cplx,
    :synthesis_sph_ml, :synthesis_tor_l, :synthesis_tor_l_cplx,
    :synthesis_tor_ml, :analysis_qst,
    :analysis_qst_cplx, :synthesis_qst, :synthesis_qst_cplx,
    :analysis_qst_l, :analysis_qst_ml, :synthesis_qst_l,
    :synthesis_qst_l_cplx, :synthesis_qst_ml, :analysis_batch,
    :analysis_batch!, :synthesis_batch, :synthesis_batch!,
    :synthesis_batch_cplx, :analysis_sphtor_batch,
    :synthesis_sphtor_batch, :synthesis_sphtor_batch_cplx,
    :analysis_qst_batch, :synthesis_qst_batch, :synthesis_qst_batch_cplx,
    :analysis_packed, :analysis_packed_l, :analysis_packed_cplx,
    :analysis_packed_cplx_l, :analysis_packed_ml, :analysis_axisym,
    :analysis_axisym_l, :synthesis_packed, :synthesis_packed_l,
    :synthesis_packed_cplx, :synthesis_packed_cplx_l,
    :synthesis_packed_ml, :synthesis_axisym, :synthesis_axisym_l,
    :synthesis_point, :synthesis_point_cplx, :SH_to_lat,
    :SH_to_lat_cplx, :SHqst_to_point, :SHqst_to_lat,
    :SH_to_grad_point, :synthesis_grad, :synthesis_grad_l,
    :synthesis_grad_ml, :dist_analysis, :dist_synthesis,
    :dist_SH_to_lat, :dist_SH_to_point,
    :dist_SHqst_to_point, :dist_SHqst_to_lat,
    :dist_analysis_packed, :dist_synthesis_packed,
    :dist_analysis_packed_cplx, :dist_synthesis_packed_cplx,
    :dist_analysis_sphtor, :dist_synthesis_sphtor,
    :dist_analysis_qst, :dist_synthesis_qst,
    :dist_scalar_roundtrip!, :dist_vector_roundtrip!,
    :divergence_from_spheroidal, :divergence_from_spheroidal!,
    :spheroidal_from_divergence, :spheroidal_from_divergence!,
    :vorticity_from_toroidal, :vorticity_from_toroidal!,
    :toroidal_from_vorticity, :toroidal_from_vorticity!,
    :dist_apply_laplacian!, :SH_mul_mx, :dist_SH_mul_mx!,
    :dist_spatial_divergence,
    :dist_spatial_vorticity, :dist_scalar_laplacian,
    :dist_scalar_laplacian!, :dist_SH_Zrotate, :dist_SH_Yrotate,
    :dist_SH_Yrotate_allgatherm!, :dist_SH_Yrotate_truncgatherm!,
    :dist_SH_Yrotate90, :dist_SH_Xrotate90, :dist_SH_rotate_euler,
    :dist_SH_Zrotate_packed, :dist_SH_Yrotate_packed,
    :dist_SH_Yrotate90_packed, :dist_SH_Xrotate90_packed, :energy_scalar,
    :energy_scalar_l_spectrum, :energy_scalar_m_spectrum,
    :energy_vector_l_spectrum, :energy_vector_m_spectrum,
    :enstrophy_l_spectrum, :enstrophy_m_spectrum,
    :grid_energy_scalar, :grid_energy_vector, :grid_enstrophy,
    :dist_analysis!, :dist_synthesis!, :dist_analysis_sphtor!,
    :dist_synthesis_sphtor!, :dist_analysis_qst!, :dist_synthesis_qst!,
)

function test_mpi_gpu_source_contract(root::AbstractString, vendor::Symbol,
                                      compound_extension)
    parallel_extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    parallel_gpu = read(joinpath(root, "ext", "ParallelGPU.jl"), String)
    @test occursin("WeakRef", parallel_gpu)
    @test occursin("ReentrantLock", parallel_gpu)
    @test occursin("function exchange!", parallel_gpu)
    @test occursin("function allreduce!", parallel_gpu)
    for wrapper in (
        "LogicalIndex", "PermutedDimsArray", "Adjoint", "Transpose",
        "Symmetric", "Hermitian", "Diagonal", "Bidiagonal", "Tridiagonal",
        "UpperTriangular", "LowerTriangular", "UnitUpperTriangular",
        "UnitLowerTriangular",
    )
        @test occursin(wrapper, parallel_gpu)
    end
    @test occursin("logical_id", parallel_gpu)
    @test occursin("adapter.with_device(device)", parallel_gpu)
    @test occursin("_with_owner_device", parallel_gpu)
    @test occursin("_device_to_host_snapshot!", parallel_gpu)
    @test occursin("_host_to_device_snapshot!", parallel_gpu)
    @test occursin("local_device_mismatch", parallel_gpu)
    @test occursin("_validate_parallel_storage!(comm, :exchange", parallel_gpu)
    @test !occursin("using CUDA", parallel_gpu)
    @test !occursin("using AMDGPU", parallel_gpu)

    extension_file = vendor === :cuda ? "SHTnsKitParallelCUDAExt.jl" :
                     "SHTnsKitParallelAMDGPUExt.jl"
    source = read(joinpath(root, "ext", extension_file), String)
    @test occursin("_register_parallel_gpu_adapter!", source)
    @test occursin(vendor === :cuda ? "MPI.has_cuda" : "MPI.has_rocm", source)
    @test occursin("_parallel_root_buffer(value)", source)
    @test occursin(vendor === :cuda ? "CUDA.device(" : "AMDGPU.device(", source)
    @test occursin(vendor === :cuda ? "CUDA.device!(f, device)" :
                                    "AMDGPU.device!(f, device)", source)

    runner_file = vendor === :cuda ?
        joinpath(root, "test", "gpu", "cuda", "mpi_runtests.jl") :
        joinpath(root, "test", "gpu", "amdgpu", "mpi_runtests.jl")
    runner = read(runner_file, String)
    @test isdefined(@__MODULE__, :run_mpi_gpu_full_parity)
    @test occursin("run_mpi_gpu_full_parity(", runner)
    for family in (
        "native scalar/vector/QST transpose nonzero numerics",
        "scalar/vector/QST cfg parity",
        "scalar packed complex axisym _l _ml compatibility",
        "batch sizes 1/2/5 and bang identity",
        "fixed/local/operator/rotation staged parity",
        "vector QST _l _ml local gradient all operators",
        "general rotations diagnostics storage and compatibility",
        "actual allocation device cache key",
        "hardware multi-device context and rejection",
        "repeated-plan cache and residency",
    )
        @test occursin(family, read(@__FILE__, String))
    end
    matrix_source = read(@__FILE__, String)
    for call_marker in (
        "run_scalar_full_parity(", "run_sphtor_full_parity(",
        "run_qst_full_parity(", "analysis_packed_cplx_l(",
        "analysis_axisym_l(", "analysis_packed_ml(",
        "analysis_sphtor_batch(", "analysis_qst_batch(",
        "analysis_sphtor_l(", "analysis_qst_l(", "analysis_qst_ml(",
        "synthesis_sph_l_cplx(", "synthesis_tor_l_cplx(",
        "SHqst_to_point(", "SH_to_grad_point(",
        "divergence_from_spheroidal(", "spheroidal_from_divergence(",
        "vorticity_from_toroidal(", "toroidal_from_vorticity(",
        "dist_spatial_divergence(", "dist_spatial_vorticity(",
        "dist_scalar_laplacian!(", "dist_SH_rotate_euler(",
        "dist_SH_Xrotate90(", "dist_SH_Yrotate90(",
        "energy_scalar_l_spectrum(", "energy_vector_m_spectrum(",
        "enstrophy_l_spectrum(", "grid_energy_vector(",
        "dist_analysis_qst(", "dist_analysis_qst!(",
        "parallel_gpu_stats() == before",
    )
        @test occursin(call_marker, matrix_source)
    end

    project = read(joinpath(root, "Project.toml"), String)
    trigger = vendor === :cuda ? "SHTnsKitParallelCUDAExt" :
              "SHTnsKitParallelAMDGPUExt"
    @test occursin(trigger, project)

    firewall_path = joinpath(root, "ext", "ParallelGPUVendorFirewall.jl")
    @test isfile(firewall_path)
    isfile(firewall_path) || return
    firewall = read(firewall_path, String)
    @test compound_extension !== nothing
    @test isempty(Test.detect_ambiguities(
        SHTnsKit, compound_extension; recursive=true,
    ))
    for api in MPI_GPU_FIREWALL_GROUPS
        function_object = getfield(SHTnsKit, api)
        owned = filter(method -> method.module === compound_extension,
                       methods(function_object))
        @test !isempty(owned)
        @test all(method -> method.module !== parallel_extension, owned)
    end
    @test occursin("include(\"ParallelGPUVendorFirewall.jl\")", source)
    @test !occursin("Array(", firewall)
    @test !occursin("collect(", firewall)
    @test !occursin("allowscalar", firewall)
    @test occursin("_dist_transpose_gpu_analysis!", firewall)
    @test occursin("_staged_gpu_call", firewall)
    @test occursin(":synthesis_sph_l_cplx", firewall)
    @test occursin(":synthesis_tor_l_cplx", firewall)

    transpose_source = read(
        joinpath(root, "ext", "ParallelTransposeTransforms.jl"), String,
    )
    @test occursin("plan.F_buf, plan.F_buf2, all_values...", transpose_source)

    # Dealiased decompositions may leave a rank owning only Fourier bins above
    # mmax. The native kernel offset must still be the rank's real first bin,
    # never a fallback to m=0 when `plan.m_local` is empty.
    cfg = SHTnsKit.create_gauss_config(3, 5; nlon=13)
    plan = SHTnsKit.DistTransposePlan(
        cfg; comm=MPI.COMM_WORLD, nlev=1, with_vector=true,
    )
    expected_first_m = first(PencilArrays.range_local(
        PencilArrays.pencil(plan.F_buf),
    )[1]) - 1
    @test compound_extension._first_m(plan) == expected_first_m
end
