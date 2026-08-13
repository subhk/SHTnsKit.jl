using Test

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

Base.IndexStyle(::Type{<:MockMultiDeviceArray}) = IndexLinear()
Base.size(array::MockMultiDeviceArray) = size(array.data)
Base.getindex(array::MockMultiDeviceArray, indices...) = array.data[indices...]
Base.setindex!(array::MockMultiDeviceArray, value, indices...) =
    (array.data[indices...] = value)
Base.copyto!(destination::MockMultiDeviceArray, source::AbstractArray) =
    (copyto!(destination.data, source); destination)
Base.copyto!(destination::AbstractArray, source::MockMultiDeviceArray) =
    copyto!(destination, source.data)

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
        norms=(:orthonormal,), real_norm_values=(false,),
        cs_phase_values=(true,), pole_orders=(false,),
    )
    run_scalar_full_parity(MPIGPUScalarAdapter(array_type, is_vendor, comm);
                           shared_axes...)
    run_sphtor_full_parity(MPIGPUVectorAdapter(array_type, is_vendor, comm);
        shared_axes..., robert_values=(false,))
    run_qst_full_parity(MPIGPUQSTAdapter(array_type, is_vendor, comm);
        shared_axes..., robert_values=(false,))

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
        end
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

        @testset "batch sizes 1/2/5 and bang identity" begin
            for nfields in (1, 2, 5)
                fields = PencilArrays.PencilArray{RT}(
                    undef, PencilArrays.pencil(spatial), nfields,
                )
                local_scalar = Array(parent(spatial))
                host_fields = repeat(reshape(local_scalar, size(local_scalar)..., 1),
                                     1, 1, nfields)
                copyto!(parent(fields), host_fields)
                coefficients = SHTnsKit.analysis_batch(cfg, fields)
                _mpi_gpu_assert_resident(coefficients, is_vendor)
                coefficients_bang = similar(coefficients)
                @test SHTnsKit.analysis_batch!(
                    cfg, coefficients_bang, fields,
                ) === coefficients_bang
                reconstructed = SHTnsKit.synthesis_batch(
                    cfg, coefficients; prototype_θφ=fields,
                )
                reconstructed_bang = similar(reconstructed)
                @test SHTnsKit.synthesis_batch!(
                    cfg, reconstructed_bang, coefficients;
                    prototype_θφ=fields,
                ) === reconstructed_bang
                _mpi_gpu_assert_resident(reconstructed_bang, is_vendor)
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

        @testset "native scalar/vector/QST transpose" begin
            staged_before = extension.parallel_gpu_stats().staged_calls
            for nlev in (1, 2, 5)
                plan = SHTnsKit.DistTransposePlan(
                    cfg; comm, nlev, array_type, real_type=RT,
                    with_vector=true,
                )
                Vr = SHTnsKit.allocate_spatial(plan)
                Vt = SHTnsKit.allocate_spatial(plan)
                Vp = SHTnsKit.allocate_spatial(plan)
                fill!(parent(Vr), RT(0.25))
                fill!(parent(Vt), zero(RT))
                fill!(parent(Vp), zero(RT))
                Q = SHTnsKit.allocate_spectral(plan)
                S = SHTnsKit.allocate_spectral(plan)
                T = SHTnsKit.allocate_spectral(plan)
                @test SHTnsKit.dist_analysis!(plan, Q, Vr) === Q
                @test SHTnsKit.dist_synthesis!(plan, Vr, Q) === Vr
                @test SHTnsKit.dist_analysis_sphtor!(
                    plan, S, T, Vt, Vp,
                ) === (S, T)
                @test SHTnsKit.dist_synthesis_sphtor!(
                    plan, Vt, Vp, S, T,
                ) === (Vt, Vp)
                @test SHTnsKit.dist_analysis_qst!(
                    plan, Q, S, T, Vr, Vt, Vp,
                ) === (Q, S, T)
                @test SHTnsKit.dist_synthesis_qst!(
                    plan, Vr, Vt, Vp, Q, S, T,
                ) === (Vr, Vt, Vp)
                _mpi_gpu_assert_resident((Vr, Vt, Vp, Q, S, T), is_vendor)
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
    multi_adapter = extension.ParallelGPUAdapter(
        :mock_multidevice,
        value -> extension._parallel_root_buffer(value) isa MockMultiDeviceArray,
        _ -> MockMultiDeviceArray,
        value -> extension._parallel_root_buffer(value).device,
        _ -> false,
        _ -> nothing,
        (T, n) -> Vector{T}(undef, n),
        copyto!, copyto!,
    )
    multi_comm = Ref(:multidevice_subgroup)
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
        event_buffer, event_receive, Ref(:event_exchange);
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
    :synthesis_sphtor_ml, :synthesis_sph_l, :synthesis_sph_ml,
    :synthesis_tor_l, :synthesis_tor_ml, :analysis_qst,
    :analysis_qst_cplx, :synthesis_qst, :synthesis_qst_cplx,
    :analysis_qst_l, :analysis_qst_ml, :synthesis_qst_l,
    :synthesis_qst_l_cplx, :synthesis_qst_ml, :analysis_batch,
    :analysis_batch!, :synthesis_batch, :synthesis_batch!,
    :synthesis_batch_cplx, :analysis_sphtor_batch,
    :synthesis_sphtor_batch, :analysis_qst_batch, :synthesis_qst_batch,
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
    :dist_apply_laplacian!, :SH_mul_mx, :dist_spatial_divergence,
    :dist_spatial_vorticity, :dist_scalar_laplacian,
    :dist_scalar_laplacian!, :dist_SH_Zrotate, :dist_SH_Yrotate,
    :dist_SH_Xrotate90, :dist_SH_rotate_euler,
    :dist_SH_Zrotate_packed, :dist_SH_Yrotate_packed,
    :dist_SH_Xrotate90_packed, :energy_scalar,
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
    @test !occursin("using CUDA", parallel_gpu)
    @test !occursin("using AMDGPU", parallel_gpu)

    extension_file = vendor === :cuda ? "SHTnsKitParallelCUDAExt.jl" :
                     "SHTnsKitParallelAMDGPUExt.jl"
    source = read(joinpath(root, "ext", extension_file), String)
    @test occursin("_register_parallel_gpu_adapter!", source)
    @test occursin(vendor === :cuda ? "MPI.has_cuda" : "MPI.has_rocm", source)
    @test occursin("_parallel_root_buffer(value)", source)
    @test occursin(vendor === :cuda ? "CUDA.device(" : "AMDGPU.device(", source)

    runner_file = vendor === :cuda ?
        joinpath(root, "test", "gpu", "cuda", "mpi_runtests.jl") :
        joinpath(root, "test", "gpu", "amdgpu", "mpi_runtests.jl")
    runner = read(runner_file, String)
    @test isdefined(@__MODULE__, :run_mpi_gpu_full_parity)
    @test occursin("run_mpi_gpu_full_parity(", runner)
    for family in (
        "native scalar/vector/QST transpose",
        "scalar/vector/QST cfg parity",
        "batch sizes 1/2/5 and bang identity",
        "fixed/local/operator/rotation staged parity",
        "actual allocation device cache key",
        "repeated-plan cache and residency",
    )
        @test occursin(family, read(@__FILE__, String))
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
