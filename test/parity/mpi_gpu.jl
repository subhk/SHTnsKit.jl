using Test

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
    @test sync_calls[] == 4
    @test staged_collectives[] == 2
    @test extension.parallel_gpu_cache_sizes().staging == 1

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
