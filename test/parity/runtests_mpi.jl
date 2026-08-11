using Test
using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit

MPI.Init()

include("scalar_full.jl")

struct MPIScalarAdapter <: ScalarParityAdapter
    comm
end

function place(adapter::MPIScalarAdapter, cfg::SHTConfig, value::AbstractMatrix, kind::Symbol)
    if kind === :spectral
        return matrix_to_spectral_pencil(cfg, value; comm=adapter.comm)
    end
    pen = create_spatial_pencil(cfg; comm=adapter.comm)
    placed = PencilArray{eltype(value)}(undef, pen)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        parent(placed)[i, j] = value[iglobal, jglobal]
    end
    return placed
end

collect_result(::MPIScalarAdapter, value::PencilArray, cfg::SHTConfig) =
    size_global(value) == (cfg.lmax + 1, cfg.mmax + 1) ?
        spectral_pencil_to_matrix(cfg, value) : _collect_spatial(value, cfg)

function _collect_spatial(value::PencilArray, cfg::SHTConfig)
    result = zeros(eltype(value), cfg.nlat, cfg.nlon)
    pen = pencil(value)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        result[iglobal, jglobal] = parent(value)[i, j]
    end
    MPI.Allreduce!(result, +, PencilArrays.get_comm(value))
    return result
end

analysis_call(::MPIScalarAdapter, cfg, field; use_rfft=false) =
    analysis(cfg, field; use_rfft)
synthesis_call(::MPIScalarAdapter, cfg, coefficients, prototype;
               real_output, use_rfft=false) =
    synthesis(cfg, coefficients; prototype_θφ=prototype, real_output, use_rfft)
synthesis_cplx_call(::MPIScalarAdapter, cfg, coefficients, prototype) =
    synthesis_cplx(cfg, coefficients; prototype_θφ=prototype)
assert_resident(::MPIScalarAdapter, value) = @test value isa PencilArray

function test_phi_split_complex_float32(adapter::MPIScalarAdapter)
    cfg = _scalar_config(:gauss, 3, 8)
    coefficients = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
    coefficients[1, 1] = ComplexF32(0.25, -0.15)
    coefficients[3, 3] = ComplexF32(-0.2, 0.1)
    field = _direct_scalar_sum(cfg, coefficients; real_output=false)

    pen = Pencil((cfg.nlat, cfg.nlon), adapter.comm)
    distributed = PencilArray{ComplexF32}(undef, pen)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        parent(distributed)[i, j] = field[iglobal, jglobal]
    end

    analyzed = analysis(cfg, distributed; return_pencil=true)
    @test analyzed isa PencilArray
    analyzed_host = spectral_pencil_to_matrix(cfg, analyzed)
    @test eltype(analyzed_host) === ComplexF32
    @test analyzed_host ≈ analysis(CPU(), cfg, field) atol=2f-4 rtol=2f-4

    reconstructed = synthesis(
        cfg, analyzed; prototype_θφ=distributed, real_output=false,
    )
    @test reconstructed isa PencilArray
    reconstructed_host = _collect_spatial(reconstructed, cfg)
    @test eltype(reconstructed_host) === ComplexF32
    @test reconstructed_host ≈ field atol=2f-4 rtol=2f-4
    return nothing
end

function test_pencil_native_path(adapter::MPIScalarAdapter)
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    cfg = _scalar_config(:gauss, 5, 10)
    coefficients, field = _closed_form_low_order(cfg, Float64)
    spatial = place(adapter, cfg, field, :spatial)
    spectral = place(adapter, cfg, coefficients, :spectral)

    extension._reset_pencil_scalar_stats!()
    analyzed = analysis(cfg, spatial)
    after_analysis = extension._pencil_scalar_stats()
    @test after_analysis.full_matrix_helper_calls == 0
    max_spectral_local = MPI.Allreduce(length(parent(analyzed)), max, adapter.comm)
    @test after_analysis.analysis_max_message_elements <= max_spectral_local

    extension._reset_pencil_scalar_stats!()
    reconstructed = synthesis(cfg, spectral; prototype_θφ=spatial)
    after_synthesis = extension._pencil_scalar_stats()
    @test after_synthesis.full_matrix_helper_calls == 0
    max_theta_local = MPI.Allreduce(size(parent(spatial), 1), max, adapter.comm)
    @test after_synthesis.synthesis_max_message_elements <= max_theta_local * cfg.nlon
    @test _collect_spatial(reconstructed, cfg) ≈ field atol=3e-12 rtol=3e-12

    dense_compat = analysis(cfg, spatial; return_pencil=false)
    @test dense_compat isa Matrix
    @test dense_compat ≈ coefficients atol=3e-12 rtol=3e-12
    local_compat = dist_synthesis(cfg, dense_compat; prototype_θφ=spatial)
    compat = PencilArray{eltype(local_compat)}(undef, pencil(spatial))
    copyto!(parent(compat), local_compat)
    @test _collect_spatial(compat, cfg) ≈ field atol=3e-12 rtol=3e-12

    pen2 = try
        Pencil((cfg.nlat, cfg.nlon), (1, 2), adapter.comm)
    catch
        nothing
    end
    if pen2 !== nothing
        ranges = PencilArrays.range_local(pen2)
        both_split = MPI.Allreduce(
            length(ranges[1]) < cfg.nlat && length(ranges[2]) < cfg.nlon,
            &, adapter.comm,
        )
        if both_split
            spatial2 = PencilArray{Float64}(undef, pen2)
            for (j, jglobal) in pairs(ranges[2]), (i, iglobal) in pairs(ranges[1])
                parent(spatial2)[i, j] = field[iglobal, jglobal]
            end
            extension._reset_pencil_scalar_stats!()
            spectral2 = analysis(cfg, spatial2)
            reconstructed2 = synthesis(cfg, spectral2; prototype_θφ=spatial2)
            @test extension._pencil_scalar_stats().full_matrix_helper_calls == 0
            @test _collect_spatial(reconstructed2, cfg) ≈ field atol=3e-12 rtol=3e-12
        else
            @test true
        end
    else
        @test true
    end
    return nothing
end

function _all_ranks_catch(call, comm)
    message = try
        call()
        ""
    catch err
        sprint(showerror, err)
    end
    reference = MPI.bcast(message, 0, comm)
    caught_same = !isempty(message) && message == reference
    total = MPI.Allreduce(caught_same ? 1 : 0, +, comm)
    MPI.Barrier(comm)
    return total == MPI.Comm_size(comm)
end

function test_collective_validation(adapter::MPIScalarAdapter)
    cfg = _scalar_config(:gauss, 3, 8)
    _, field = _closed_form_low_order(cfg, Float64)
    spatial = place(adapter, cfg, field, :spatial)

    malformed_pen = Pencil((cfg.lmax + 2, cfg.mmax + 1), adapter.comm)
    malformed = PencilArray{ComplexF64}(undef, malformed_pen)
    fill!(parent(malformed), 0)
    @test _all_ranks_catch(adapter.comm) do
        synthesis(cfg, malformed; prototype_θφ=spatial)
    end

    rank_varying = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    rank_varying[1, 1] = MPI.Comm_rank(adapter.comm)
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis(cfg, rank_varying; prototype_θφ=spatial)
    end
    return nothing
end

adapter = MPIScalarAdapter(MPI.COMM_WORLD)
# Exercise every mathematical axis without taking their full Cartesian product:
# each Pencil construction owns an MPI Cartesian communicator, and a giant
# product needlessly exhausts MPI implementations with a 2048-context limit.
run_scalar_full_parity(
    adapter;
    grid_kinds=_SCALAR_GRID_KINDS,
    precisions=(Float32, Float64),
    mres_values=(1, 2),
    norms=(:orthonormal,),
    real_norm_values=(false,),
    cs_phase_values=(true,),
    pole_orders=(false,),
)
run_scalar_full_parity(
    adapter;
    grid_kinds=(:gauss,),
    precisions=(Float64,),
    mres_values=(1,),
    norms=(:orthonormal, :fourpi, :schmidt),
    real_norm_values=(false, true),
    cs_phase_values=(false, true),
    pole_orders=(false, true),
)
@testset "scalar full-grid parity φ-split complex Float32" begin
    test_phi_split_complex_float32(adapter)
end
@testset "Pencil-native scalar path" begin
    test_pencil_native_path(adapter)
end
@testset "collective scalar validation" begin
    test_collective_validation(adapter)
end

MPI.Barrier(MPI.COMM_WORLD)
