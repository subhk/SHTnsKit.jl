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

analysis_call(::MPIScalarAdapter, cfg, field) = analysis(cfg, field; return_pencil=true)
synthesis_call(::MPIScalarAdapter, cfg, coefficients, prototype; real_output) =
    synthesis(cfg, coefficients; prototype_θφ=prototype, real_output)
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

MPI.Barrier(MPI.COMM_WORLD)
