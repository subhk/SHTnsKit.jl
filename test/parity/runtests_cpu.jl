using Test
using SHTnsKit

include("scalar_full.jl")
include("scalar_variants.jl")
include("sphtor_full.jl")
include("qst_full.jl")
include("vector_variants.jl")

run_scalar_full_parity(CPUScalarAdapter())
@testset "scalar mres adjoint parity" begin
    test_mres_scalar_adjoints()
end

run_cpu_qst_full_parity()
test_cpu_vector_variant_reds()

@testset "typed explicit complex synthesis" begin
    cfg = _scalar_config(:gauss, 3, 8)
    coefficients = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
    coefficients[3, 3] = 0.2f0 - 0.1f0im
    @test synthesis_cplx(CPU(), cfg, coefficients) ≈ synthesis_cplx(cfg, coefficients)
end
