using Test
using SHTnsKit

include("scalar_full.jl")
include("scalar_variants.jl")
include("sphtor_full.jl")
include("qst_full.jl")
include("vector_variants.jl")
include("local_evaluation.jl")
include("operators.jl")
include("rotations.jl")

run_scalar_full_parity(CPUScalarAdapter())
@testset "scalar mres adjoint parity" begin
    test_mres_scalar_adjoints()
end

run_cpu_qst_full_parity()
test_cpu_vector_variant_reds()
run_local_evaluation_parity(CPULocalEvaluationAdapter())
test_cpu_local_compatibility_and_validation()
test_cpu_local_mixed_coordinate_precision()
test_cpu_operator_parity()
run_rotation_parity(CPURotationAdapter())
test_cpu_rotation_conventions_and_validation()

@testset "typed explicit complex synthesis" begin
    cfg = _scalar_config(:gauss, 3, 8)
    coefficients = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
    coefficients[3, 3] = 0.2f0 - 0.1f0im
    @test synthesis_cplx(CPU(), cfg, coefficients) ≈ synthesis_cplx(cfg, coefficients)
end
