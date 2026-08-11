using Test
using SHTnsKit

include("scalar_full.jl")

run_scalar_full_parity(CPUScalarAdapter())
@testset "scalar mres adjoint parity" begin
    test_mres_scalar_adjoints()
end
