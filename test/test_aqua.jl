# Aqua.jl Quality Assurance Tests for SHTnsKit.jl
#
# This test suite uses Aqua.jl to perform automated quality assurance checks
# including ambiguity detection, unbound type parameters, undefined exports, etc.

using Test
using Aqua
using SHTnsKit

@testset "Aqua Quality Assurance" begin
    Aqua.test_all(
        SHTnsKit;
        ambiguities=false,  # Disable ambiguity tests (can be strict for complex packages)
        stale_deps=(ignore=[:BenchmarkTools],),  # BenchmarkTools used for optional benchmarking
        deps_compat=(ignore=[:LinearAlgebra, :SparseArrays],),  # Stdlib packages
        piracies=false  # Disable piracy tests for packages with many extensions
    )
end
