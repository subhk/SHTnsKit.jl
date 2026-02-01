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
        # Undefined exports: some symbols are defined in extensions (GPU, parallel, etc.)
        undefined_exports=false,
        # Stale deps: BenchmarkTools for benchmarking, SparseArrays used conditionally
        stale_deps=(ignore=[:BenchmarkTools, :SparseArrays],),
        # Compat: ignore stdlib packages; check_extras=false to skip Random/Test
        deps_compat=(ignore=[:LinearAlgebra, :SparseArrays], check_extras=false),
        piracies=false  # Disable piracy tests for packages with many extensions
    )
end
