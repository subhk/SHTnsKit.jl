using Test
using SHTnsKit

include(joinpath(@__DIR__, "..", "parity", "capabilities.jl"))
using .SHTns37TestCapabilities

@testset "SHTns 3.7 capability contract" begin
    contract_names = (
        :SHTns37Capability, :SHTNS37_BACKENDS,
        :SHTNS37_CAPABILITIES, :shtns37_capabilities,
    )
    for name in contract_names
        @test isdefined(SHTnsKit, name)
        @test Base.isexported(SHTnsKit, name)
    end

    if all(name -> isdefined(SHTnsKit, name), contract_names)
        @test SHTnsKit.SHTNS37_BACKENDS == SHTns37TestCapabilities.BACKENDS
        @test SHTnsKit.SHTNS37_CAPABILITIES == SHTns37TestCapabilities.CAPABILITIES

        rows = SHTnsKit.shtns37_capabilities()
        expected_cells = Set(Iterators.product(
            SHTns37TestCapabilities.CAPABILITIES,
            SHTns37TestCapabilities.BACKENDS,
        ))
        actual_cells = Set((row.feature, row.backend) for row in rows)

        @test length(rows) == length(expected_cells)
        @test length(actual_cells) == length(rows)
        @test actual_cells == expected_cells

        for row in rows
            @test row isa SHTnsKit.SHTns37Capability
            @test row.testfile == SHTns37TestCapabilities.TESTFILES[row.backend]
            expected_status = row.backend in (:cpu, :mpi_cpu) ?
                :required : :unverified_hardware
            @test row.status == expected_status
        end

        empty!(rows)
        @test length(SHTnsKit.shtns37_capabilities()) == length(expected_cells)
    end

    for name in SHTns37TestCapabilities.ENTRYPOINTS
        @test isdefined(SHTnsKit, name)
        @test Base.isexported(SHTnsKit, name)
    end
end
