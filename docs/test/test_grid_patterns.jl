using Test

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "docs", "scripts", "generate_grid_patterns.jl"))
using .GridPatternPlots

@testset "grid-pattern documentation figure" begin
    specs = grid_specs()
    @test getproperty.(specs, :key) ==
          [:gauss, :regular, :regular_poles, :driscoll_healy]
    @test getproperty.(getproperty.(specs, :config), :grid_type) ==
          [:gauss, :regular, :regular_poles, :driscoll_healy]

    by_key = Dict(spec.key => spec for spec in specs)
    @test first(by_key[:gauss].config.θ) > 0
    @test last(by_key[:gauss].config.θ) < π
    @test first(by_key[:regular].config.θ) > 0
    @test last(by_key[:regular].config.θ) < π
    @test first(by_key[:regular_poles].config.θ) == 0
    @test last(by_key[:regular_poles].config.θ) ≈ π
    @test first(by_key[:driscoll_healy].config.θ) == 0
    @test last(by_key[:driscoll_healy].config.θ) < π

    for spec in specs
        points = projected_points(spec.config)
        @test length(points.x) == spec.config.nlat * spec.config.nlon
        @test all(points.x .^ 2 .+ points.y .^ 2 .<= 1 + 100eps())
        @test count(points.front) > 0
        @test count(.!points.front) > 0
    end

    mktempdir() do dir
        output = joinpath(dir, "grid-patterns.svg")
        @test generate_grid_patterns(output) == output
        @test isfile(output)
        svg = read(output, String)
        @test occursin("<svg", svg)
        @test occursin("<title id=\"grid-patterns-title\">", svg)
        @test occursin("<desc id=\"grid-patterns-description\">", svg)
        @test filesize(output) > 10_000
    end
end

@testset "grid page wiring" begin
    page = read(joinpath(ROOT, "docs", "src", "grids.md"), String)
    makefile = read(joinpath(ROOT, "docs", "make.jl"), String)
    @test occursin("assets/grid-patterns.svg", page)
    @test occursin("Grid Types\" => \"grids.md", makefile)
end
