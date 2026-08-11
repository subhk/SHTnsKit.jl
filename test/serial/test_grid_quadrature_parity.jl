# SHTns-compatible regular-grid quadrature tests.

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

_regular_exact_degree(n::Int) = isodd(n) ? n : n - 1
_exact_monomial_moment(k::Int) = isodd(k) ? 0.0 : 2.0 / (k + 1)

function _regular_grid_config(grid_type::Symbol, nlat::Int; lmax::Int=0,
                              mmax::Int=lmax, mres::Int=1, nlon::Int=max(2mmax + 1, 4))
    return create_regular_config(
        lmax, nlat;
        mmax, mres, nlon,
        include_poles=grid_type === :regular_poles,
        precompute_plm=false,
    )
end

function _test_quadrature_moments(cfg)
    degree = _regular_exact_degree(cfg.nlat)
    atol = 80eps(Float64)
    for k in 0:degree
        numerical = sum(cfg.w .* cfg.x .^ k)
        @test numerical ≈ _exact_monomial_moment(k) atol=atol rtol=atol
    end
end

@testset "SHTns-compatible regular-grid quadrature" begin
    @testset "Fejér first rule ($nlat latitudes)" for nlat in (2, 5, 6, 9)
        cfg = _regular_grid_config(:regular, nlat)
        expected_θ = [(j + 0.5) * π / nlat for j in 0:(nlat - 1)]

        @test cfg.grid_type === :regular
        @test cfg.θ ≈ expected_θ atol=8eps(Float64) rtol=0
        @test cfg.x ≈ cos.(cfg.θ) atol=8eps(Float64) rtol=0
        @test all(abs.(cfg.x) .< 1)
        @test cfg.x ≈ .-reverse(cfg.x) atol=8eps(Float64) rtol=0
        @test cfg.w ≈ reverse(cfg.w) atol=8eps(Float64) rtol=0
        @test sum(cfg.w) ≈ 2.0 atol=16eps(Float64) rtol=0
        _test_quadrature_moments(cfg)
    end

    @testset "Clenshaw–Curtis rule ($nlat latitudes)" for nlat in (2, 3, 6, 9)
        cfg = _regular_grid_config(:regular_poles, nlat)
        expected_θ = [j * π / (nlat - 1) for j in 0:(nlat - 1)]

        @test cfg.grid_type === :regular_poles
        @test cfg.θ ≈ expected_θ atol=8eps(Float64) rtol=0
        @test cfg.x ≈ cos.(cfg.θ) atol=8eps(Float64) rtol=0
        @test cfg.x[1] == 1.0
        @test cfg.x[end] == -1.0
        @test cfg.x ≈ .-reverse(cfg.x) atol=8eps(Float64) rtol=0
        @test cfg.w ≈ reverse(cfg.w) atol=8eps(Float64) rtol=0
        @test sum(cfg.w) ≈ 2.0 atol=16eps(Float64) rtol=0
        _test_quadrature_moments(cfg)
    end

    @testset "Latitude ordering keeps geometry and weights aligned" begin
        for grid_type in (:regular, :regular_poles)
            cfg = _regular_grid_config(grid_type, 9; lmax=6, mmax=6, mres=2, nlon=13)
            north = (copy(cfg.θ), copy(cfg.x), copy(cfg.w), copy(cfg.st))

            @test cfg.mres == 2
            @test !is_south_pole_first(cfg)
            set_south_pole_first!(cfg)
            @test is_south_pole_first(cfg)
            @test cfg.θ == reverse(north[1])
            @test cfg.x == reverse(north[2])
            @test cfg.w == reverse(north[3])
            @test cfg.st == reverse(north[4])
            @test cfg.x ≈ cos.(cfg.θ) atol=8eps(Float64) rtol=0
            @test cfg.st ≈ sin.(cfg.θ) atol=8eps(Float64) rtol=0
            _test_quadrature_moments(cfg)

            set_north_pole_first!(cfg)
            @test (cfg.θ, cfg.x, cfg.w, cfg.st) == north
        end
    end

    @testset "Float32/Float64 transforms accept both storage orientations" begin
        rng = MersenneTwister(0x5fe3)
        lmax = 4
        nlat = 11
        nlon = 12

        for grid_type in (:regular, :regular_poles), T in (Float32, Float64)
            cfg = _regular_grid_config(grid_type, nlat; lmax, nlon)
            CT = Complex{T}
            alm = zeros(CT, lmax + 1, lmax + 1)
            for m in 0:lmax, l in m:lmax
                alm[l + 1, m + 1] = CT(randn(rng, T), m == 0 ? zero(T) : randn(rng, T))
            end

            field = synthesis(cfg, alm; real_output=true)
            @test eltype(field) === T

            # The logical shape stays (nlat, nlon), but the parent stores the
            # longitude dimension first. This exercises generic AbstractMatrix
            # indexing without introducing a second layout API.
            longitude_first_parent = permutedims(field, (2, 1))
            latitude_longitude = PermutedDimsArray(longitude_first_parent, (2, 1))
            @test size(latitude_longitude) == (nlat, nlon)
            @test parent(latitude_longitude) === longitude_first_parent

            alm_contiguous = analysis(cfg, field)
            alm_permuted = analysis(cfg, latitude_longitude)
            tol = T === Float32 ? 8f-5 : 2e-12
            @test alm_permuted ≈ alm_contiguous atol=tol rtol=tol
            @test alm_permuted ≈ alm atol=tol rtol=tol
        end
    end
end
