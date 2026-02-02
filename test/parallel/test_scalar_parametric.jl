# SHTnsKit.jl - Parametric Scalar Transform Tests
# Tests scalar transforms across multiple latitude/longitude grid configurations

using Test
using Random
using LinearAlgebra
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Scalar Transforms - Multiple Grid Resolutions" begin
    # Test transforms across different latitude and longitude configurations
    grid_configs = [
        (lmax=4,  nlat=6,  nlon=9),    # Minimal resolution
        (lmax=8,  nlat=10, nlon=17),   # Low resolution
        (lmax=8,  nlat=12, nlon=21),   # Higher nlat
        (lmax=8,  nlat=10, nlon=25),   # Higher nlon
        (lmax=16, nlat=20, nlon=33),   # Medium resolution
        (lmax=16, nlat=24, nlon=41),   # Medium with extra lat/lon
        (lmax=32, nlat=36, nlon=65),   # Higher resolution
    ]

    for (lmax, nlat, nlon) in grid_configs
        @testset "lmax=$lmax, nlat=$nlat, nlon=$nlon" begin
            cfg = create_gauss_config(lmax, nlat; nlon=nlon)
            rng = MersenneTwister(200 + lmax)

            # Create valid spectral coefficients
            alm = randn(rng, ComplexF64, lmax+1, lmax+1)
            alm[:, 1] .= real.(alm[:, 1])  # m=0 must be real
            for m in 0:lmax, l in 0:(m-1)
                alm[l+1, m+1] = 0.0
            end

            @testset "Basic synthesis/analysis roundtrip" begin
                f = synthesis(cfg, alm; real_output=true)
                @test size(f) == (nlat, nlon)
                @test all(isfinite, f)

                alm_rec = analysis(cfg, f)
                @test size(alm_rec) == (lmax+1, lmax+1)
                @test isapprox(alm_rec, alm; rtol=1e-9, atol=1e-11)
            end

            @testset "Packed format" begin
                Qlm = randn(rng, ComplexF64, cfg.nlm)
                Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

                f_packed = synthesis_packed(cfg, Qlm)
                @test length(f_packed) == nlat * nlon

                Qlm_rec = analysis_packed(cfg, f_packed)
                @test isapprox(Qlm_rec, Qlm; rtol=1e-9, atol=1e-11)
            end

            @testset "SHTPlan transforms" begin
                plan = SHTPlan(cfg)
                f_plan = zeros(nlat, nlon)
                synthesis!(plan, f_plan, alm)
                @test isapprox(f_plan, synthesis(cfg, alm; real_output=true); rtol=1e-12)

                alm_plan = zeros(ComplexF64, lmax+1, lmax+1)
                analysis!(plan, alm_plan, f_plan)
                @test isapprox(alm_plan, alm; rtol=1e-9, atol=1e-11)
            end

            @testset "Truncated transforms" begin
                ltr = max(1, lmax - 2)

                # Create coefficients truncated to ltr
                alm_tr = copy(alm)
                for m in 0:lmax, l in (ltr+1):lmax
                    alm_tr[l+1, m+1] = 0
                end

                # Truncated packed transforms
                Qlm = randn(rng, ComplexF64, cfg.nlm)
                Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

                f_l = synthesis_packed_l(cfg, Qlm, ltr)
                @test length(f_l) == nlat * nlon

                Qlm_rec_l = analysis_packed_l(cfg, f_l, ltr)
                @test length(Qlm_rec_l) == cfg.nlm
            end

            @testset "Fused vs unfused loops" begin
                f_fused = synthesis(cfg, alm; real_output=true, use_fused_loops=true)
                f_unfused = synthesis(cfg, alm; real_output=true, use_fused_loops=false)
                @test isapprox(f_fused, f_unfused; rtol=1e-10, atol=1e-12)

                alm_fused = analysis(cfg, f_fused; use_fused_loops=true)
                alm_unfused = analysis(cfg, f_fused; use_fused_loops=false)
                @test isapprox(alm_fused, alm_unfused; rtol=1e-10, atol=1e-12)
            end

            @testset "PLM tables vs on-the-fly" begin
                cfg_plm = create_gauss_config(lmax, nlat; nlon=nlon)
                prepare_plm_tables!(cfg_plm)

                f_plm = synthesis(cfg_plm, alm; real_output=true)
                f_otf = synthesis(cfg, alm; real_output=true)
                @test isapprox(f_plm, f_otf; rtol=1e-10, atol=1e-12)

                alm_plm = analysis(cfg_plm, f_plm)
                alm_otf = analysis(cfg, f_otf)
                @test isapprox(alm_plm, alm_otf; rtol=1e-10, atol=1e-12)
            end

            @testset "Axisymmetric transforms" begin
                Ql = complex.(randn(rng, lmax+1))

                f_lat = synthesis_axisym(cfg, Ql)
                @test length(f_lat) == nlat
                @test all(isfinite, f_lat)

                Ql_rec = analysis_axisym(cfg, f_lat)
                @test length(Ql_rec) == lmax + 1

                # Shape preservation test
                Ql_norm = real.(Ql) / norm(real.(Ql))
                Ql_rec_norm = real.(Ql_rec) / norm(real.(Ql_rec))
                @test isapprox(Ql_rec_norm, Ql_norm; rtol=1e-9, atol=1e-11)
            end

            VERBOSE && @info "Grid test passed" lmax nlat nlon
        end
    end
end
