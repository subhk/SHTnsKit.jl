# SHTnsKit.jl - Parametric Vector Transform Tests
# Tests vector transforms across multiple latitude/longitude grid configurations

using Test
using Random
using LinearAlgebra
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Vector Transforms - Multiple Grid Resolutions" begin
    # Test vector transforms across different latitude and longitude configurations
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
            rng = MersenneTwister(300 + lmax)

            # Create valid vector spectral coefficients
            Slm = zeros(ComplexF64, lmax+1, lmax+1)
            Tlm = zeros(ComplexF64, lmax+1, lmax+1)

            for m in 0:lmax
                for l in max(1, m):lmax  # l >= 1 for vector fields
                    Slm[l+1, m+1] = randn(rng) + im * randn(rng)
                    Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
                end
                if m == 0
                    Slm[:, 1] .= real.(Slm[:, 1])
                    Tlm[:, 1] .= real.(Tlm[:, 1])
                end
            end

            @testset "Sphtor synthesis/analysis roundtrip" begin
                Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
                @test size(Vt) == (nlat, nlon)
                @test size(Vp) == (nlat, nlon)
                @test all(isfinite, Vt)
                @test all(isfinite, Vp)

                Slm_rec, Tlm_rec = analysis_sphtor(cfg, Vt, Vp)
                @test size(Slm_rec) == (lmax+1, lmax+1)
                @test isapprox(Slm_rec, Slm; rtol=1e-8, atol=1e-10)
                @test isapprox(Tlm_rec, Tlm; rtol=1e-8, atol=1e-10)
            end

            @testset "Individual S and T components" begin
                # Use real_output=true for consistency
                Vt_s, Vp_s = synthesis_sph(cfg, Slm; real_output=true)
                Vt_t, Vp_t = synthesis_tor(cfg, Tlm; real_output=true)
                @test size(Vt_s) == (nlat, nlon)
                @test size(Vt_t) == (nlat, nlon)
                @test all(isfinite, Vt_s) && all(isfinite, Vp_s)
                @test all(isfinite, Vt_t) && all(isfinite, Vp_t)

                # Combined should match full sphtor
                Vt_combined, Vp_combined = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
                @test isapprox(Vt_s .+ Vt_t, Vt_combined; rtol=1e-10, atol=1e-12)
                @test isapprox(Vp_s .+ Vp_t, Vp_combined; rtol=1e-10, atol=1e-12)
            end

            @testset "In-place transforms" begin
                plan = SHTPlan(cfg)
                Vt_ip = zeros(nlat, nlon)
                Vp_ip = zeros(nlat, nlon)
                synthesis_sphtor!(plan, Vt_ip, Vp_ip, Slm, Tlm)

                Vt_ref, Vp_ref = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
                @test isapprox(Vt_ip, Vt_ref; rtol=1e-12)
                @test isapprox(Vp_ip, Vp_ref; rtol=1e-12)

                Slm_ip = zeros(ComplexF64, lmax+1, lmax+1)
                Tlm_ip = zeros(ComplexF64, lmax+1, lmax+1)
                analysis_sphtor!(plan, Slm_ip, Tlm_ip, Vt_ip, Vp_ip)
                @test isapprox(Slm_ip, Slm; rtol=1e-8, atol=1e-10)
                @test isapprox(Tlm_ip, Tlm; rtol=1e-8, atol=1e-10)
            end

            @testset "Truncated transforms" begin
                ltr = max(2, lmax - 2)

                Vt_l, Vp_l = synthesis_sphtor_l(cfg, Slm, Tlm, ltr)
                @test size(Vt_l) == (nlat, nlon)
                @test size(Vp_l) == (nlat, nlon)
                @test all(isfinite, Vt_l) && all(isfinite, Vp_l)

                # Reference: zero high modes
                Slm_z = copy(Slm); Tlm_z = copy(Tlm)
                for m in 0:lmax, l in (ltr+1):lmax
                    Slm_z[l+1, m+1] = 0
                    Tlm_z[l+1, m+1] = 0
                end
                Vt_ref, Vp_ref = synthesis_sphtor(cfg, Slm_z, Tlm_z; real_output=true)
                @test isapprox(Vt_l, Vt_ref; rtol=1e-10, atol=1e-12)
                @test isapprox(Vp_l, Vp_ref; rtol=1e-10, atol=1e-12)
            end

            @testset "Divergence/vorticity operations" begin
                div_lm = divergence_from_spheroidal(cfg, Slm)
                vort_lm = vorticity_from_toroidal(cfg, Tlm)
                @test size(div_lm) == (lmax+1, lmax+1)
                @test size(vort_lm) == (lmax+1, lmax+1)

                Slm_from_div = spheroidal_from_divergence(cfg, div_lm)
                Tlm_from_vort = toroidal_from_vorticity(cfg, vort_lm)
                @test isapprox(Slm_from_div, Slm; rtol=1e-8, atol=1e-10)
                @test isapprox(Tlm_from_vort, Tlm; rtol=1e-8, atol=1e-10)
            end

            @testset "Complex transforms" begin
                Vt_cplx, Vp_cplx = synthesis_sphtor_cplx(cfg, Slm, Tlm)
                @test size(Vt_cplx) == (nlat, nlon)
                @test eltype(Vt_cplx) <: Complex

                Slm_back, Tlm_back = analysis_sphtor_cplx(cfg, Vt_cplx, Vp_cplx)
                @test isapprox(Slm_back, Slm; rtol=1e-8, atol=1e-10)
                @test isapprox(Tlm_back, Tlm; rtol=1e-8, atol=1e-10)
            end

            @testset "Gradient transform" begin
                # Random scalar field coefficients
                alm = randn(rng, ComplexF64, lmax+1, lmax+1)
                alm[:, 1] .= real.(alm[:, 1])
                for m in 0:lmax, l in 0:(m-1)
                    alm[l+1, m+1] = 0.0
                end

                Gt, Gp = synthesis_grad(cfg, alm)
                @test size(Gt) == (nlat, nlon)
                @test all(isfinite, Gt) && all(isfinite, Gp)

                # Gradient should match synthesis_sph
                Gt_ref, Gp_ref = synthesis_sph(cfg, alm)
                @test isapprox(Gt, Gt_ref; rtol=1e-10, atol=1e-12)
                @test isapprox(Gp, Gp_ref; rtol=1e-10, atol=1e-12)
            end

            VERBOSE && @info "Grid test passed" lmax nlat nlon
        end
    end
end
