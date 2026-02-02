# SHTnsKit.jl - Parametric QST Transform Tests
# Tests 3D vector (QST) transforms across multiple latitude/longitude grid configurations

using Test
using Random
using LinearAlgebra
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "QST Transforms - Multiple Grid Resolutions" begin
    # Test QST transforms across different latitude and longitude configurations
    grid_configs = [
        (lmax=4,  nlat=6,  nlon=9),    # Minimal resolution
        (lmax=6,  nlat=8,  nlon=13),   # Low resolution
        (lmax=8,  nlat=12, nlon=21),   # Higher nlat
        (lmax=8,  nlat=10, nlon=25),   # Higher nlon
        (lmax=12, nlat=16, nlon=25),   # Medium resolution
        (lmax=16, nlat=20, nlon=33),   # Higher resolution
    ]

    for (lmax, nlat, nlon) in grid_configs
        @testset "lmax=$lmax, nlat=$nlat, nlon=$nlon" begin
            cfg = create_gauss_config(lmax, nlat; nlon=nlon)
            rng = MersenneTwister(400 + lmax)

            # Create valid QST spectral coefficients
            Qlm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)

            # m=0 must be real
            Qlm[:, 1] .= real.(Qlm[:, 1])
            Slm[:, 1] .= real.(Slm[:, 1])
            Tlm[:, 1] .= real.(Tlm[:, 1])

            # l=0 for S,T should be zero
            Slm[1, :] .= 0
            Tlm[1, :] .= 0

            # Zero invalid (l < m)
            for m in 0:lmax, l in 0:(m-1)
                Qlm[l+1, m+1] = 0
                Slm[l+1, m+1] = 0
                Tlm[l+1, m+1] = 0
            end

            @testset "QST synthesis/analysis roundtrip" begin
                Vr, Vt, Vp = synthesis_qst(cfg, Qlm, Slm, Tlm; real_output=true)
                @test size(Vr) == (nlat, nlon)
                @test size(Vt) == (nlat, nlon)
                @test size(Vp) == (nlat, nlon)
                @test all(isfinite, Vr)
                @test all(isfinite, Vt)
                @test all(isfinite, Vp)

                Qlm_rec, Slm_rec, Tlm_rec = analysis_qst(cfg, Vr, Vt, Vp)
                @test isapprox(Qlm_rec, Qlm; rtol=1e-9, atol=1e-11)
                @test isapprox(Slm_rec, Slm; rtol=1e-9, atol=1e-11)
                @test isapprox(Tlm_rec, Tlm; rtol=1e-9, atol=1e-11)
            end

            @testset "Truncated QST transforms" begin
                ltr = max(2, lmax - 2)

                Vr_l, Vt_l, Vp_l = synthesis_qst_l(cfg, Qlm, Slm, Tlm, ltr; real_output=true)
                @test size(Vr_l) == (nlat, nlon)
                @test all(isfinite, Vr_l) && all(isfinite, Vt_l) && all(isfinite, Vp_l)

                # Reference: zero high modes
                Qlm_z = copy(Qlm); Slm_z = copy(Slm); Tlm_z = copy(Tlm)
                for m in 0:lmax, l in (ltr+1):lmax
                    Qlm_z[l+1, m+1] = 0
                    Slm_z[l+1, m+1] = 0
                    Tlm_z[l+1, m+1] = 0
                end
                Vr_ref, Vt_ref, Vp_ref = synthesis_qst(cfg, Qlm_z, Slm_z, Tlm_z; real_output=true)
                @test isapprox(Vr_l, Vr_ref; rtol=1e-10, atol=1e-12)
                @test isapprox(Vt_l, Vt_ref; rtol=1e-10, atol=1e-12)
                @test isapprox(Vp_l, Vp_ref; rtol=1e-10, atol=1e-12)
            end

            @testset "Complex QST transforms" begin
                Vr_cplx, Vt_cplx, Vp_cplx = synthesis_qst_cplx(cfg, Qlm, Slm, Tlm)
                @test size(Vr_cplx) == (nlat, nlon)
                @test eltype(Vr_cplx) <: Complex

                Qlm_back, Slm_back, Tlm_back = analysis_qst_cplx(cfg, Vr_cplx, Vt_cplx, Vp_cplx)

                # Valid entries should match
                for m in 0:lmax
                    for l in m:lmax
                        @test isapprox(Qlm_back[l+1, m+1], Qlm[l+1, m+1]; rtol=1e-8, atol=1e-10)
                    end
                    for l in max(1, m):lmax
                        @test isapprox(Slm_back[l+1, m+1], Slm[l+1, m+1]; rtol=1e-8, atol=1e-10)
                        @test isapprox(Tlm_back[l+1, m+1], Tlm[l+1, m+1]; rtol=1e-8, atol=1e-10)
                    end
                end
            end

            @testset "QST decomposition orthogonality" begin
                # Pure radial field (only Q)
                Qlm_only = copy(Qlm)
                Slm_zero = zeros(ComplexF64, lmax+1, lmax+1)
                Tlm_zero = zeros(ComplexF64, lmax+1, lmax+1)

                Vr, Vt, Vp = synthesis_qst(cfg, Qlm_only, Slm_zero, Tlm_zero; real_output=true)

                # Vt and Vp should be zero for pure radial field
                @test maximum(abs.(Vt)) < 1e-12
                @test maximum(abs.(Vp)) < 1e-12

                # Pure toroidal field (only T)
                Qlm_zero = zeros(ComplexF64, lmax+1, lmax+1)
                Tlm_only = copy(Tlm)

                Vr2, Vt2, Vp2 = synthesis_qst(cfg, Qlm_zero, Slm_zero, Tlm_only; real_output=true)

                # Vr should be zero for pure tangential field
                @test maximum(abs.(Vr2)) < 1e-12
            end

            VERBOSE && @info "QST grid test passed" lmax nlat nlon
        end
    end
end
