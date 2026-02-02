# SHTnsKit.jl - QST (3D Vector) Transform Tests
# Tests for radial-spheroidal-toroidal vector field transforms

using Test
using Random
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "QST (3D Vector) Transforms" begin
    @testset "QST roundtrip" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(60)

        # Random QST coefficients
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

        # Synthesis
        Vr, Vt, Vp = SHqst_to_spat(cfg, Qlm, Slm, Tlm; real_output=true)

        # Analysis
        Qlm_rec, Slm_rec, Tlm_rec = spat_to_SHqst(cfg, Vr, Vt, Vp)

        @test isapprox(Qlm_rec, Qlm; rtol=1e-9, atol=1e-11)
        @test isapprox(Slm_rec, Slm; rtol=1e-9, atol=1e-11)
        @test isapprox(Tlm_rec, Tlm; rtol=1e-9, atol=1e-11)
    end

    @testset "QST truncated transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        ltr = lmax - 2
        rng = MersenneTwister(61)

        Qlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Qlm[:, 1] .= real.(Qlm[:, 1])
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Slm[1, :] .= 0; Tlm[1, :] .= 0

        # Truncated synthesis
        Vr_l, Vt_l, Vp_l = SHqst_to_spat_l(cfg, Qlm, Slm, Tlm, ltr; real_output=true)

        # Reference
        Qlm_z = copy(Qlm); Slm_z = copy(Slm); Tlm_z = copy(Tlm)
        for m in 0:lmax, l in (ltr+1):lmax
            Qlm_z[l+1, m+1] = 0
            Slm_z[l+1, m+1] = 0
            Tlm_z[l+1, m+1] = 0
        end
        Vr_ref, Vt_ref, Vp_ref = SHqst_to_spat(cfg, Qlm_z, Slm_z, Tlm_z; real_output=true)

        @test isapprox(Vr_l, Vr_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vt_l, Vt_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_l, Vp_ref; rtol=1e-10, atol=1e-12)
    end

    @testset "QST complex transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(62)

        # Complex spatial fields
        Vr = randn(rng, ComplexF64, nlat, nlon)
        Vt = randn(rng, ComplexF64, nlat, nlon)
        Vp = randn(rng, ComplexF64, nlat, nlon)

        # Analysis
        Qlm, Slm, Tlm = spat_cplx_to_SHqst(cfg, Vr, Vt, Vp)

        # Synthesis
        Vr_back, Vt_back, Vp_back = SHqst_to_spat_cplx(cfg, Qlm, Slm, Tlm)

        @test isapprox(Vr_back, Vr; rtol=1e-9, atol=1e-11)
        @test isapprox(Vt_back, Vt; rtol=1e-9, atol=1e-11)
        @test isapprox(Vp_back, Vp; rtol=1e-9, atol=1e-11)
    end

    @testset "QST mode-limited transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(63)

        im = 2
        ltr = lmax - 1
        len = ltr - im + 1

        Ql = randn(rng, ComplexF64, len)
        Sl = randn(rng, ComplexF64, len)
        Tl = randn(rng, ComplexF64, len)

        # Mode-limited synthesis
        Vr_ml, Vt_ml, Vp_ml = SHqst_to_spat_ml(cfg, im, Ql, Sl, Tl, ltr)

        @test length(Vr_ml) == nlat
        @test length(Vt_ml) == nlat
        @test length(Vp_ml) == nlat

        # Roundtrip
        Ql_back, Sl_back, Tl_back = spat_to_SHqst_ml(cfg, im, Vr_ml, Vt_ml, Vp_ml, ltr)

        @test isapprox(Ql_back, Ql; rtol=1e-9, atol=1e-11)
        @test isapprox(Sl_back, Sl; rtol=1e-9, atol=1e-11)
        @test isapprox(Tl_back, Tl; rtol=1e-9, atol=1e-11)
    end

    @testset "QST decomposition orthogonality" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(64)

        # Pure radial field (only Q)
        Qlm_only = randn(rng, ComplexF64, lmax+1, lmax+1)
        Qlm_only[:, 1] .= real.(Qlm_only[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            Qlm_only[l+1, m+1] = 0
        end
        Slm_zero = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm_zero = zeros(ComplexF64, lmax+1, lmax+1)

        Vr, Vt, Vp = SHqst_to_spat(cfg, Qlm_only, Slm_zero, Tlm_zero; real_output=true)

        # Vt and Vp should be zero for pure radial field
        @test maximum(abs.(Vt)) < 1e-12
        @test maximum(abs.(Vp)) < 1e-12

        # Pure toroidal field (only T)
        Qlm_zero = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm_only = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm_only[:, 1] .= real.(Tlm_only[:, 1])
        Tlm_only[1, :] .= 0

        Vr2, Vt2, Vp2 = SHqst_to_spat(cfg, Qlm_zero, Slm_zero, Tlm_only; real_output=true)

        # Vr should be zero for pure tangential field
        @test maximum(abs.(Vr2)) < 1e-12
    end
end
