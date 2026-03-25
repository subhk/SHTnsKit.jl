# SHTnsKit.jl - Local Evaluation Tests
# Tests for SH_to_lat, SH_to_lat_cplx, SHqst_to_point, SHqst_to_lat

using Test
using SHTnsKit
using Random

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Local Evaluations" begin
    @testset "SH_to_lat matches synthesis at grid latitudes" begin
        lmax = 8
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        rng = MersenneTwister(42)
        # Create random real-field coefficients in dense (l,m) matrix format
        alm = zeros(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in m:lmax
            alm[l+1, m+1] = randn(rng, ComplexF64)
        end
        # m=0 must be real for real fields
        alm[:, 1] .= real.(alm[:, 1])
        # Zero invalid entries (l < m)
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Full synthesis for reference
        f = synthesis(cfg, alm; real_output=true)

        # Pack coefficients for SH_to_lat (needs packed LM format)
        Qlm = Vector{ComplexF64}(undef, cfg.nlm)
        for k in 1:cfg.nlm
            l = cfg.li[k]
            m = cfg.mi[k]
            Qlm[k] = alm[l+1, m+1]
        end

        # Test at each grid latitude: SH_to_lat should match synthesis row
        for i in 1:cfg.nlat
            cost = cfg.x[i]
            vals = SH_to_lat(cfg, Qlm, cost)
            @test length(vals) == cfg.nlon
            @test isapprox(vals, f[i, :]; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "SH_to_lat with degree truncation" begin
        lmax = 8
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)
        ltr = 4  # Truncate at degree 4

        rng = MersenneTwister(43)
        alm = zeros(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in m:lmax
            alm[l+1, m+1] = randn(rng, ComplexF64)
        end
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Truncated reference: zero modes above ltr
        alm_trunc = copy(alm)
        for m in 0:lmax, l in (ltr+1):lmax
            alm_trunc[l+1, m+1] = 0
        end
        f_trunc = synthesis(cfg, alm_trunc; real_output=true)

        # Pack
        Qlm = Vector{ComplexF64}(undef, cfg.nlm)
        for k in 1:cfg.nlm
            l = cfg.li[k]; m = cfg.mi[k]
            Qlm[k] = alm[l+1, m+1]
        end

        # SH_to_lat with ltr should match truncated synthesis
        for i in 1:min(3, cfg.nlat)  # Test a few latitudes
            cost = cfg.x[i]
            vals = SH_to_lat(cfg, Qlm, cost; ltr=ltr)
            @test isapprox(vals, f_trunc[i, :]; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "SH_to_lat_cplx matches complex synthesis" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        rng = MersenneTwister(44)
        # Create complex-packed coefficients (LM_cplx format, all m including negative)
        nlm_c = nlm_cplx_calc(lmax, lmax, 1)
        alm_cplx = randn(rng, ComplexF64, nlm_c)

        # Evaluate at a grid latitude
        cost = cfg.x[1]
        vals = SHTnsKit.SH_to_lat_cplx(cfg, alm_cplx, cost)
        @test length(vals) == cfg.nlon
        @test eltype(vals) == ComplexF64
        # Basic sanity: values should be finite
        @test all(isfinite, vals)
    end

    @testset "SHqst_to_point basic evaluation" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Set up trivial QST coefficients
        Qlm = zeros(ComplexF64, cfg.nlm)
        Slm = zeros(ComplexF64, cfg.nlm)
        Tlm = zeros(ComplexF64, cfg.nlm)

        # Set Q_{1,0} = 1 (a simple radial dipole)
        lm10 = LM_index(lmax, 1, 1, 0) + 1
        Qlm[lm10] = 1.0 + 0im

        cost = 0.5  # Some latitude
        phi = 0.3   # Some longitude
        Vr, Vt, Vp = SHTnsKit.SHqst_to_point(cfg, Qlm, Slm, Tlm, cost, phi)

        # With only Q_{1,0} set, should get nonzero Vr and zero Vt, Vp
        @test abs(Vr) > 1e-10  # Radial component should be nonzero
        @test isapprox(Vt, 0.0; atol=1e-12)
        @test isapprox(Vp, 0.0; atol=1e-12)
    end

    @testset "SHqst_to_lat matches full QST synthesis at grid latitudes" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        rng = MersenneTwister(45)
        # Create random QST coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Slm = randn(rng, ComplexF64, cfg.nlm)
        Tlm = randn(rng, ComplexF64, cfg.nlm)

        # Make m=0 real
        for k in 1:cfg.nlm
            if cfg.mi[k] == 0
                Qlm[k] = real(Qlm[k])
                Slm[k] = real(Slm[k])
                Tlm[k] = real(Tlm[k])
            end
        end

        # Full QST synthesis for reference — unpack to matrix form
        Q_mat = zeros(ComplexF64, lmax + 1, lmax + 1)
        S_mat = zeros(ComplexF64, lmax + 1, lmax + 1)
        T_mat = zeros(ComplexF64, lmax + 1, lmax + 1)
        for k in 1:cfg.nlm
            l = cfg.li[k]; m = cfg.mi[k]
            Q_mat[l+1, m+1] = Qlm[k]
            S_mat[l+1, m+1] = Slm[k]
            T_mat[l+1, m+1] = Tlm[k]
        end
        Vr_ref, Vt_ref, Vp_ref = synthesis_qst(cfg, Q_mat, S_mat, T_mat; real_output=true)

        # Test at a grid latitude
        i = 2  # Pick a non-pole latitude
        cost = cfg.x[i]
        Vr_lat, Vt_lat, Vp_lat = SHTnsKit.SHqst_to_lat(cfg, Qlm, Slm, Tlm, cost)

        @test length(Vr_lat) == cfg.nlon
        @test isapprox(Vr_lat, Vr_ref[i, :]; rtol=1e-9, atol=1e-11)
        @test isapprox(Vt_lat, Vt_ref[i, :]; rtol=1e-9, atol=1e-11)
        @test isapprox(Vp_lat, Vp_ref[i, :]; rtol=1e-9, atol=1e-11)
    end
end
