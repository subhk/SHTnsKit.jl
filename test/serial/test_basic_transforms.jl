# SHTnsKit.jl - Basic Scalar Transform Tests
# Tests for analysis, synthesis, and SHTPlan

using Test
using Random
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Basic Scalar Transforms" begin
    @testset "Analysis-synthesis roundtrip" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(42)

        # Create random spectral coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 must be real for real fields
        # Zero invalid entries (l < m)
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0.0
        end

        # Roundtrip: spectral -> spatial -> spectral
        f = synthesis(cfg, alm; real_output=true)
        alm_recovered = analysis(cfg, f)

        @test isapprox(alm_recovered, alm; rtol=1e-10, atol=1e-12)
        VERBOSE && @info "Scalar roundtrip" max_err=maximum(abs.(alm_recovered - alm))
    end

    @testset "SHTPlan optimized transforms" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(43)

        # Create plan
        plan = SHTPlan(cfg)

        # Start with spectral coefficients for reliable roundtrip
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 real
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # In-place synthesis
        f = zeros(nlat, nlon)
        synthesis!(plan, f, alm)

        # In-place analysis
        alm_back = zeros(ComplexF64, lmax+1, lmax+1)
        analysis!(plan, alm_back, f)

        # Compare recovered coefficients
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)

        # Also verify plan matches non-planned version
        alm_ref = analysis(cfg, f)
        @test isapprox(alm_back, alm_ref; rtol=1e-10, atol=1e-12)
    end

    @testset "Packed vector format" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(44)

        # Start with packed spectral coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])  # m=0 real (first lmax+1 indices)

        # Packed synthesis
        f = synthesis_packed(cfg, Qlm)
        @test length(f) == nlat * nlon

        # Packed analysis
        Qlm_back = analysis_packed(cfg, f)
        @test length(Qlm_back) == cfg.nlm

        # Verify roundtrip
        @test isapprox(Qlm_back, Qlm; rtol=1e-10, atol=1e-12)
    end

    @testset "Single spherical harmonic modes" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Test individual modes
        for l in 0:lmax
            for m in 0:l
                alm = zeros(ComplexF64, lmax+1, lmax+1)
                alm[l+1, m+1] = m == 0 ? 1.0 : 1.0 + 0.5im

                f = synthesis(cfg, alm; real_output=true)
                alm_rec = analysis(cfg, f)

                @test isapprox(alm_rec[l+1, m+1], alm[l+1, m+1]; rtol=1e-10, atol=1e-12)
            end
        end
    end

    @testset "Real field consistency" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(45)

        # Start with spectral coefficients, real for m=0
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 real
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Synthesize to real field
        f = synthesis(cfg, alm; real_output=true)

        # Analyze back
        alm_back = analysis(cfg, f)

        # m=0 coefficients should remain real
        @test all(abs.(imag.(alm_back[:, 1])) .< 1e-12)

        # Roundtrip should preserve coefficients
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "Unfused loop path" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(46)

        # Random spectral coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Compare fused vs unfused synthesis
        f_fused = synthesis(cfg, alm; real_output=true, use_fused_loops=true)
        f_unfused = synthesis(cfg, alm; real_output=true, use_fused_loops=false)
        @test isapprox(f_fused, f_unfused; rtol=1e-10, atol=1e-12)

        # Compare fused vs unfused analysis
        alm_fused = analysis(cfg, f_fused; use_fused_loops=true)
        alm_unfused = analysis(cfg, f_fused; use_fused_loops=false)
        @test isapprox(alm_fused, alm_unfused; rtol=1e-10, atol=1e-12)

        # Verify roundtrip with unfused path
        @test isapprox(alm_unfused, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "FFT scratch buffer reuse" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(47)

        # Preallocate scratch buffer
        scratch = zeros(ComplexF64, nlat, nlon)

        # Random spectral coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Synthesis with scratch buffer
        f_scratch = synthesis(cfg, alm; real_output=true, fft_scratch=scratch)
        f_no_scratch = synthesis(cfg, alm; real_output=true)
        @test isapprox(f_scratch, f_no_scratch; rtol=1e-12, atol=1e-14)

        # Analysis with scratch buffer
        alm_scratch = analysis(cfg, f_scratch; fft_scratch=scratch)
        alm_no_scratch = analysis(cfg, f_scratch)
        @test isapprox(alm_scratch, alm_no_scratch; rtol=1e-12, atol=1e-14)
    end

    @testset "PLM tables path" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1

        # Config with precomputed PLM tables (use prepare_plm_tables!)
        cfg_plm = create_gauss_config(lmax, nlat; nlon=nlon)
        prepare_plm_tables!(cfg_plm)  # Enable PLM tables

        cfg_otf = create_gauss_config(lmax, nlat; nlon=nlon)
        # Default is on-the-fly (no tables)

        @test cfg_plm.use_plm_tables == true
        @test cfg_otf.use_plm_tables == false

        rng = MersenneTwister(48)
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Both paths should give same results
        f_plm = synthesis(cfg_plm, alm; real_output=true)
        f_otf = synthesis(cfg_otf, alm; real_output=true)
        @test isapprox(f_plm, f_otf; rtol=1e-10, atol=1e-12)

        alm_plm = analysis(cfg_plm, f_plm)
        alm_otf = analysis(cfg_otf, f_otf)
        @test isapprox(alm_plm, alm_otf; rtol=1e-10, atol=1e-12)

        # Roundtrip should work with PLM tables
        @test isapprox(alm_plm, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "Axisymmetric transforms (analysis_axisym/synthesis_axisym)" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(104)

        # Create random m=0 only coefficients (axisymmetric field)
        Ql = randn(rng, lmax+1)  # Real coefficients for m=0

        # Axisymmetric synthesis: spectral -> latitude values
        f_lat = synthesis_axisym(cfg, complex.(Ql))

        @test length(f_lat) == nlat
        @test eltype(f_lat) <: Real

        # Axisymmetric analysis: latitude values -> spectral
        Ql_rec = analysis_axisym(cfg, f_lat)

        @test length(Ql_rec) == lmax + 1
        # m=0 coefficients should be real (imaginary part ~0)
        @test maximum(abs.(imag.(Ql_rec))) < 1e-10
        @test isapprox(real.(Ql_rec), Ql; rtol=1e-9, atol=1e-11)
    end

    @testset "Axisymmetric truncated transforms (analysis_axisym_l/synthesis_axisym_l)" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        ltr = lmax - 3
        rng = MersenneTwister(105)

        # Create random m=0 only coefficients up to ltr
        Ql = zeros(lmax+1)
        Ql[1:ltr+1] = randn(rng, ltr+1)

        # Truncated axisymmetric synthesis
        f_lat = synthesis_axisym_l(cfg, complex.(Ql), ltr)

        @test length(f_lat) == nlat

        # Reference: full synthesis with zeroed high modes
        f_ref = synthesis_axisym(cfg, complex.(Ql))
        @test isapprox(f_lat, f_ref; rtol=1e-10, atol=1e-12)

        # Truncated axisymmetric analysis
        Ql_rec = analysis_axisym_l(cfg, f_lat, ltr)

        @test length(Ql_rec) == ltr + 1
        @test isapprox(real.(Ql_rec), Ql[1:ltr+1]; rtol=1e-9, atol=1e-11)
    end
end
