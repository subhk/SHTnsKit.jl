# SHTnsKit.jl - Complex and Packed Format Tests
# Tests for complex transforms and packed coefficient storage

using Test
using Random
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Complex and Packed Format" begin
    @testset "Complex field transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(130)

        # Complex spatial field
        f_cplx = randn(rng, ComplexF64, nlat, nlon)

        # Analysis
        alm = spat_cplx_to_SH(cfg, vec(f_cplx))

        # Synthesis
        f_back = SH_to_spat_cplx(cfg, alm)

        @test isapprox(f_back, vec(f_cplx); rtol=1e-10, atol=1e-12)
    end

    @testset "LM_cplx indexing consistency" begin
        lmax = 5
        mmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)

        # Test index consistency
        for l in 0:lmax
            for m in -l:l
                if abs(m) <= mmax
                    idx1 = LM_cplx(cfg, l, m)
                    idx2 = LM_cplx_index(lmax, mmax, l, m)
                    @test idx1 == idx2
                end
            end
        end
    end

    @testset "LM_cplx index bounds" begin
        lmax = 5
        mmax = 5
        nlm_cplx = nlm_cplx_calc(lmax, mmax, 1)

        # All indices should be in [0, nlm_cplx)
        for l in 0:lmax
            for m in -l:l
                if abs(m) <= mmax
                    idx = LM_cplx_index(lmax, mmax, l, m)
                    @test 0 <= idx < nlm_cplx
                end
            end
        end

        # No duplicate indices
        indices = Set{Int}()
        for l in 0:lmax
            for m in -l:l
                if abs(m) <= mmax
                    idx = LM_cplx_index(lmax, mmax, l, m)
                    @test !(idx in indices)
                    push!(indices, idx)
                end
            end
        end
        @test length(indices) == nlm_cplx
    end

    @testset "Hermitian symmetry for real fields" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(131)

        # Real spatial field
        f_real = randn(rng, nlat, nlon)

        # Analyze to dense matrix
        alm = analysis(cfg, f_real)

        # m=0 coefficients should be real
        @test all(abs.(imag.(alm[:, 1])) .< 1e-12)

        # Synthesize back should give real field
        f_back = synthesis(cfg, alm; real_output=true)
        @test isapprox(f_back, f_real; rtol=1e-10, atol=1e-12)
    end

    @testset "Packed to matrix conversion" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(132)

        # Random packed coefficients
        Qlm_packed = randn(rng, ComplexF64, cfg.nlm)
        Qlm_packed[1:lmax+1] .= real.(Qlm_packed[1:lmax+1])

        # Convert to matrix
        alm_matrix = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:cfg.mmax
            for l in m:cfg.lmax
                idx = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                alm_matrix[l+1, m+1] = Qlm_packed[idx]
            end
        end

        # Energy should be equal
        E_packed = energy_scalar_packed(cfg, Qlm_packed)
        E_matrix = energy_scalar(cfg, alm_matrix)
        @test isapprox(E_packed, E_matrix; rtol=1e-10)
    end

    @testset "Complex point evaluation" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(133)

        # Complex coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Evaluate at points
        for cost in [-0.5, 0.0, 0.5]
            for phi in [0.0, π/2, π]
                val = SH_to_point_cplx(cfg, alm, cost, phi)
                @test !isnan(val) && !isinf(val)
            end
        end
    end

    @testset "Complex vector transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(134)

        # Complex velocity fields
        Vt = randn(rng, ComplexF64, nlat, nlon)
        Vp = randn(rng, ComplexF64, nlat, nlon)

        # Roundtrip
        Slm, Tlm = spat_cplx_to_SHsphtor(cfg, Vt, Vp)
        Vt_back, Vp_back = SHsphtor_to_spat_cplx(cfg, Slm, Tlm)

        @test isapprox(Vt_back, Vt; rtol=1e-9, atol=1e-11)
        @test isapprox(Vp_back, Vp; rtol=1e-9, atol=1e-11)
    end

    @testset "Packed format efficiency" begin
        # Verify packed format uses less memory than full complex
        for lmax in [8, 16, 32]
            nlm_real = nlm_calc(lmax, lmax, 1)
            nlm_cplx = nlm_cplx_calc(lmax, lmax, 1)

            # Real-packed should be smaller than complex
            @test nlm_real < nlm_cplx

            # Check expected sizes
            @test nlm_real == (lmax + 1) * (lmax + 2) ÷ 2
            @test nlm_cplx == (lmax + 1)^2
        end
    end

    @testset "Zero coefficient handling" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # All-zero coefficients
        alm = zeros(ComplexF64, lmax+1, lmax+1)

        # Synthesis should give zero field
        f = synthesis(cfg, alm; real_output=true)
        @test all(abs.(f) .< 1e-14)

        # Zero spatial field
        f_zero = zeros(nlat, nlon)
        alm_back = analysis(cfg, f_zero)
        @test all(abs.(alm_back) .< 1e-14)
    end
end
