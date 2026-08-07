# SHTnsKit.jl - Complex and Packed Format Tests
# Tests for complex transforms and packed coefficient storage

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Complex and Packed Format" begin
    @testset "Complex field transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(130)

        # Complex coefficients in packed format (for reliable roundtrip)
        nlm_cplx = nlm_cplx_calc(lmax, lmax, 1)
        alm = randn(rng, ComplexF64, nlm_cplx)

        # Synthesis
        f = synthesis_packed_cplx(cfg, alm)
        @test size(f) == (nlat, nlon)

        # Analysis
        alm_back = analysis_packed_cplx(cfg, f)

        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
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

        # Random spectral coefficients with real m=0
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 must be real
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Synthesize to real field
        f = synthesis(cfg, alm; real_output=true)

        # m=0 coefficients remain real after analysis
        alm_back = analysis(cfg, f)
        @test all(abs.(imag.(alm_back[:, 1])) .< 1e-12)

        # Roundtrip should recover original coefficients
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "LM_cplx negative-m rule for real fields" begin
        # This layout uses P̄_l^{|m|} for BOTH signs of m, so a real field's
        # Hermitian relation is a_{l,-m} = conj(a_{l,m}) with NO (-1)^m factor
        # (that factor belongs to the Y_l^m convention). Pin it: the distributed
        # `dist_analysis_packed_cplx` builds its −m half from exactly this rule.
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(20260807)
        f = randn(rng, nlat, nlon)

        alm = analysis(cfg, f)                       # m ≥ 0 half
        alm_c = analysis_packed_cplx(cfg, complex.(f))
        for l in 0:lmax, m in 1:min(l, cfg.mmax)
            ip = LM_cplx_index(lmax, cfg.mmax, l, m) + 1
            im_ = LM_cplx_index(lmax, cfg.mmax, l, -m) + 1
            @test isapprox(alm_c[ip], alm[l+1, m+1]; rtol=1e-10, atol=1e-12)
            @test isapprox(alm_c[im_], conj(alm[l+1, m+1]); rtol=1e-10, atol=1e-12)
        end

        # Filling the −m half with conj(a) reproduces the field; the (-1)^m
        # variant does not (and is not even real).
        packed = Vector{ComplexF64}(undef, SHTnsKit.nlm_cplx_calc(lmax, cfg.mmax, 1))
        wrong = similar(packed)
        for l in 0:lmax
            packed[LM_cplx_index(lmax, cfg.mmax, l, 0) + 1] = alm[l+1, 1]
            wrong[LM_cplx_index(lmax, cfg.mmax, l, 0) + 1] = alm[l+1, 1]
            for m in 1:min(l, cfg.mmax)
                ap = alm[l+1, m+1]
                packed[LM_cplx_index(lmax, cfg.mmax, l, m) + 1] = ap
                packed[LM_cplx_index(lmax, cfg.mmax, l, -m) + 1] = conj(ap)
                wrong[LM_cplx_index(lmax, cfg.mmax, l, m) + 1] = ap
                wrong[LM_cplx_index(lmax, cfg.mmax, l, -m) + 1] = ((-1)^m) * conj(ap)
            end
        end
        z = synthesis_packed_cplx(cfg, packed)
        @test isapprox(z, complex.(synthesis(cfg, alm; real_output=true)); rtol=1e-10, atol=1e-12)
        @test maximum(abs, imag.(z)) < 1e-12
        zw = synthesis_packed_cplx(cfg, wrong)
        @test !isapprox(zw, z; rtol=1e-6, atol=1e-8)
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

        # Complex coefficients in packed format (vector, not matrix)
        nlm_cplx = nlm_cplx_calc(lmax, lmax, 1)
        alm = randn(rng, ComplexF64, nlm_cplx)

        # Evaluate at points - synthesis_point_cplx takes packed vector
        for cost in [-0.5, 0.0, 0.5]
            for phi in [0.0, π/2, π]
                val = synthesis_point_cplx(cfg, alm, cost, phi)
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

        # Random S/T coefficients (spectral domain)
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax
            for l in max(1, m):lmax
                Slm[l+1, m+1] = randn(rng) + im * randn(rng)
                Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
        end

        # Roundtrip in spectral domain: synth then analysis
        Vt, Vp = synthesis_sphtor_cplx(cfg, Slm, Tlm)
        Slm_back, Tlm_back = analysis_sphtor_cplx(cfg, Vt, Vp)

        @test isapprox(Slm_back, Slm; rtol=1e-8, atol=1e-10)
        @test isapprox(Tlm_back, Tlm; rtol=1e-8, atol=1e-10)
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
