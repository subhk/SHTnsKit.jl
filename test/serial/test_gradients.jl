# SHTnsKit.jl - Energy Gradient Tests
# Tests for analytic gradients validated against finite differences

using Test
using Random
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Energy Gradients" begin
    @testset "Scalar energy gradient (matrix form)" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(120)

        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Analytic gradient
        grad = grad_energy_scalar_alm(cfg, alm)

        # Finite difference validation
        ϵ = 1e-7
        h = randn(rng, ComplexF64, lmax+1, lmax+1)
        h[:, 1] .= real.(h[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            h[l+1, m+1] = 0
        end

        E_plus = energy_scalar(cfg, alm .+ ϵ .* h)
        E_minus = energy_scalar(cfg, alm .- ϵ .* h)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(grad) .* h))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Scalar energy gradient multiple directions" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(121)

        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        grad = grad_energy_scalar_alm(cfg, alm)
        ϵ = 1e-7

        # Test multiple random directions
        for _ in 1:5
            h = randn(rng, ComplexF64, lmax+1, lmax+1)
            h[:, 1] .= real.(h[:, 1])
            for m in 0:lmax, l in 0:(m-1)
                h[l+1, m+1] = 0
            end

            E_plus = energy_scalar(cfg, alm .+ ϵ .* h)
            E_minus = energy_scalar(cfg, alm .- ϵ .* h)
            dE_fd = (E_plus - E_minus) / (2ϵ)
            dE_ad = real(sum(conj(grad) .* h))

            @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
        end
    end

    @testset "Vector energy gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(122)

        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Slm[1, :] .= 0; Tlm[1, :] .= 0

        # Analytic gradient
        gS, gT = grad_energy_vector_Slm_Tlm(cfg, Slm, Tlm)

        # Finite difference
        ϵ = 1e-7
        hS = randn(rng, ComplexF64, lmax+1, lmax+1)
        hT = randn(rng, ComplexF64, lmax+1, lmax+1)
        hS[:, 1] .= real.(hS[:, 1])
        hT[:, 1] .= real.(hT[:, 1])
        hS[1, :] .= 0; hT[1, :] .= 0

        E_plus = energy_vector(cfg, Slm .+ ϵ .* hS, Tlm .+ ϵ .* hT)
        E_minus = energy_vector(cfg, Slm .- ϵ .* hS, Tlm .- ϵ .* hT)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(gS) .* hS) + sum(conj(gT) .* hT))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Enstrophy gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(123)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        # Analytic gradient
        gT = grad_enstrophy_Tlm(cfg, Tlm)

        # Finite difference
        ϵ = 1e-7
        h = randn(rng, ComplexF64, lmax+1, lmax+1)
        h[:, 1] .= real.(h[:, 1])
        h[1, :] .= 0

        Z_plus = enstrophy(cfg, Tlm .+ ϵ .* h)
        Z_minus = enstrophy(cfg, Tlm .- ϵ .* h)
        dZ_fd = (Z_plus - Z_minus) / (2ϵ)
        dZ_ad = real(sum(conj(gT) .* h))

        @test isapprox(dZ_ad, dZ_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Packed scalar gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(124)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        gQ = grad_energy_scalar_packed(cfg, Qlm)
        @test length(gQ) == cfg.nlm

        ϵ = 1e-7
        h = randn(rng, ComplexF64, cfg.nlm)
        h[1:lmax+1] .= real.(h[1:lmax+1])

        E_plus = energy_scalar_packed(cfg, Qlm .+ ϵ .* h)
        E_minus = energy_scalar_packed(cfg, Qlm .- ϵ .* h)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(gQ) .* h))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Packed vector gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(125)

        Sp = randn(rng, ComplexF64, cfg.nlm)
        Tp = randn(rng, ComplexF64, cfg.nlm)
        Sp[1:lmax+1] .= real.(Sp[1:lmax+1])
        Tp[1:lmax+1] .= real.(Tp[1:lmax+1])
        # l=0 should be zero for vectors
        Sp[1] = 0; Tp[1] = 0

        gS, gT = grad_energy_vector_packed(cfg, Sp, Tp)

        ϵ = 1e-7
        hS = randn(rng, ComplexF64, cfg.nlm)
        hT = randn(rng, ComplexF64, cfg.nlm)
        hS[1:lmax+1] .= real.(hS[1:lmax+1])
        hT[1:lmax+1] .= real.(hT[1:lmax+1])
        hS[1] = 0; hT[1] = 0

        E_plus = energy_vector_packed(cfg, Sp .+ ϵ .* hS, Tp .+ ϵ .* hT)
        E_minus = energy_vector_packed(cfg, Sp .- ϵ .* hS, Tp .- ϵ .* hT)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(gS) .* hS) + sum(conj(gT) .* hT))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Grid energy scalar gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(126)

        f = randn(rng, nlat, nlon)
        gf = grad_grid_energy_scalar_field(cfg, f)

        @test size(gf) == (nlat, nlon)

        ϵ = 1e-7
        h = randn(rng, nlat, nlon)

        E_plus = grid_energy_scalar(cfg, f .+ ϵ .* h)
        E_minus = grid_energy_scalar(cfg, f .- ϵ .* h)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = sum(gf .* h)

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Grid energy vector gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(127)

        Vt = randn(rng, nlat, nlon)
        Vp = randn(rng, nlat, nlon)
        gVt, gVp = grad_grid_energy_vector_fields(cfg, Vt, Vp)

        ϵ = 1e-7
        ht = randn(rng, nlat, nlon)
        hp = randn(rng, nlat, nlon)

        E_plus = grid_energy_vector(cfg, Vt .+ ϵ .* ht, Vp .+ ϵ .* hp)
        E_minus = grid_energy_vector(cfg, Vt .- ϵ .* ht, Vp .- ϵ .* hp)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = sum(gVt .* ht) + sum(gVp .* hp)

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end
end
