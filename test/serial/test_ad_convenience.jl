# SHTnsKit.jl - AD convenience-wrapper tests
#
# Covers the exported gradient helpers implemented in the Zygote / ForwardDiff
# package extensions, previously untested:
#   zgrad_scalar_energy, zgrad_vector_energy, zgrad_enstrophy_Tlm
#   fdgrad_scalar_energy, fdgrad_vector_energy
#
# Each wrapper packages a standard energy loss; we validate it against a central
# finite-difference directional derivative.

using Test
using Random
using LinearAlgebra
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

_has_zygote = try; @eval using Zygote; true; catch; false; end
_has_forwarddiff = try; @eval using ForwardDiff; true; catch; false; end

@testset "AD convenience wrappers" begin
    lmax = 4
    nlat = lmax + 2
    nlon = 2 * lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(606)
    ϵ = 1e-6

    f0 = randn(rng, nlat, nlon)
    Lscalar(x) = energy_scalar(cfg, analysis(cfg, x))
    h = randn(rng, nlat, nlon)
    fd_scalar = (Lscalar(f0 .+ ϵ .* h) - Lscalar(f0 .- ϵ .* h)) / (2ϵ)

    if _has_zygote
        @testset "zgrad_scalar_energy" begin
            g = zgrad_scalar_energy(cfg, f0)
            @test size(g) == size(f0)
            @test isapprox(sum(g .* h), fd_scalar; rtol=5e-4, atol=1e-7)
        end

        @testset "zgrad_vector_energy" begin
            Vt = randn(rng, nlat, nlon); Vp = randn(rng, nlat, nlon)
            Lvec(a, b) = begin
                S, T = analysis_sphtor(cfg, a, b)
                energy_vector(cfg, S, T)
            end
            gVt, gVp = zgrad_vector_energy(cfg, Vt, Vp)
            @test size(gVt) == size(Vt) && size(gVp) == size(Vp)
            ht = randn(rng, nlat, nlon); hp = randn(rng, nlat, nlon)
            fd_vt = (Lvec(Vt .+ ϵ .* ht, Vp) - Lvec(Vt .- ϵ .* ht, Vp)) / (2ϵ)
            fd_vp = (Lvec(Vt, Vp .+ ϵ .* hp) - Lvec(Vt, Vp .- ϵ .* hp)) / (2ϵ)
            @test isapprox(sum(gVt .* ht), fd_vt; rtol=5e-4, atol=1e-7)
            @test isapprox(sum(gVp .* hp), fd_vp; rtol=5e-4, atol=1e-7)
            @test_throws DimensionMismatch zgrad_vector_energy(cfg, Vt, randn(rng, nlat, nlon - 1))
        end

        @testset "zgrad_enstrophy_Tlm" begin
            T0 = randn(rng, ComplexF64, lmax + 1, lmax + 1)
            T0[:, 1] .= real.(T0[:, 1]); T0[1, :] .= 0
            g = zgrad_enstrophy_Tlm(cfg, T0)
            @test size(g) == size(T0)
            hT = randn(rng, ComplexF64, lmax + 1, lmax + 1)
            hT[:, 1] .= real.(hT[:, 1]); hT[1, :] .= 0
            fd = (enstrophy(cfg, T0 .+ ϵ .* hT) - enstrophy(cfg, T0 .- ϵ .* hT)) / (2ϵ)
            @test isapprox(real(sum(conj(g) .* hT)), fd; rtol=5e-4, atol=1e-7)
        end
    else
        @info "Skipping Zygote convenience-gradient tests (package not available)"
    end

    if _has_forwarddiff
        @testset "fdgrad_scalar_energy" begin
            g = fdgrad_scalar_energy(cfg, f0)
            @test size(g) == size(f0)
            @test isapprox(sum(g .* h), fd_scalar; rtol=5e-4, atol=1e-7)
        end

        @testset "fdgrad_vector_energy" begin
            Vt = randn(rng, nlat, nlon); Vp = randn(rng, nlat, nlon)
            ht = randn(rng, nlat, nlon); hp = randn(rng, nlat, nlon)
            Lvec(a, b) = begin
                S, T = analysis_sphtor(cfg, a, b)
                energy_vector(cfg, S, T)
            end
            fd_vt = (Lvec(Vt .+ ϵ .* ht, Vp) - Lvec(Vt .- ϵ .* ht, Vp)) / (2ϵ)
            fd_vp = (Lvec(Vt, Vp .+ ϵ .* hp) - Lvec(Vt, Vp .- ϵ .* hp)) / (2ϵ)
            gVt, gVp = fdgrad_vector_energy(cfg, Vt, Vp)
            @test size(gVt) == size(Vt) && size(gVp) == size(Vp)
            @test isapprox(sum(gVt .* ht), fd_vt; rtol=5e-4, atol=1e-7)
            @test isapprox(sum(gVp .* hp), fd_vp; rtol=5e-4, atol=1e-7)
        end
    else
        @info "Skipping ForwardDiff convenience-gradient tests (package not available)"
    end
end
