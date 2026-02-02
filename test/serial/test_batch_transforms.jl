# SHTnsKit.jl - Batch Transform Tests
# Tests for multi-field batch transforms

using Test
using Random
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Batch Transforms" begin
    @testset "Scalar batch analysis" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(70)

        nfields = 5

        # Multiple fields as 3D array
        fields = randn(rng, nlat, nlon, nfields)

        # Batch analysis
        alm_batch = analysis_batch(cfg, fields)
        @test size(alm_batch) == (lmax+1, lmax+1, nfields)

        # Verify matches individual analysis
        for k in 1:nfields
            alm_single = analysis(cfg, fields[:, :, k])
            @test isapprox(alm_batch[:, :, k], alm_single; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "Scalar batch synthesis" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(71)

        nfields = 4

        # Random coefficients
        alm_batch = randn(rng, ComplexF64, lmax+1, lmax+1, nfields)
        for k in 1:nfields
            alm_batch[:, 1, k] .= real.(alm_batch[:, 1, k])
            for m in 0:lmax, l in 0:(m-1)
                alm_batch[l+1, m+1, k] = 0
            end
        end

        # Batch synthesis
        fields = synthesis_batch(cfg, alm_batch)
        @test size(fields) == (nlat, nlon, nfields)

        # Verify matches individual synthesis
        for k in 1:nfields
            f_single = synthesis(cfg, alm_batch[:, :, k]; real_output=true)
            @test isapprox(fields[:, :, k], f_single; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "Scalar batch roundtrip" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(72)

        nfields = 3
        fields = randn(rng, nlat, nlon, nfields)

        # Forward and back
        alm_batch = analysis_batch(cfg, fields)
        fields_back = synthesis_batch(cfg, alm_batch)

        @test isapprox(fields_back, fields; rtol=1e-10, atol=1e-12)
    end

    @testset "Batch size configuration" begin
        # Test batch size controls
        original_size = get_batch_size()

        set_batch_size!(4)
        @test get_batch_size() == 4

        set_batch_size!(8)
        @test get_batch_size() == 8

        set_batch_size!(16)
        @test get_batch_size() == 16

        reset_batch_size!()
        @test get_batch_size() == original_size
    end

    @testset "Vector batch transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(73)

        nfields = 3

        # Multiple vector fields
        Vt_batch = randn(rng, nlat, nlon, nfields)
        Vp_batch = randn(rng, nlat, nlon, nfields)

        # Batch analysis
        Slm_batch, Tlm_batch = spat_to_SHsphtor_batch(cfg, Vt_batch, Vp_batch)
        @test size(Slm_batch) == (lmax+1, lmax+1, nfields)
        @test size(Tlm_batch) == (lmax+1, lmax+1, nfields)

        # Batch synthesis
        Vt_back, Vp_back = SHsphtor_to_spat_batch(cfg, Slm_batch, Tlm_batch)
        @test size(Vt_back) == (nlat, nlon, nfields)
        @test size(Vp_back) == (nlat, nlon, nfields)

        # Verify roundtrip
        @test isapprox(Vt_back, Vt_batch; rtol=1e-9, atol=1e-11)
        @test isapprox(Vp_back, Vp_batch; rtol=1e-9, atol=1e-11)
    end

    @testset "QST batch transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(74)

        nfields = 2

        # Multiple 3D vector fields
        Vr_batch = randn(rng, nlat, nlon, nfields)
        Vt_batch = randn(rng, nlat, nlon, nfields)
        Vp_batch = randn(rng, nlat, nlon, nfields)

        # Batch analysis
        Qlm_batch, Slm_batch, Tlm_batch = spat_to_SHqst_batch(cfg, Vr_batch, Vt_batch, Vp_batch)
        @test size(Qlm_batch) == (lmax+1, lmax+1, nfields)

        # Batch synthesis
        Vr_back, Vt_back, Vp_back = SHqst_to_spat_batch(cfg, Qlm_batch, Slm_batch, Tlm_batch)

        # Verify roundtrip
        @test isapprox(Vr_back, Vr_batch; rtol=1e-9, atol=1e-11)
        @test isapprox(Vt_back, Vt_batch; rtol=1e-9, atol=1e-11)
        @test isapprox(Vp_back, Vp_batch; rtol=1e-9, atol=1e-11)
    end
end
