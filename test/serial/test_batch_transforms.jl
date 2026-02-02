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

        # Start with spectral coefficients
        alm_batch = randn(rng, ComplexF64, lmax+1, lmax+1, nfields)
        for k in 1:nfields
            alm_batch[:, 1, k] .= real.(alm_batch[:, 1, k])
            for m in 0:lmax, l in 0:(m-1)
                alm_batch[l+1, m+1, k] = 0
            end
        end

        # Synth then analysis
        fields = synthesis_batch(cfg, alm_batch)
        alm_back = analysis_batch(cfg, fields)

        @test isapprox(alm_back, alm_batch; rtol=1e-10, atol=1e-12)
    end

    @testset "Batch size configuration" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Test batch size controls
        original_size = get_batch_size(cfg)

        set_batch_size!(cfg, 4)
        @test get_batch_size(cfg) == 4

        set_batch_size!(cfg, 8)
        @test get_batch_size(cfg) == 8

        set_batch_size!(cfg, 16)
        @test get_batch_size(cfg) == 16

        reset_batch_size!(cfg)
        @test get_batch_size(cfg) == original_size
    end

    @testset "Vector batch transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(73)

        nfields = 3

        # Test individual transforms with spectral-to-spatial-to-spectral roundtrip
        for k in 1:nfields
            Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Slm[:, 1] .= real.(Slm[:, 1])
            Tlm[:, 1] .= real.(Tlm[:, 1])
            Slm[1, :] .= 0  # l=0 must be zero for vectors
            Tlm[1, :] .= 0
            # Zero invalid (l < m) positions
            for m in 0:lmax, l in 0:(m-1)
                Slm[l+1, m+1] = 0
                Tlm[l+1, m+1] = 0
            end

            Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm; real_output=true)
            @test size(Vt) == (nlat, nlon)
            @test size(Vp) == (nlat, nlon)

            Slm_back, Tlm_back = spat_to_SHsphtor(cfg, Vt, Vp)
            @test size(Slm_back) == (lmax+1, lmax+1)
            @test size(Tlm_back) == (lmax+1, lmax+1)

            # Verify roundtrip
            @test isapprox(Slm_back, Slm; rtol=1e-9, atol=1e-11)
            @test isapprox(Tlm_back, Tlm; rtol=1e-9, atol=1e-11)
        end
    end

    @testset "QST batch transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(74)

        nfields = 2

        # Test individual transforms with spectral-to-spatial-to-spectral roundtrip
        for k in 1:nfields
            Qlm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Qlm[:, 1] .= real.(Qlm[:, 1])
            Slm[:, 1] .= real.(Slm[:, 1])
            Tlm[:, 1] .= real.(Tlm[:, 1])
            Slm[1, :] .= 0; Tlm[1, :] .= 0
            # Zero invalid (l < m) positions for all coefficients
            for m in 0:lmax, l in 0:(m-1)
                Qlm[l+1, m+1] = 0
                Slm[l+1, m+1] = 0
                Tlm[l+1, m+1] = 0
            end

            Vr, Vt, Vp = SHqst_to_spat(cfg, Qlm, Slm, Tlm; real_output=true)
            @test size(Vr) == (nlat, nlon)

            Qlm_back, Slm_back, Tlm_back = spat_to_SHqst(cfg, Vr, Vt, Vp)
            @test size(Qlm_back) == (lmax+1, lmax+1)

            # Verify roundtrip
            @test isapprox(Qlm_back, Qlm; rtol=1e-9, atol=1e-11)
            @test isapprox(Slm_back, Slm; rtol=1e-9, atol=1e-11)
            @test isapprox(Tlm_back, Tlm; rtol=1e-9, atol=1e-11)
        end
    end
end
