# SHTnsKit.jl - API Compatibility Layer Tests
# Exercises the SHTns C-API parity shims in src/api_compat.jl.

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "API Compatibility (shtns.h parity)" begin
    @testset "Flag constants" begin
        @test SHT_GAUSS == 0
        @test SHT_AUTO == 1
        @test SHT_REGULAR == 2
        @test SHT_REG_FAST == 3
        @test SHT_QUICK_INIT == 4
        @test SHT_REGULAR_POLES == 5
        @test SHT_GAUSS_FLY == 6
        @test SHT_REG_DCT == SHT_REG_FAST
        @test SHT_NO_CS_PHASE == 256*4
        @test SHT_SOUTH_POLE_FIRST == 256*32
        @test SHT_ALLOW_PADDING == 256*256
        @test SHT_ROBERT_FORM == 256*512
    end

    @testset "shtns_init: grid selection" begin
        lmax, mmax, mres = 6, 6, 1
        nlat, nphi = lmax + 2, 2*mmax + 1

        cfg_g = shtns_init(SHT_GAUSS, lmax, mmax, mres, nlat, nphi)
        @test cfg_g.grid_type == :gauss
        @test cfg_g.lmax == lmax
        @test cfg_g.mmax == mmax
        @test cfg_g.mres == mres

        cfg_a = shtns_init(SHT_AUTO, lmax, mmax, mres, nlat, nphi)
        @test cfg_a.grid_type == :gauss  # AUTO → gauss

        cfg_r = shtns_init(SHT_REGULAR, lmax, mmax, mres, nlat + 2, nphi)
        @test cfg_r.grid_type == :regular

        cfg_rp = shtns_init(SHT_REGULAR_POLES, lmax, mmax, mres, nlat + 2, nphi)
        @test cfg_rp.grid_type == :regular_poles

        cfg_fly = shtns_init(SHT_GAUSS_FLY, lmax, mmax, mres, nlat, nphi)
        @test cfg_fly.grid_type == :gauss
        @test cfg_fly.on_the_fly == true
    end

    @testset "shtns_init: option flags" begin
        lmax, mmax, mres = 5, 5, 1
        nlat, nphi = lmax + 2, 2*mmax + 1

        cfg_spf = shtns_init(SHT_GAUSS | SHT_SOUTH_POLE_FIRST, lmax, mmax, mres, nlat, nphi)
        @test cfg_spf.south_pole_first == true

        cfg_pad = shtns_init(SHT_GAUSS | SHT_ALLOW_PADDING, lmax, mmax, mres, nlat, nphi)
        @test cfg_pad.allow_padding == true
    end

    @testset "shtns_init: minimum grid size enforcement" begin
        lmax = 8
        # Ask for too-small nlat — library bumps up
        cfg = shtns_init(SHT_GAUSS, lmax, lmax, 1, 2, 3)
        @test cfg.nlat ≥ lmax + 1
        @test cfg.nlon ≥ 4
    end

    @testset "shtns_create + shtns_set_grid" begin
        lmax, mmax, mres = 6, 6, 1
        cfg = shtns_create(lmax, mmax, mres, 0)  # norm=0 orthonormal, CS-phase on
        @test cfg.lmax == lmax
        @test cfg.norm == :orthonormal
        @test cfg.cs_phase == true

        status = shtns_set_grid(cfg, SHT_GAUSS, 1e-10, lmax + 4, 2*mmax + 3)
        @test status == 0
        @test cfg.grid_type == :gauss
        @test cfg.nlat == lmax + 4
        @test cfg.nlon == 2*mmax + 3

        # Regular grid rebuild
        status2 = shtns_set_grid(cfg, SHT_REGULAR, 1e-10, lmax + 4, 2*mmax + 3)
        @test status2 == 0
        @test cfg.grid_type == :regular

        # Poles grid
        status3 = shtns_set_grid(cfg, SHT_REGULAR_POLES, 1e-10, lmax + 4, 2*mmax + 3)
        @test status3 == 0
        @test cfg.grid_type == :regular_poles
    end

    @testset "shtns_create parses norm bits" begin
        # base_norm=2 → schmidt; NO_CS_PHASE flag disables CS; REAL_NORM flag sets real_norm
        flags = 2 | SHT_NO_CS_PHASE | SHT_REAL_NORM
        cfg = shtns_create(4, 4, 1, flags)
        @test cfg.norm == :schmidt
        @test cfg.cs_phase == false
        @test cfg.real_norm == true
    end

    @testset "shtns_set_grid_auto" begin
        cfg = shtns_create(10, 10, 1, 0)
        nlat_ref = Ref(0); nphi_ref = Ref(0)
        status = shtns_set_grid_auto(cfg, SHT_GAUSS, 1e-10, 2, nlat_ref, nphi_ref)
        @test status == 0
        @test nlat_ref[] == cfg.lmax + 1
        @test nphi_ref[] ≥ 2*cfg.mmax + 1
    end

    @testset "shtns_create_with_grid reduces mmax" begin
        cfg = shtns_init(SHT_GAUSS, 8, 8, 1, 10, 17)
        cfg2 = shtns_create_with_grid(cfg, 4, 0)
        @test cfg2.mmax == 4
        @test cfg2.lmax == cfg.lmax
        @test cfg2.nlat == cfg.nlat

        @test_throws ArgumentError shtns_create_with_grid(cfg, cfg.mmax + 1, 0)
    end

    @testset "shtns_use_threads / reset / destroy / unset" begin
        @test shtns_use_threads(0) == 1       # clamped to ≥ 1
        @test shtns_use_threads(4) == 4
        @test shtns_reset() === nothing
        cfg = shtns_create(4, 4, 1, 0)
        @test shtns_destroy(cfg) === nothing
        @test shtns_unset_grid(cfg) === nothing
    end

    @testset "shtns_robert_form" begin
        cfg = shtns_init(SHT_GAUSS, 4, 4, 1, 6, 9)
        @test cfg.robert_form == false
        shtns_robert_form(cfg, 1)
        @test cfg.robert_form == true
        shtns_robert_form(cfg, 0)
        @test cfg.robert_form == false
    end

    @testset "shtns_verbose / print helpers" begin
        @test shtns_verbose(2) === nothing
        # Just ensure prints don't error
        @test shtns_print_version() === nothing
        buf = IOBuffer()
        info = shtns_get_build_info()
        @test isa(info, String) && !isempty(info)
        cfg = shtns_create(3, 3, 1, 0)
        redirect_stdout(devnull) do
            shtns_print_cfg(cfg)
        end
    end

    @testset "sh00_1 / sh10_ct / sh11_st / shlm_e1" begin
        cfg = shtns_create(4, 4, 1, 0)
        @test sh00_1(cfg) ≈ sqrt(4π)
        @test sh10_ct(cfg) ≈ sqrt(4π/3)
        @test sh11_st(cfg) ≈ -sqrt(2π/3)
        @test shlm_e1(cfg, 0, 0) == 1.0
        @test shlm_e1(cfg, 2, 1) == 1.0
        @test shlm_e1(cfg, 1, 5) == 0.0   # m > mmax
        @test shlm_e1(cfg, -1, 0) == 0.0  # l < m invalid
    end

    @testset "shtns_gauss_wts copies weights" begin
        cfg = shtns_init(SHT_GAUSS, 4, 4, 1, 6, 9)
        wts = zeros(cfg.nlat)
        n = shtns_gauss_wts(cfg, wts)
        @test n == cfg.nlat
        @test isapprox(sum(wts), 2.0; atol=1e-12)  # Gauss weights sum to 2

        # Shorter buffer — partial fill
        wts_short = zeros(3)
        n2 = shtns_gauss_wts(cfg, wts_short)
        @test n2 == 3
        @test wts_short == cfg.w[1:3]
    end

    @testset "legendre_sphPlm_array" begin
        cfg = shtns_init(SHT_GAUSS, 8, 8, 1, 10, 17)
        x = cfg.x[3]
        yl = zeros(cfg.lmax + 1)

        # m = 0
        n = legendre_sphPlm_array(cfg, cfg.lmax, 0, x, yl)
        @test n == cfg.lmax + 1
        @test all(isfinite, yl[1:n])

        # m = 2
        yl2 = zeros(cfg.lmax + 1)
        n2 = legendre_sphPlm_array(cfg, cfg.lmax, 2, x, yl2)
        @test n2 == cfg.lmax - 2 + 1

        # Out-of-range: im > mmax
        @test legendre_sphPlm_array(cfg, cfg.lmax, cfg.mmax + 1, x, yl) == 0
        # lmax > cfg.lmax
        @test legendre_sphPlm_array(cfg, cfg.lmax + 1, 0, x, yl) == 0
    end

    @testset "legendre_sphPlm_deriv_array" begin
        cfg = shtns_init(SHT_GAUSS, 6, 6, 1, 8, 13)
        x = cfg.x[2]
        sint = sqrt(1 - x*x)
        yl = zeros(cfg.lmax + 1); dyl = zeros(cfg.lmax + 1)

        n = legendre_sphPlm_deriv_array(cfg, cfg.lmax, 0, x, sint, yl, dyl)
        @test n == cfg.lmax + 1
        @test all(isfinite, yl[1:n])
        @test all(isfinite, dyl[1:n])
        # l=0 derivative is exactly zero
        @test abs(dyl[1]) < 1e-14
    end

    @testset "Memory helpers" begin
        buf = shtns_malloc(128)
        @test length(buf) == 128
        @test eltype(buf) == UInt8
        @test shtns_free(buf) === nothing
        @test shtns_free(nothing) === nothing
    end

    @testset "shtns_set_many / batch size" begin
        cfg = shtns_init(SHT_GAUSS, 4, 4, 1, 6, 9)
        got = shtns_set_many(cfg, 3, 0)
        @test got == 3
        @test get_batch_size(cfg) == 3
        # Clamp at ≥1
        got0 = shtns_set_many(cfg, 0, 0)
        @test got0 == 1
    end

    @testset "synthesis_packed_time / analysis_packed_time" begin
        cfg = shtns_init(SHT_GAUSS, 6, 6, 1, 8, 13)
        Qlm = randn(ComplexF64, cfg.nlm)
        Qlm[1:cfg.lmax+1] .= real.(Qlm[1:cfg.lmax+1])

        Vr = zeros(cfg.nspat)
        t = SHTnsKit.synthesis_packed_time(cfg, Qlm, Vr)
        @test t ≥ 0
        @test all(isfinite, Vr)

        Qlm_back = zeros(ComplexF64, cfg.nlm)
        t2 = SHTnsKit.analysis_packed_time(cfg, Vr, Qlm_back)
        @test t2 ≥ 0
        @test isapprox(Qlm_back, Qlm; rtol=1e-10, atol=1e-12)
    end

    @testset "Profiling stubs" begin
        cfg = shtns_create(2, 2, 1, 0)
        @test SHTnsKit.shtns_profiling(cfg, 1) === nothing
        t1 = Ref(1.0); t2 = Ref(2.0)
        r = SHTnsKit.shtns_profiling_read_time(cfg, t1, t2)
        @test r == 0.0
        @test t1[] == 0.0 && t2[] == 0.0
    end

    @testset "Coordinate macros" begin
        cfg = shtns_init(SHT_GAUSS, 4, 4, 1, 6, 8)
        nlon = cfg.nlon
        @test PHI_DEG(cfg, 0) == 0.0
        @test PHI_DEG(cfg, nlon) ≈ 360.0
        @test PHI_RAD(cfg, 0) == 0.0
        @test PHI_RAD(cfg, nlon) ≈ 2π
        @test PHI_DEG(cfg, 2) ≈ rad2deg(PHI_RAD(cfg, 2))
        @test THETA_RAD(cfg, 1) == cfg.θ[1]
        @test isapprox(THETA_DEG(cfg, 1), rad2deg(cfg.θ[1]); atol=1e-14)
    end

    @testset "NSPAT_ALLOC / NLM_ALLOC" begin
        cfg = shtns_init(SHT_GAUSS, 4, 4, 1, 6, 9)
        @test NLM_ALLOC(cfg) == cfg.nlm
        @test NSPAT_ALLOC(cfg) == cfg.nspat

        # With padding: spat_dist used if > 0
        set_allow_padding!(cfg)
        if cfg.spat_dist > 0
            @test NSPAT_ALLOC(cfg) == cfg.spat_dist
        end
    end

    @testset "save_config / load_config roundtrip" begin
        cfg = shtns_init(SHT_GAUSS, 6, 6, 1, 8, 13)
        shtns_robert_form(cfg, 1)
        path, io = mktemp()
        close(io)
        try
            save_config(cfg, path)
            cfg2 = load_config(path)
            @test cfg2.lmax == cfg.lmax
            @test cfg2.mmax == cfg.mmax
            @test cfg2.mres == cfg.mres
            @test cfg2.nlat == cfg.nlat
            @test cfg2.nlon == cfg.nlon
            @test cfg2.grid_type == cfg.grid_type
            @test cfg2.norm == cfg.norm
            @test cfg2.cs_phase == cfg.cs_phase
            @test cfg2.robert_form == cfg.robert_form
        finally
            rm(path; force=true)
        end
    end

    @testset "End-to-end: init → transform roundtrip" begin
        cfg = shtns_init(SHT_GAUSS, 8, 8, 1, 10, 17)
        f = randn(cfg.nlat, cfg.nlon)
        alm = analysis(cfg, f)
        f2 = synthesis(cfg, alm; real_output=true)
        alm2 = analysis(cfg, f2)
        @test isapprox(alm, alm2; rtol=1e-10, atol=1e-12)
    end
end
