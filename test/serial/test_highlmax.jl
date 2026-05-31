using Test, SHTnsKit

@testset "High-lmax transforms are finite and accurate" begin
    for lmax in (150, 160, 256, 512)
        nlat = lmax + 2; nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        # band-limited field from a decaying spectrum
        alm0 = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax, l in m:lmax
            sc = 1/(1+l)^2
            alm0[l+1, m+1] = m==0 ? complex(sc) : complex(sc, 0.5sc)
        end
        f = SHTnsKit.synthesis(cfg, alm0; real_output=true)
        @test all(isfinite, f)
        a = SHTnsKit.analysis(cfg, f)
        @test all(isfinite, a)
        @test isapprox(a, alm0; rtol=1e-9, atol=1e-11)
    end
end

@testset "scalar OTF round-trip lmax 256/512 orthonormal" begin
    for lmax in (256, 512)
        cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)  # on_the_fly default
        alm0 = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax, l in m:lmax; sc=1/(1+l)^2; alm0[l+1,m+1]= m==0 ? complex(sc) : complex(sc,0.5sc); end
        f = SHTnsKit.synthesis(cfg, alm0; real_output=true); @test all(isfinite, f)
        a = SHTnsKit.analysis(cfg, f); @test isapprox(a, alm0; rtol=1e-9, atol=1e-11)
    end
end

@testset "tables path round-trip lmax 256" begin
    lmax=256; cfg=create_gauss_config(lmax,lmax+2;nlon=2*lmax+1)
    SHTnsKit.prepare_plm_tables!(cfg)
    alm0=zeros(ComplexF64,lmax+1,lmax+1)
    for m in 0:lmax,l in m:lmax; sc=1/(1+l)^2; alm0[l+1,m+1]= m==0 ? complex(sc) : complex(sc,0.5sc); end
    f=SHTnsKit.synthesis(cfg,alm0;real_output=true); @test all(isfinite,f)
    a=SHTnsKit.analysis(cfg,f); @test isapprox(a,alm0;rtol=1e-9,atol=1e-11)
end

@testset "sphtor round-trip lmax 256 (OTF + tables)" begin
    lmax=256
    for usetbl in (false, true)
        cfg=create_gauss_config(lmax,lmax+2;nlon=2*lmax+1)
        usetbl && SHTnsKit.prepare_plm_tables!(cfg)
        S0=zeros(ComplexF64,lmax+1,lmax+1); T0=zeros(ComplexF64,lmax+1,lmax+1)
        for m in 0:lmax, l in max(1,m):lmax
            sc=1/(1+l)^2
            S0[l+1,m+1]=complex(sc, m==0 ? 0.0 : 0.5sc); T0[l+1,m+1]=complex(0.7sc, m==0 ? 0.0 : -0.3sc)
        end
        Vt,Vp = SHTnsKit.synthesis_sphtor(cfg,S0,T0;real_output=true)
        @test all(isfinite,Vt) && all(isfinite,Vp)
        S,T = SHTnsKit.analysis_sphtor(cfg,Vt,Vp)
        @test isapprox(S,S0;rtol=1e-8,atol=1e-10) && isapprox(T,T0;rtol=1e-8,atol=1e-10)
    end
end

@testset "packed + point/lat eval finite at lmax 256" begin
    lmax = 256
    cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

    # --- packed real round-trip (synthesis_packed / analysis_packed via core) ---
    Qlm = zeros(ComplexF64, cfg.nlm)
    for i in eachindex(Qlm); Qlm[i] = complex(1.0/(1+i), 0.0); end
    fp = synthesis_packed(cfg, Qlm)
    @test all(isfinite, fp)
    Qr = analysis_packed(cfg, fp)
    @test all(isfinite, Qr)

    # --- complex packed round-trip (synthesis_packed_cplx / analysis_packed_cplx) ---
    nlm_cplx = SHTnsKit.nlm_cplx_calc(lmax, lmax, 1)
    alm_cplx = zeros(ComplexF64, nlm_cplx)
    for i in eachindex(alm_cplx); alm_cplx[i] = complex(1.0/(1+i), -0.5/(1+i)); end
    zcplx = synthesis_packed_cplx(cfg, alm_cplx)
    @test all(isfinite, zcplx)
    alm_cplx2 = analysis_packed_cplx(cfg, zcplx)
    @test all(isfinite, alm_cplx2)
    @test isapprox(alm_cplx, alm_cplx2; rtol=1e-9, atol=1e-11)

    # --- SH_to_lat: real packed latitude strip ---
    vals_lat = SH_to_lat(cfg, Qlm, 0.5)
    @test all(isfinite, vals_lat)
    @test length(vals_lat) == cfg.nlon

    # --- synthesis_point_cplx: single-point evaluation ---
    sp = synthesis_point_cplx(cfg, alm_cplx, 0.5, 1.2)
    @test isfinite(sp)

    # --- SHqst_to_point: vector point evaluation ---
    Slm = zeros(ComplexF64, cfg.nlm)
    Tlm = zeros(ComplexF64, cfg.nlm)
    for i in eachindex(Slm); Slm[i] = complex(0.5/(1+i), 0.0); end
    for i in eachindex(Tlm); Tlm[i] = complex(0.0, 0.3/(1+i)); end
    vr, vt, vp = SHTnsKit.SHqst_to_point(cfg, Qlm, Slm, Tlm, 0.5, 1.2)
    @test isfinite(vr) && isfinite(vt) && isfinite(vp)

    # --- SHqst_to_lat: vector latitude strip ---
    Vr_lat, Vt_lat, Vp_lat = SHTnsKit.SHqst_to_lat(cfg, Qlm, Slm, Tlm, 0.5)
    @test all(isfinite, Vr_lat) && all(isfinite, Vt_lat) && all(isfinite, Vp_lat)

    # --- batch transforms round-trip ---
    nfields = 3
    alm_3d = zeros(ComplexF64, lmax+1, lmax+1, nfields)
    for k in 1:nfields, m in 0:lmax, l in m:lmax
        sc = 1/(1+l+k)^2
        alm_3d[l+1, m+1, k] = m==0 ? complex(sc) : complex(sc, 0.3sc)
    end
    fb = SHTnsKit.synthesis_batch(cfg, alm_3d; real_output=true)
    @test all(isfinite, fb)
    ab = SHTnsKit.analysis_batch(cfg, fb)
    @test all(isfinite, ab)
    @test isapprox(ab, alm_3d; rtol=1e-8, atol=1e-10)
end

@testset "planned sphtor + point/packed/truncated finite at lmax 256" begin
    lmax = 256
    cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

    # --- planned sphtor round-trip (was NaN before this task) ---
    S0 = zeros(ComplexF64, lmax+1, lmax+1); T0 = copy(S0)
    for m in 0:lmax, l in max(1,m):lmax
        sc = 1/(1+l)^2
        S0[l+1,m+1] = complex(sc, m==0 ? 0.0 : 0.5sc)
        T0[l+1,m+1] = complex(0.7sc, m==0 ? 0.0 : -0.3sc)
    end

    # synthesis_sphtor (non-planned) produces reference spatial fields
    Vt, Vp = SHTnsKit.synthesis_sphtor(cfg, S0, T0; real_output=true)
    @test all(isfinite, Vt) && all(isfinite, Vp)

    # planned analysis_sphtor! (was the confirmed-broken path)
    plan = SHTPlan(cfg)
    Slm = zeros(ComplexF64, lmax+1, lmax+1); Tlm = copy(Slm)
    SHTnsKit.analysis_sphtor!(plan, Slm, Tlm, Vt, Vp)
    @test all(isfinite, Slm) && all(isfinite, Tlm)
    @test isapprox(Slm, S0; rtol=1e-7, atol=1e-9) && isapprox(Tlm, T0; rtol=1e-7, atol=1e-9)

    # planned synthesis_sphtor!
    Vt2 = similar(Vt); Vp2 = similar(Vp)
    SHTnsKit.synthesis_sphtor!(plan, Vt2, Vp2, S0, T0)
    @test all(isfinite, Vt2) && all(isfinite, Vp2)
    @test isapprox(Vt2, Vt; rtol=1e-7, atol=1e-9) && isapprox(Vp2, Vp; rtol=1e-7, atol=1e-9)

    # --- scalar point evaluation (synthesis_point; transforms.jl) ---
    a0 = zeros(ComplexF64, lmax+1, lmax+1)
    for m in 0:lmax, l in m:lmax
        sc = 1/(1+l)^2
        a0[l+1,m+1] = m==0 ? complex(sc) : complex(sc, 0.5sc)
    end
    pt_val = SHTnsKit.synthesis_point(cfg, a0, 0.3, 0.7)
    @test isfinite(pt_val)

    # --- axisymmetric analysis/synthesis (transforms.jl): check finite ---
    # The axisym pair integrates over θ only (no φ factor), so analysis∘synthesis
    # returns alm / (2π); just verify finiteness here.
    f_ax = SHTnsKit.synthesis_axisym(cfg, a0[:, 1])
    @test all(isfinite, f_ax)
    ql_ax = SHTnsKit.analysis_axisym(cfg, real.(f_ax))
    @test all(isfinite, ql_ax)

    # --- analysis_axisym_l / synthesis_axisym_l (truncated, transforms.jl) ---
    ltr = lmax ÷ 2
    f_axl = SHTnsKit.synthesis_axisym_l(cfg, a0[:, 1], ltr)
    @test all(isfinite, f_axl)
    ql_axl = SHTnsKit.analysis_axisym_l(cfg, real.(f_axl), ltr)
    @test all(isfinite, ql_axl)

    # --- analysis_packed_ml / synthesis_packed_ml (per-m, transforms.jl) ---
    im_test = 3; ltr2 = lmax ÷ 2
    Ql_ml = a0[im_test+1:ltr2+1, im_test+1]  # coefficients for m=im_test, l=im_test:ltr2
    Vr_ml = SHTnsKit.synthesis_packed_ml(cfg, im_test, Ql_ml, ltr2)
    @test all(isfinite, Vr_ml)
    Ql_ml2 = SHTnsKit.analysis_packed_ml(cfg, im_test, Vr_ml, ltr2)
    @test all(isfinite, Ql_ml2)
    @test isapprox(Ql_ml2, Ql_ml; rtol=1e-7, atol=1e-9)
end
