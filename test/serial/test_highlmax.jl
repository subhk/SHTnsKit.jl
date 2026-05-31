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
