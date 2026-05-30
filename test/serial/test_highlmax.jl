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
