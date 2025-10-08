using SHTnsKit
using CUDA

CUDA.functional() || error("CUDA device not available")

lmax = 32
cfg = create_gauss_config_gpu(lmax, lmax + 2; nlon=2*lmax + 1, device=SHTnsKit.GPU)

θ, φ = cfg.θ, cfg.φ
spatial = [sin(θ[i]) * cos(φ[j]) for i in eachindex(θ), j in eachindex(φ)]

alm = analysis(cfg, spatial)
recon = synthesis(cfg, alm; real_output=true)

println("GPU roundtrip error = ", maximum(abs.(spatial .- recon)))

destroy_config(cfg)
