using SHTnsKit
using Zygote
using CUDA

CUDA.functional() || error("CUDA device not available")

lmax = 16
cfg = create_gauss_config_gpu(lmax, lmax + 2; nlon=2*lmax + 1, device=SHTnsKit.GPU)

θ, φ = cfg.θ, cfg.φ
spatial = [sin(θ[i]) * cos(φ[j]) for i in eachindex(θ), j in eachindex(φ)]

loss_field(x) = SHTnsKit.energy_scalar(cfg, x)

f = copy(spatial)
gpu_grad = Zygote.gradient(loss_field, f)[1]

println("Gradient norm = ", norm(gpu_grad))

destroy_config(cfg)
