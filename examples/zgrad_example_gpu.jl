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

using Zygote
cfg_cpu = create_gauss_config(16, 18; nlon=33)
field_cpu = copy(field)
energy_cpu(x) = energy_scalar(cfg_cpu, x)

grad_cpu = Zygote.gradient(energy_cpu, field_cpu)[1]
println("CPU gradient max = ", maximum(abs.(grad_cpu)))

L = sum
println(Zygote.gradient(L, grad_cpu)[1])

