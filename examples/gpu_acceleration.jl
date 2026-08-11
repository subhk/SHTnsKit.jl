#!/usr/bin/env julia

"""
Single-GPU spherical-harmonic transform example.

Install CUDA, GPUArrays, GPUArraysCore, and KernelAbstractions before running:

    julia --project=. examples/gpu_acceleration.jl
"""

using SHTnsKit
using CUDA, GPUArrays, GPUArraysCore, KernelAbstractions
using Printf

CUDA.functional() || error("CUDA is not functional on this system")

lmax = 64
nlat = lmax + 2
nlon = 2lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon)

field = [
    1 + 0.5 * (3cos(θ)^2 - 1) + 0.3sin(θ) * cos(φ)
    for θ in cfg.θ, φ in cfg.φ
]

cpu_coeffs = analysis(cfg, field)
cpu_roundtrip = synthesis(cfg, cpu_coeffs; real_output=true)
@printf "CPU roundtrip error: %.3e\n" maximum(abs, field - cpu_roundtrip)

set_device!(GPU())
gpu_coeffs = gpu_analysis(cfg, field; device=GPU())
gpu_roundtrip = gpu_synthesis(cfg, gpu_coeffs; device=GPU(), real_output=true)
@printf "GPU roundtrip error: %.3e\n" maximum(abs, field - gpu_roundtrip)
@printf "CPU/GPU coefficient difference: %.3e\n" maximum(abs, cpu_coeffs - Array(gpu_coeffs))

println("Active device: ", get_device())
println("Available CUDA devices: ", get_available_gpus())
println("CUDA memory: ", gpu_memory_info())

set_device!(CPU())
