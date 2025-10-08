#!/usr/bin/env julia

using SHTnsKit
using BenchmarkTools

try
    using CUDA
catch
    @warn "CUDA.jl not available; install CUDA.jl to run GPU benchmarks"
    exit(1)
end

CUDA.functional() || error("CUDA device not available")

function build_test_field(cfg)
    θ, φ = cfg.θ, cfg.φ
    return [sin(θ[i]) * cos(φ[j]) + 0.3 * sin(2θ[i]) * sin(φ[j]) for i in eachindex(θ), j in eachindex(φ)]
end

function run_benchmarks(lmax::Int=64)
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg_cpu = create_gauss_config(lmax, nlat; nlon=nlon)
    cfg_gpu = create_gauss_config_gpu(lmax, nlat; nlon=nlon, device=SHTnsKit.GPU)

    spatial_cpu = build_test_field(cfg_cpu)

    println("=== Analysis Benchmark ===")
    bench_cpu = @benchmark analysis($cfg_cpu, $spatial_cpu)
    println("CPU analysis:")
    println(bench_cpu)

    bench_gpu = @benchmark analysis($cfg_gpu, $spatial_cpu)
    println("GPU analysis:")
    println(bench_gpu)

    alm_cpu = analysis(cfg_cpu, spatial_cpu)
    alm_gpu = analysis(cfg_gpu, spatial_cpu)

    println("=== Synthesis Benchmark ===")
    bench_cpu_syn = @benchmark synthesis($cfg_cpu, $alm_cpu; real_output=true)
    println("CPU synthesis:")
    println(bench_cpu_syn)

    bench_gpu_syn = @benchmark synthesis($cfg_gpu, $alm_gpu; real_output=true)
    println("GPU synthesis:")
    println(bench_gpu_syn)

    destroy_config(cfg_cpu)
    destroy_config(cfg_gpu)
end

run_benchmarks(parse(Int, get(ENV, "SHTNSKIT_LMAX", "64")))
