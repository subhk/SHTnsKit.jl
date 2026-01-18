# Performance Guide

```@raw html
<div style="background: linear-gradient(135deg, #dc2626 0%, #f97316 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
    <h2 style="margin: 0 0 0.5rem 0; color: white; border: none;">Optimize Your Transforms</h2>
    <p style="margin: 0; opacity: 0.9;">Threading, memory management, and algorithm-level optimizations</p>
</div>
```

This guide provides comprehensive information on optimizing SHTnsKit.jl performance for various computational scenarios, including serial, parallel (MPI), and SIMD optimizations.

!!! tip "Quick Wins"
    - Pre-allocate arrays and reuse buffers
    - Use in-place operations (`analysis!`, `synthesis!`)
    - Set FFTW threads appropriately
    - For lmax > 64, consider GPU acceleration

## Understanding Performance Characteristics

### Transform Complexity

Spherical harmonic transforms have the following computational characteristics:
- Practical implementations: approximately O(L³) in maximum degree L
- Memory: O(L²) for spectral coefficients and spatial grid

### Performance Scaling

```julia
using SHTnsKit
using BenchmarkTools

function benchmark_transforms(lmax_values)
    results = []

    for lmax in lmax_values
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Create bandlimited test data
        spatial = zeros(cfg.nlat, cfg.nlon)
        for i in 1:cfg.nlat
            x = cfg.x[i]
            spatial[i, :] .= (3*x^2 - 1)/2  # Y_2^0
        end

        # Benchmark forward transform (synthesis)
        Alm = analysis(cfg, spatial)
        forward_time = @belapsed synthesis($cfg, $Alm)

        # Benchmark backward transform (analysis)
        backward_time = @belapsed analysis($cfg, $spatial)

        push!(results, (lmax=lmax, forward=forward_time, backward=backward_time))
        destroy_config(cfg)
    end

    return results
end

# Test scaling
lmax_range = [16, 32, 64, 128, 256]
results = benchmark_transforms(lmax_range)

for r in results
    println("lmax=$(r.lmax): forward=$(r.forward)s, backward=$(r.backward)s")
end
```

## Parallel Computing Performance

### MPI Parallelization

For large problems, MPI parallelization provides significant speedup:

```julia
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

# Configuration
lmax = 64
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create distributed array
pen = Pencil((nlat, nlon), MPI.COMM_WORLD)
fθφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

# Fill with test data
ranges = PencilArrays.range_local(pen)
for (i_local, i_global) in enumerate(ranges[1])
    x = cfg.x[i_global]
    for j in 1:length(ranges[2])
        fθφ[i_local, j] = (3*x^2 - 1)/2
    end
end

# Benchmark distributed transforms
function benchmark_parallel_performance()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    # Warm up
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)

    # Benchmark
    time_analysis = @elapsed begin
        for i in 1:50
            SHTnsKit.dist_analysis(cfg, fθφ)
        end
    end

    if rank == 0
        println("Parallel performance ($nprocs processes):")
        println("  Analysis: $(time_analysis/50*1000) ms per transform")
    end
end

benchmark_parallel_performance()
destroy_config(cfg)
MPI.Finalize()
```

## Threading Optimization

### Julia Threads and FFTW

SHTnsKit uses Julia `Threads.@threads` and FFTW's internal threads. Configure them for best results:

```julia
using SHTnsKit
using FFTW

# Check system capabilities
println("System threads: ", Sys.CPU_THREADS)
println("Julia threads: ", Threads.nthreads())

# Manual FFTW thread control
function benchmark_threading(lmax=64)
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)

    # Create bandlimited test data
    spatial = zeros(cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat
        x = cfg.x[i]
        spatial[i, :] .= (3*x^2 - 1)/2
    end

    thread_counts = [1, 2, 4, min(8, Sys.CPU_THREADS)]
    times = Float64[]

    for nthreads in thread_counts
        FFTW.set_num_threads(nthreads)
        time = @elapsed begin
            for i in 1:10
                analysis(cfg, spatial)
            end
        end
        push!(times, time)
        println("$nthreads FFTW threads: $(time/10*1000) ms per transform")
    end

    destroy_config(cfg)
end

benchmark_threading()
```

### Avoiding Oversubscription

```julia
# Prevent thread oversubscription with other libraries
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
ENV["FFTW_NUM_THREADS"] = "1"

# Keep FFTW threads modest to avoid contention
set_fft_threads(min(Sys.CPU_THREADS ÷ 2, 8))
```

## Memory Optimization

### Pre-allocation Strategies

```julia
using SHTnsKit

lmax = 64
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

# Method 1: Pre-allocate buffers for in-place operations
Alm_buffer = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
spatial_buffer = zeros(cfg.nlat, cfg.nlon)
fft_scratch = scratch_fft(cfg)

function process_many_fields_optimized(cfg, n_fields)
    results = Float64[]

    for i in 1:n_fields
        # Generate field data
        for j in 1:cfg.nlat
            x = cfg.x[j]
            spatial_buffer[j, :] .= x^2 + 0.1*sin(i)
        end

        # In-place transform (reuses fft_scratch)
        analysis!(cfg, Alm_buffer, spatial_buffer; fft_scratch=fft_scratch)

        # Process result
        energy = sum(abs2, Alm_buffer)
        push!(results, energy)
    end

    return results
end

# vs Method 2: Allocate every time (slower)
function process_many_fields_naive(cfg, n_fields)
    results = Float64[]

    for i in 1:n_fields
        spatial = zeros(cfg.nlat, cfg.nlon)
        for j in 1:cfg.nlat
            x = cfg.x[j]
            spatial[j, :] .= x^2 + 0.1*sin(i)
        end
        Alm = analysis(cfg, spatial)  # Allocates new array
        energy = sum(abs2, Alm)
        push!(results, energy)
    end

    return results
end

destroy_config(cfg)
```

### Memory Layout Optimization

```julia
# For batch processing, consider array-of-arrays vs matrix layout
using SHTnsKit

lmax = 32
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)
n_fields = 100

# Layout 1: Array of matrices (better for random access)
spectral_data_aoa = [zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1) for _ in 1:n_fields]
for arr in spectral_data_aoa
    arr[1,1] = 1.0
    arr[3,1] = 0.5
end

# Process with array of arrays
@time begin
    for i in 1:n_fields
        spatial = synthesis(cfg, spectral_data_aoa[i])
    end
end

destroy_config(cfg)
```

### Large Problem Memory Management

```julia
using SHTnsKit
using Statistics

function process_large_dataset(lmax=256, n_fields=1000)
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)

    # For very large problems, process in chunks
    chunk_size = 100
    n_chunks = div(n_fields, chunk_size)

    results = Float64[]

    # Pre-allocate buffers to reuse
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)

    for chunk in 1:n_chunks
        chunk_results = Float64[]

        for i in 1:chunk_size
            # Modify coefficients in place
            fill!(Alm, 0.0)
            Alm[1,1] = 1.0 + 0.01*i
            Alm[3,1] = 0.5

            spatial = synthesis(cfg, Alm)
            push!(chunk_results, mean(spatial))
        end

        append!(results, chunk_results)
        GC.gc()  # Force garbage collection between chunks
    end

    destroy_config(cfg)
    return results
end
```

## GPU Acceleration

For GPU-accelerated transforms, see the dedicated [GPU Guide](gpu.md). GPU acceleration provides 10-30× speedup for large problems (lmax > 64).

```julia
using SHTnsKit, CUDA

cfg = create_gauss_config(128, 130)
spatial = rand(cfg.nlat, cfg.nlon)

# GPU transforms
Alm = gpu_analysis(cfg, spatial)
recovered = gpu_synthesis(cfg, Alm)
```

!!! tip "When to Use GPU"
    GPU acceleration is most beneficial for **lmax ≥ 64**. For smaller problems, CPU is often faster due to data transfer overhead.

## Algorithm-Specific Optimizations

### Transform Direction Optimization

```julia
using SHTnsKit

lmax = 64
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

# Forward transforms (synthesis) are generally faster than backward (analysis)
# Plan your algorithm to minimize analysis operations

function optimize_transform_direction(cfg)
    # Create test coefficients
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Alm[1,1] = 1.0
    Alm[3,1] = 0.5

    # Create test spatial data
    spatial = zeros(cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat
        x = cfg.x[i]
        spatial[i, :] .= (3*x^2 - 1)/2
    end

    # Forward transform timing (synthesis)
    forward_time = @elapsed begin
        for i in 1:100
            synthesis(cfg, Alm)
        end
    end

    # Backward transform timing (analysis)
    backward_time = @elapsed begin
        for i in 1:100
            analysis(cfg, spatial)
        end
    end

    println("Synthesis: $(forward_time/100*1000) ms")
    println("Analysis: $(backward_time/100*1000) ms")
    println("Ratio: $(backward_time/forward_time)")
end

optimize_transform_direction(cfg)
destroy_config(cfg)
```

### Grid Type Selection

```julia
using SHTnsKit

function compare_grid_types(lmax=32)
    nlat = lmax + 2
    nlon = 2*lmax + 1

    # Gauss grids: optimal for accuracy
    cfg_gauss = create_gauss_config(lmax, nlat; nlon=nlon)

    # Regular grids: uniform spacing
    cfg_regular = create_regular_config(lmax, nlat; nlon=nlon)

    println("Grid Comparison (lmax=$lmax):")
    println("Gauss: $(cfg_gauss.nlat) × $(cfg_gauss.nlon) points")
    println("Regular: $(cfg_regular.nlat) × $(cfg_regular.nlon) points")

    # Create test coefficients
    Alm = zeros(ComplexF64, cfg_gauss.lmax+1, cfg_gauss.mmax+1)
    Alm[1,1] = 1.0
    Alm[3,1] = 0.5

    gauss_time = @elapsed begin
        for i in 1:50
            synthesis(cfg_gauss, Alm)
        end
    end

    regular_time = @elapsed begin
        for i in 1:50
            synthesis(cfg_regular, Alm)
        end
    end

    println("Gauss time: $(gauss_time/50*1000) ms")
    println("Regular time: $(regular_time/50*1000) ms")

    destroy_config(cfg_gauss)
    destroy_config(cfg_regular)
end

compare_grid_types()
```

## Vector Field Performance

```julia
using SHTnsKit

lmax = 48
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

# Vector transforms are more expensive than scalar
function benchmark_vector_vs_scalar(cfg)
    # Scalar coefficients
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Alm[1,1] = 1.0
    Alm[3,1] = 0.5

    # Scalar spatial field
    spatial_scalar = zeros(cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat
        x = cfg.x[i]
        spatial_scalar[i, :] .= (3*x^2 - 1)/2
    end

    # Vector coefficients
    Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Slm[2,1] = 1.0
    Tlm[3,2] = 0.5

    # Vector spatial fields
    Vθ, Vφ = SHsphtor_to_spat(cfg, Slm, Tlm)

    # Scalar benchmarks
    scalar_synth = @elapsed begin
        for i in 1:20
            synthesis(cfg, Alm)
        end
    end

    scalar_analysis = @elapsed begin
        for i in 1:20
            analysis(cfg, spatial_scalar)
        end
    end

    # Vector benchmarks
    vector_synth = @elapsed begin
        for i in 1:20
            SHsphtor_to_spat(cfg, Slm, Tlm)
        end
    end

    vector_analysis = @elapsed begin
        for i in 1:20
            spat_to_SHsphtor(cfg, Vθ, Vφ)
        end
    end

    println("Transform Performance Comparison:")
    println("Scalar synthesis: $(scalar_synth/20*1000) ms")
    println("Vector synthesis: $(vector_synth/20*1000) ms")
    println("Scalar analysis: $(scalar_analysis/20*1000) ms")
    println("Vector analysis: $(vector_analysis/20*1000) ms")
end

benchmark_vector_vs_scalar(cfg)
destroy_config(cfg)
```

<!-- Distributed/MPI performance guidance omitted for this package. -->

## Performance Monitoring and Profiling

### Built-in Benchmarking

```julia
using SHTnsKit
using Profile
using BenchmarkTools
using Statistics

lmax = 64
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

function profile_transforms(cfg)
    # Create test coefficients
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Alm[1,1] = 1.0
    Alm[3,1] = 0.5

    # Detailed benchmarking
    forward_bench = @benchmark synthesis($cfg, $Alm)
    println("Forward transform statistics:")
    println("  Median: $(median(forward_bench.times)/1e6) ms")
    println("  Mean: $(mean(forward_bench.times)/1e6) ms")

    # Memory allocation tracking
    spatial = synthesis(cfg, Alm)
    backward_bench = @benchmark analysis($cfg, $spatial)

    println("Backward transform statistics:")
    println("  Median: $(median(backward_bench.times)/1e6) ms")
    println("  Allocations: $(backward_bench.memory) bytes")
end

profile_transforms(cfg)

# Julia profiling
function profile_detailed(cfg)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Alm[1,1] = 1.0

    Profile.clear()
    @profile begin
        for i in 1:100
            synthesis(cfg, Alm)
        end
    end

    Profile.print()
end

destroy_config(cfg)
```

### Custom Performance Metrics

```julia
using SHTnsKit
using Statistics

function performance_report(lmax, n_runs=100)
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)

    # Create test coefficients
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Alm[1,1] = 1.0
    Alm[3,1] = 0.5

    # Warm up
    for i in 1:5
        synthesis(cfg, Alm)
    end

    # Collect metrics
    times = Float64[]

    for i in 1:n_runs
        time = @elapsed synthesis(cfg, Alm)
        push!(times, time)
    end

    # Statistics
    mean_time = mean(times)
    std_time = std(times)
    min_time = minimum(times)
    max_time = maximum(times)

    # Compute derived metrics
    operations_per_sec = 1.0 / mean_time
    points_per_sec = (cfg.nlat * cfg.nlon) / mean_time

    println("Performance Report (lmax=$lmax, $n_runs runs):")
    println("  Mean time: $(mean_time*1000) ms (±$(std_time*1000) ms)")
    println("  Min/Max: $(min_time*1000) ms / $(max_time*1000) ms")
    println("  Transforms/sec: $(round(operations_per_sec, digits=1))")
    println("  Points/sec: $(round(points_per_sec/1e6, digits=2)) M")

    destroy_config(cfg)
end

performance_report(32)
```

## Optimization Checklist

### Before Optimization
- [ ] Profile your code to identify bottlenecks
- [ ] Understand your problem's computational characteristics
- [ ] Measure baseline performance

### Threading Optimization  
- [ ] Set `OMP_NUM_THREADS` appropriately
- [ ] Use `set_optimal_threads()` for automatic tuning
- [ ] Disable threading in other libraries (BLAS, FFTW)
- [ ] Consider NUMA topology for large systems

### Memory Optimization
- [ ] Pre-allocate buffers for repeated operations
- [ ] Use in-place transforms when possible
- [ ] Process data in chunks for large datasets
- [ ] Monitor memory usage and fragmentation

### Algorithm Optimization
- [ ] Minimize backward transforms (analysis)
- [ ] Choose appropriate grid type (Gauss vs regular)
- [ ] Batch operations when possible
- [ ] Cache frequently used configurations

<!-- GPU optimization checklist removed -->
- [ ] Use appropriate batch sizes

### System-Level Optimization
- [ ] Use high-performance BLAS library
- [ ] Enable CPU optimizations (AVX, etc.)
- [ ] Consider process/thread affinity
- [ ] Monitor system resource utilization

### Performance Validation
- [ ] Compare with baseline measurements
- [ ] Verify numerical accuracy after optimization
- [ ] Test with realistic problem sizes
- [ ] Document performance characteristics

## Common Performance Pitfalls

1. **Thread Oversubscription**: Too many threads can hurt performance
2. **Memory Allocation**: Repeated allocation in inner loops
3. **Wrong Grid Type**: Regular grids when Gauss would suffice
4. **Unnecessary Transforms**: Computing both directions when only one needed
5. Performance pitfalls: array allocations in hot loops, oversubscription of threads

Following these guidelines will help you achieve optimal performance for your specific SHTnsKit.jl applications.
