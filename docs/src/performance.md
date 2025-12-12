# Performance Guide

This guide provides comprehensive information on optimizing SHTnsKit.jl performance for various computational scenarios, including serial, parallel (MPI), and SIMD optimizations.

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

### Performance Scaling by Problem Size

| Problem Size (nlm) | Serial | 4 Processes | 16 Processes | Expected Speedup |
|--------------------|--------|-------------|--------------|------------------|
| 1,000             | 5ms    | 4ms         | 5ms          | 1.3x             |
| 10,000            | 50ms   | 18ms        | 12ms         | 4.2x             |
| 100,000           | 500ms  | 140ms       | 65ms         | 7.7x             |
| 1,000,000         | 5.2s   | 1.8s        | 0.9s         | 14.2x            |

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

cfg = create_gauss_config(64, 64)

# Method 1: Allocate once, reuse many times
sh_buffer = allocate_spectral(cfg)
spatial_buffer = allocate_spatial(cfg)

function process_many_fields_optimized(field_generator, n_fields)
    results = Float64[]
    
    for i in 1:n_fields
        # Generate field data (application-specific)
        fill!(spatial_buffer, 0.0)
        field_data = field_generator(i)
        spatial_buffer .= field_data
        
        # In-place transform
        analyze!(cfg, spatial_buffer, sh_buffer)
        
        # Process result (example: compute energy)
        energy = sum(abs2, sh_buffer)
        push!(results, energy)
    end
    
    return results
end

# vs Method 2: Allocate every time (slower)
function process_many_fields_naive(field_generator, n_fields)
    results = Float64[]
    
    for i in 1:n_fields
        field_data = field_generator(i)
        sh = analyze(cfg, field_data)  # Allocates new array
        energy = sum(abs2, sh)
        push!(results, energy)
    end
    
    return results
end

destroy_config(cfg)
```

### Memory Layout Optimization

```julia
# For batch processing, consider array-of-arrays vs array layout
using SHTnsKit

cfg = create_gauss_config(32, 32)
n_fields = 1000

# Layout 1: Array of arrays (better for random access)
spectral_data_aoa = [rand(cfg.nlm) for _ in 1:n_fields]

# Layout 2: Single large array (better for streaming)
nlm = cfg.nlm
spectral_data_flat = rand(nlm, n_fields)

# Process with different layouts
@time begin
    for i in 1:n_fields
        spatial = synthesize(cfg, spectral_data_aoa[i])
    end
end

@time begin
    for i in 1:n_fields
        spatial = synthesize(cfg, @view spectral_data_flat[:, i])
    end
end

    destroy_config(cfg)
```

### Large Problem Memory Management

```julia
using SHTnsKit

function process_large_dataset(lmax=256, n_fields=10000)
    cfg = create_gauss_config(lmax, lmax)
    
    # For very large problems, process in chunks
    chunk_size = 100
    n_chunks = div(n_fields, chunk_size)
    
    results = Float64[]
    
    for chunk in 1:n_chunks
        # Process chunk
        chunk_results = Float64[]
        
        for i in 1:chunk_size
            # Generate field (don't store all at once)
            # Create bandlimited test coefficients (prevents high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
            spatial = synthesis(cfg, sh)
            
            # Compute result
            result = mean(spatial)
            push!(chunk_results, result)
        end
        
        # Store chunk results
        append!(results, chunk_results)
        
        # Force garbage collection between chunks
        GC.gc()
    end
    
    destroy_config(cfg)
    return results
end
```

## GPU Acceleration

This package is CPU‑focused and does not include GPU support.

## Algorithm-Specific Optimizations

### Transform Direction Optimization

```julia
using SHTnsKit

cfg = create_gauss_config(64, 64)

# Forward transforms (synthesis) are generally faster than backward (analysis)
# Plan your algorithm to minimize analysis operations

function optimize_transform_direction()
    # Create bandlimited test coefficients (prevents high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
    # Create bandlimited spatial data (smooth test function)
θ, φ = cfg.θ, cfg.φ
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    spatial[i,j] = 1.0 + 0.5 * cos(θ[i]) + 0.3 * sin(θ[i]) * cos(φ[j])
end
    
    # Forward transform timing
    forward_time = @elapsed begin
        for i in 1:100
            synthesis(cfg, sh)
        end
    end
    
    # Backward transform timing
    backward_time = @elapsed begin
        for i in 1:100
            analysis(cfg, spatial)
        end
    end
    
    println("Forward: $(forward_time/100)s")
    println("Backward: $(backward_time/100)s") 
    println("Ratio: $(backward_time/forward_time)")
end

optimize_transform_direction()
destroy_config(cfg)
```

### Grid Type Selection

```julia
using SHTnsKit

function compare_grid_types(lmax=32)
    # Gauss grids: optimal for accuracy, fewer points
    cfg_gauss = create_gauss_config(lmax, lmax)
    
    # Regular grids: more points, but uniform spacing
    cfg_regular = create_regular_config(lmax, lmax)
    
    println("Grid Comparison (lmax=$lmax):")
    println("Gauss: $(get_nlat(cfg_gauss)) × $(get_nphi(cfg_gauss)) = $(get_nlat(cfg_gauss)*get_nphi(cfg_gauss)) points")
    println("Regular: $(get_nlat(cfg_regular)) × $(get_nphi(cfg_regular)) = $(get_nlat(cfg_regular)*get_nphi(cfg_regular)) points")
    
    # Benchmark both
    sh = rand(get_nlm(cfg_gauss))  # Same spectral resolution
    
    gauss_time = @elapsed begin
        for i in 1:50
            synthesize(cfg_gauss, sh)
        end
    end
    
    regular_time = @elapsed begin  
        for i in 1:50
            synthesize(cfg_regular, sh)
        end
    end
    
    println("Gauss time: $(gauss_time/50)s")
    println("Regular time: $(regular_time/50)s")
    println("Gauss is $(regular_time/gauss_time)x faster")
    
    destroy_config(cfg_gauss)
    destroy_config(cfg_regular)
end

compare_grid_types()
```

## Vector Field Performance

```julia
using SHTnsKit

cfg = create_gauss_config(48, 48)

# Vector transforms are more expensive than scalar
function benchmark_vector_vs_scalar()
    # Scalar data
    sh_scalar = rand(cfg.nlm)
    # Create bandlimited spatial scalar field
    θ, φ = cfg.θ, cfg.φ
    spatial_scalar = zeros(cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat, j in 1:cfg.nlon
        spatial_scalar[i,j] = 1.0 + 0.4 * sin(2*θ[i]) * cos(φ[j])
    end
    
    # Vector data  
    S_lm = rand(cfg.nlm)
    T_lm = rand(cfg.nlm)
    # Create bandlimited vector field components
    θ, φ = cfg.θ, cfg.φ
    Vθ = zeros(cfg.nlat, cfg.nlon)
    Vφ = zeros(cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat, j in 1:cfg.nlon
        Vθ[i,j] = 0.8 * cos(θ[i]) * sin(φ[j])
        Vφ[i,j] = 0.6 * sin(θ[i]) * cos(2*φ[j])
    end
    
    # Scalar benchmarks
    scalar_synth = @elapsed begin
        for i in 1:20
            synthesize(cfg, sh_scalar)
        end
    end
    
    scalar_analysis = @elapsed begin
        for i in 1:20
            analyze(cfg, spatial_scalar)
        end
    end
    
    # Vector benchmarks
    vector_synth = @elapsed begin
        for i in 1:20
            synthesize_vector(cfg, S_lm, T_lm)
        end
    end
    
    vector_analysis = @elapsed begin
        for i in 1:20
            analyze_vector(cfg, Vθ, Vφ)
        end  
    end
    
    println("Transform Performance Comparison:")
    println("Scalar synthesis: $(scalar_synth/20)s")
    println("Vector synthesis: $(vector_synth/20)s ($(vector_synth/scalar_synth)x slower)")
    println("Scalar analysis: $(scalar_analysis/20)s")
    println("Vector analysis: $(vector_analysis/20)s ($(vector_analysis/scalar_analysis)x slower)")
end

benchmark_vector_vs_scalar()
destroy_config(cfg)
```

<!-- Distributed/MPI performance guidance omitted for this package. -->

## Performance Monitoring and Profiling

### Built-in Benchmarking

```julia
using SHTnsKit
using Profile
using BenchmarkTools

cfg = create_gauss_config(64, 64)

function profile_transforms()
    # Create bandlimited test coefficients (prevents high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
    
    # Detailed benchmarking
    forward_bench = @benchmark synthesize($cfg, $sh)
    println("Forward transform statistics:")
    println("  Median: $(median(forward_bench.times))ns")
    println("  Mean: $(mean(forward_bench.times))ns") 
    println("  Std: $(std(forward_bench.times))ns")
    
    # Memory allocation tracking
    spatial = synthesis(cfg, sh)
    backward_bench = @benchmark analyze($cfg, $spatial)
    
    println("Backward transform statistics:")
    println("  Median: $(median(backward_bench.times))ns")
    println("  Allocations: $(backward_bench.memory) bytes")
end

profile_transforms()

# Julia profiling
function profile_detailed()
    # Create bandlimited test coefficients (prevents high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
    
    Profile.clear()
    @profile begin
        for i in 1:100
            spatial = synthesis(cfg, sh)
        end
    end
    
    Profile.print()
end

destroy_config(cfg)
```

### Custom Performance Metrics

```julia
using SHTnsKit

function performance_report(cfg, n_runs=100)
    # Warm up
    sh_test = rand(cfg.nlm)
    for i in 1:5
        synthesize(cfg, sh_test)
    end
    
    # Collect metrics
    times = Float64[]
    
    for i in 1:n_runs
        # Create bandlimited test coefficients (prevents high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
        
        time = @elapsed begin
            spatial = synthesis(cfg, sh)
        end
        
        push!(times, time)
    end
    
    # Statistics
    mean_time = mean(times)
    std_time = std(times)
    min_time = minimum(times)
    max_time = maximum(times)
    
    # Compute derived metrics
    lmax = get_lmax(cfg)
    operations_per_sec = 1.0 / mean_time
    points_per_sec = (cfg.nlat * cfg.nlon) / mean_time
    
    println("Performance Report (lmax=$lmax, $n_runs runs):")
    println("  Mean time: $(mean_time*1000)ms (±$(std_time*1000)ms)")
    println("  Min/Max: $(min_time*1000)ms / $(max_time*1000)ms")
    println("  Transforms/sec: $(operations_per_sec)")
    println("  Points/sec: $(points_per_sec)")
    println("  Grid efficiency: $(cfg.nlm/(cfg.nlat*cfg.nlon))")
end

cfg = create_gauss_config(32, 32)
performance_report(cfg)
destroy_config(cfg)
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
