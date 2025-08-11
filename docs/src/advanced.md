# Advanced Usage Patterns

This guide covers sophisticated usage patterns and advanced techniques for experienced SHTnsKit.jl users.

## Advanced Configuration Management

### Configuration Factories

```julia
using SHTnsKit

# Create a factory for consistent configuration creation
struct SHTnsConfigFactory
    default_lmax::Int
    default_flags::UInt32
    cache::Dict{Tuple{Int,Int}, SHTnsConfig}
end

function SHTnsConfigFactory(lmax::Int=64)
    flags = SHTnsFlags.SHT_GAUSS | SHTnsFlags.SHT_REAL_NORM
    SHTnsConfigFactory(lmax, flags, Dict())
end

function get_config(factory::SHTnsConfigFactory, lmax::Int, mmax::Int)
    key = (lmax, mmax)
    if !haskey(factory.cache, key)
        factory.cache[key] = create_config(lmax, mmax, 
                                          2*lmax + 1, 
                                          factory.default_flags)
        set_grid(factory.cache[key], lmax+1, 2*mmax+1, SHTnsFlags.SHT_GAUSS)
    end
    return factory.cache[key]
end

function cleanup!(factory::SHTnsConfigFactory)
    for cfg in values(factory.cache)
        free_config(cfg)
    end
    empty!(factory.cache)
end

# Usage
factory = SHTnsConfigFactory()
cfg32 = get_config(factory, 32, 32)
cfg64 = get_config(factory, 64, 64)
# ... use configurations ...
cleanup!(factory)
```

### Dynamic Resolution Adaptation

```julia
using SHTnsKit

mutable struct AdaptiveSpectralTransform
    current_lmax::Int
    max_lmax::Int
    configs::Dict{Int, SHTnsConfig}
    tolerance::Float64
end

function AdaptiveSpectralTransform(max_lmax::Int, tolerance::Float64=1e-10)
    AdaptiveSpectralTransform(16, max_lmax, Dict{Int, SHTnsConfig}(), tolerance)
end

function get_config!(transform::AdaptiveSpectralTransform, lmax::Int)
    if !haskey(transform.configs, lmax)
        transform.configs[lmax] = create_gauss_config(lmax, lmax)
    end
    return transform.configs[lmax]
end

function adaptive_analyze(transform::AdaptiveSpectralTransform, field::Matrix{Float64})
    lmax = transform.current_lmax
    
    while lmax <= transform.max_lmax
        cfg = get_config!(transform, lmax)
        
        # Interpolate field to current resolution if needed
        field_resized = resize_spatial_field(field, cfg)
        
        # Analyze
        sh = analyze(cfg, field_resized)
        
        # Check convergence by looking at high-degree coefficients
        high_degree_power = sum(abs2, sh[end-min(10, div(length(sh), 4)):end])
        total_power = sum(abs2, sh)
        
        if high_degree_power / total_power < transform.tolerance
            transform.current_lmax = lmax
            return sh, lmax
        end
        
        lmax = min(lmax * 2, transform.max_lmax)
    end
    
    # Maximum resolution reached
    cfg = get_config!(transform, transform.max_lmax)
    field_resized = resize_spatial_field(field, cfg)
    sh = analyze(cfg, field_resized)
    transform.current_lmax = transform.max_lmax
    
    return sh, transform.max_lmax
end

function cleanup!(transform::AdaptiveSpectralTransform)
    for cfg in values(transform.configs)
        free_config(cfg)
    end
    empty!(transform.configs)
end
```

## Spectral Domain Operations

### Custom Spectral Filtering

```julia
using SHTnsKit

function create_spectral_filter(lmax::Int; 
                               low_pass::Union{Int,Nothing}=nothing,
                               high_pass::Union{Int,Nothing}=nothing,
                               band_pass::Union{Tuple{Int,Int},Nothing}=nothing)
    
    filter = ones(Float64, (lmax+1)*(lmax+2)÷2)
    
    cfg_temp = create_gauss_config(lmax, lmax)
    
    for i in 1:length(filter)
        l, m = get_lm_from_index(cfg_temp, i)
        
        if low_pass !== nothing && l > low_pass
            filter[i] = 0.0
        elseif high_pass !== nothing && l < high_pass
            filter[i] = 0.0
        elseif band_pass !== nothing
            l_min, l_max = band_pass
            if l < l_min || l > l_max
                filter[i] = 0.0
            end
        end
    end
    
    free_config(cfg_temp)
    return filter
end

function apply_spectral_filter!(sh::Vector{Float64}, filter::Vector{Float64})
    @assert length(sh) == length(filter) "Filter size mismatch"
    sh .*= filter
    return sh
end

# Example: Smooth a noisy field
cfg = create_gauss_config(64, 64)
noisy_field = rand(get_nlat(cfg), get_nphi(cfg))

# Low-pass filter (keep only l ≤ 20)
sh = analyze(cfg, noisy_field)
lowpass_filter = create_spectral_filter(64, low_pass=20)
apply_spectral_filter!(sh, lowpass_filter)
smooth_field = synthesize(cfg, sh)

free_config(cfg)
```

### Spectral Derivative Operations

```julia
using SHTnsKit

function spectral_laplacian(cfg::SHTnsConfig, sh::Vector{Float64})
    # ∇²f has spectral coefficients: -l(l+1) * f_lm
    laplacian_sh = copy(sh)
    
    for i in 1:length(sh)
        l, m = get_lm_from_index(cfg, i)
        laplacian_sh[i] *= -l * (l + 1)
    end
    
    return laplacian_sh
end

function spectral_horizontal_gradient(cfg::SHTnsConfig, sh::Vector{Float64})
    # Returns (∂f/∂θ, ∂f/∂φ/sin(θ)) in spectral domain
    # These are vector field components
    
    # This is complex - need to compute derivatives of Y_l^m
    # Implementation depends on SHTns internal representation
    # For now, use spatial domain computation
    
    spatial = synthesize(cfg, sh)
    θ, φ = get_coordinates(cfg)
    
    # Finite differences (not optimal, but illustrative)
    dθ = θ[2,1] - θ[1,1]
    dφ = φ[1,2] - φ[1,1]
    
    ∂f_∂θ = similar(spatial)
    ∂f_∂φ = similar(spatial)
    
    # Central differences
    ∂f_∂θ[2:end-1, :] = (spatial[3:end, :] - spatial[1:end-2, :]) / (2*dθ)
    ∂f_∂φ[:, 2:end-1] = (spatial[:, 3:end] - spatial[:, 1:end-2]) / (2*dφ)
    
    # Handle boundaries (simplified)
    ∂f_∂θ[[1,end], :] = ∂f_∂θ[[2,end-1], :]
    ∂f_∂φ[:, [1,end]] = ∂f_∂φ[:, [2,end-1]]
    
    return ∂f_∂θ, ∂f_∂φ
end

# Example: Compute and analyze gradients
cfg = create_gauss_config(32, 32)
θ, φ = get_coordinates(cfg)
test_field = @. sin(3θ) * cos(2φ)

sh = analyze(cfg, test_field)
∂f_∂θ, ∂f_∂φ = spectral_horizontal_gradient(cfg, sh)

println("Gradient magnitudes:")
println("  ∂f/∂θ: ", extrema(∂f_∂θ))
println("  ∂f/∂φ: ", extrema(∂f_∂φ))

free_config(cfg)
```

## Multi-Field Processing Patterns

### Coherent Transform Pipeline

```julia
using SHTnsKit

struct SpectralPipeline
    cfg::SHTnsConfig
    stages::Vector{Function}
    buffers::Dict{Symbol, Any}
end

function SpectralPipeline(lmax::Int, mmax::Int)
    cfg = create_gauss_config(lmax, mmax)
    stages = Function[]
    buffers = Dict{Symbol, Any}()
    
    # Pre-allocate common buffers
    buffers[:sh_temp] = allocate_spectral(cfg)
    buffers[:spatial_temp] = allocate_spatial(cfg)
    
    SpectralPipeline(cfg, stages, buffers)
end

function add_stage!(pipeline::SpectralPipeline, stage_func::Function)
    push!(pipeline.stages, stage_func)
end

function process(pipeline::SpectralPipeline, input_field::Matrix{Float64})
    # Initial analysis
    analyze!(pipeline.cfg, input_field, pipeline.buffers[:sh_temp])
    
    # Apply all stages
    for stage in pipeline.stages
        stage(pipeline.cfg, pipeline.buffers)
    end
    
    # Final synthesis
    synthesize!(pipeline.cfg, pipeline.buffers[:sh_temp], pipeline.buffers[:spatial_temp])
    
    return copy(pipeline.buffers[:spatial_temp])
end

function cleanup!(pipeline::SpectralPipeline)
    free_config(pipeline.cfg)
end

# Example pipeline: smooth -> amplify low modes -> threshold
pipeline = SpectralPipeline(32, 32)

# Stage 1: Low-pass filter
add_stage!(pipeline, function(cfg, buffers)
    sh = buffers[:sh_temp]
    for i in 1:length(sh)
        l, m = get_lm_from_index(cfg, i)
        if l > 16
            sh[i] = 0.0
        end
    end
end)

# Stage 2: Amplify low modes
add_stage!(pipeline, function(cfg, buffers)
    sh = buffers[:sh_temp]
    for i in 1:length(sh)
        l, m = get_lm_from_index(cfg, i)
        if l <= 8
            sh[i] *= 2.0
        end
    end
end)

# Process data
test_field = rand(get_nlat(pipeline.cfg), get_nphi(pipeline.cfg))
result = process(pipeline, test_field)

cleanup!(pipeline)
```

### Batch Transform Manager

```julia
using SHTnsKit
using Base.Threads

struct BatchTransformManager
    configs::Dict{Int, SHTnsConfig}
    thread_buffers::Vector{Dict{Symbol, Any}}
    max_lmax::Int
end

function BatchTransformManager(max_lmax::Int=128)
    configs = Dict{Int, SHTnsConfig}()
    thread_buffers = [Dict{Symbol, Any}() for _ in 1:nthreads()]
    BatchTransformManager(configs, thread_buffers, max_lmax)
end

function get_config!(manager::BatchTransformManager, lmax::Int)
    if !haskey(manager.configs, lmax)
        manager.configs[lmax] = create_gauss_config(lmax, lmax)
    end
    return manager.configs[lmax]
end

function get_buffers!(manager::BatchTransformManager, thread_id::Int, lmax::Int)
    buffers = manager.thread_buffers[thread_id]
    key = Symbol("buffers_$lmax")
    
    if !haskey(buffers, key)
        cfg = get_config!(manager, lmax)
        buffers[key] = Dict(
            :sh => allocate_spectral(cfg),
            :spatial => allocate_spatial(cfg)
        )
    end
    
    return buffers[key]
end

function batch_process(manager::BatchTransformManager, 
                      fields::Vector{Matrix{Float64}}, 
                      lmax::Int,
                      process_func::Function)
    
    results = Vector{Any}(undef, length(fields))
    cfg = get_config!(manager, lmax)
    
    @threads for i in 1:length(fields)
        thread_id = threadid()
        buffers = get_buffers!(manager, thread_id, lmax)
        
        # Resize field if necessary
        field = fields[i]
        if size(field) != (get_nlat(cfg), get_nphi(cfg))
            field = resize_spatial_field(field, cfg)
        end
        
        # Transform
        analyze!(cfg, field, buffers[:sh])
        
        # Process in spectral domain
        result_sh = process_func(cfg, buffers[:sh])
        
        # Transform back
        synthesize!(cfg, result_sh, buffers[:spatial])
        
        results[i] = copy(buffers[:spatial])
    end
    
    return results
end

function cleanup!(manager::BatchTransformManager)
    for cfg in values(manager.configs)
        free_config(cfg)
    end
    empty!(manager.configs)
end

# Example: Batch low-pass filtering
manager = BatchTransformManager()

# Generate test data
test_fields = [rand(65, 129) for _ in 1:100]

# Process function: low-pass filter
function lowpass_filter(cfg, sh)
    result = copy(sh)
    for i in 1:length(result)
        l, m = get_lm_from_index(cfg, i)
        if l > 20
            result[i] = 0.0
        end
    end
    return result
end

# Process batch
filtered_fields = batch_process(manager, test_fields, 64, lowpass_filter)

println("Processed $(length(filtered_fields)) fields")

cleanup!(manager)
```

## Advanced Vector Field Analysis

### Helmholtz Decomposition

```julia
using SHTnsKit

function helmholtz_decomposition(cfg::SHTnsConfig, u::Matrix{Float64}, v::Matrix{Float64})
    # Decompose vector field into rotational and divergent parts
    # u = u_rot + u_div, v = v_rot + v_div
    
    # Get spheroidal and toroidal components
    S_lm, T_lm = analyze_vector(cfg, u, v)
    
    # Rotational part (from toroidal component)
    u_rot, v_rot = synthesize_vector(cfg, zeros(length(S_lm)), T_lm)
    
    # Divergent part (from spheroidal component)  
    u_div, v_div = synthesize_vector(cfg, S_lm, zeros(length(T_lm)))
    
    return (u_rot, v_rot), (u_div, v_div), (S_lm, T_lm)
end

function vector_field_properties(cfg::SHTnsConfig, u::Matrix{Float64}, v::Matrix{Float64})
    # Compute various properties of vector field
    
    (u_rot, v_rot), (u_div, v_div), (S_lm, T_lm) = helmholtz_decomposition(cfg, u, v)
    
    # Energy in each component
    rot_energy = sum(u_rot.^2 + v_rot.^2)
    div_energy = sum(u_div.^2 + v_div.^2)
    total_energy = sum(u.^2 + v.^2)
    
    # Spectral energies
    spheroidal_energy = sum(abs2, S_lm)
    toroidal_energy = sum(abs2, T_lm)
    
    return Dict(
        :rotational_fraction => rot_energy / total_energy,
        :divergent_fraction => div_energy / total_energy,
        :spheroidal_energy => spheroidal_energy,
        :toroidal_energy => toroidal_energy,
        :total_spectral_energy => spheroidal_energy + toroidal_energy
    )
end

# Example analysis
cfg = create_gauss_config(48, 48)
θ, φ = get_coordinates(cfg)

# Create test vector field with known properties
u = @. 10 * sin(2θ) * cos(φ)    # Mostly divergent
v = @. 5 * cos(θ) * sin(2φ)     # Mixed

properties = vector_field_properties(cfg, u, v)

println("Vector Field Analysis:")
for (key, value) in properties
    println("  $key: $value")
end

free_config(cfg)
```

### Enstrophy and Energy Cascade Analysis

```julia
using SHTnsKit

function compute_enstrophy_spectrum(cfg::SHTnsConfig, u::Matrix{Float64}, v::Matrix{Float64})
    # Enstrophy Z(l) = l(l+1) * |ω_l|² where ω is vorticity
    
    # Get toroidal component (related to vorticity)
    S_lm, T_lm = analyze_vector(cfg, u, v)
    
    # Compute enstrophy per degree
    lmax = get_lmax(cfg)
    enstrophy = zeros(lmax + 1)
    
    for i in 1:length(T_lm)
        l, m = get_lm_from_index(cfg, i)
        enstrophy[l + 1] += l * (l + 1) * abs2(T_lm[i])
    end
    
    return enstrophy
end

function compute_energy_spectrum(cfg::SHTnsConfig, u::Matrix{Float64}, v::Matrix{Float64})
    # Kinetic energy E(l) = |u_l|²
    
    S_lm, T_lm = analyze_vector(cfg, u, v)
    
    lmax = get_lmax(cfg)
    energy = zeros(lmax + 1)
    
    for i in 1:length(S_lm)
        l, m = get_lm_from_index(cfg, i)
        energy[l + 1] += abs2(S_lm[i]) + abs2(T_lm[i])
    end
    
    return energy
end

function analyze_turbulent_cascade(cfg::SHTnsConfig, 
                                  velocity_fields::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}})
    # Analyze energy cascade in turbulent flow
    
    n_snapshots = length(velocity_fields)
    lmax = get_lmax(cfg)
    
    mean_energy = zeros(lmax + 1)
    mean_enstrophy = zeros(lmax + 1)
    
    for (u, v) in velocity_fields
        energy = compute_energy_spectrum(cfg, u, v)
        enstrophy = compute_enstrophy_spectrum(cfg, u, v)
        
        mean_energy .+= energy
        mean_enstrophy .+= enstrophy
    end
    
    mean_energy ./= n_snapshots
    mean_enstrophy ./= n_snapshots
    
    # Find inertial range (power law behavior)
    degrees = 1:lmax
    
    return Dict(
        :degrees => degrees,
        :energy_spectrum => mean_energy[2:end],  # Skip l=0
        :enstrophy_spectrum => mean_enstrophy[2:end],
        :energy_slope => fit_power_law_slope(degrees[5:end÷2], mean_energy[6:end÷2+1]),
        :enstrophy_slope => fit_power_law_slope(degrees[5:end÷2], mean_enstrophy[6:end÷2+1])
    )
end

function fit_power_law_slope(x, y)
    # Simple linear fit in log-log space
    log_x = log.(x)
    log_y = log.(y[y .> 0])  # Avoid log(0)
    
    if length(log_y) < 2
        return NaN
    end
    
    # Linear regression
    n = length(log_x)
    sum_x = sum(log_x)
    sum_y = sum(log_y[1:length(log_x)])
    sum_xy = sum(log_x .* log_y[1:length(log_x)])
    sum_x2 = sum(log_x.^2)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
    return slope
end
```

## Temporal Evolution and Time Series

### Spectral Time Series Analysis

```julia
using SHTnsKit
using FFTW

struct SpectralTimeSeries
    cfg::SHTnsConfig
    time_series::Vector{Vector{Float64}}  # Each element is sh coefficients
    times::Vector{Float64}
    lmax::Int
end

function SpectralTimeSeries(cfg::SHTnsConfig)
    SpectralTimeSeries(cfg, Vector{Float64}[], Float64[], get_lmax(cfg))
end

function add_snapshot!(sts::SpectralTimeSeries, field::Matrix{Float64}, time::Float64)
    sh = analyze(sts.cfg, field)
    push!(sts.time_series, sh)
    push!(sts.times, time)
end

function temporal_power_spectrum(sts::SpectralTimeSeries, l::Int, m::Int)
    # Get time evolution of specific (l,m) mode
    idx = get_index(sts.cfg, l, m)
    
    mode_evolution = [sh[idx] for sh in sts.time_series]
    
    # Compute temporal Fourier transform
    fft_result = fft(mode_evolution)
    power_spectrum = abs2.(fft_result)
    
    # Frequency axis
    dt = length(sts.times) > 1 ? sts.times[2] - sts.times[1] : 1.0
    frequencies = fftfreq(length(mode_evolution), 1/dt)
    
    return frequencies, power_spectrum
end

function mode_correlation_matrix(sts::SpectralTimeSeries)
    # Compute correlation between different spherical harmonic modes
    
    n_modes = length(sts.time_series[1])
    n_times = length(sts.time_series)
    
    # Create matrix: modes × time
    mode_matrix = zeros(n_modes, n_times)
    for (i, sh) in enumerate(sts.time_series)
        mode_matrix[:, i] = sh
    end
    
    # Compute correlation matrix
    correlation_matrix = cor(mode_matrix')
    
    return correlation_matrix
end

function dominant_mode_evolution(sts::SpectralTimeSeries, n_modes::Int=10)
    # Find most energetic modes and track their evolution
    
    # Compute mean energy per mode
    mean_energies = zeros(length(sts.time_series[1]))
    for sh in sts.time_series
        mean_energies .+= abs2.(sh)
    end
    mean_energies ./= length(sts.time_series)
    
    # Find dominant modes
    dominant_indices = sortperm(mean_energies, rev=true)[1:n_modes]
    
    # Extract evolution
    evolutions = []
    for idx in dominant_indices
        l, m = get_lm_from_index(sts.cfg, idx)
        evolution = [sh[idx] for sh in sts.time_series]
        push!(evolutions, (l=l, m=m, evolution=evolution, mean_energy=mean_energies[idx]))
    end
    
    return evolutions
end

# Example usage
cfg = create_gauss_config(32, 32)
sts = SpectralTimeSeries(cfg)

# Generate synthetic time series (e.g., decaying turbulence)
for t in 0:0.1:10.0
    # Synthetic field with time evolution
    θ, φ = get_coordinates(cfg)
    decay_factor = exp(-0.1 * t)
    field = decay_factor * (
        sin(3θ) .* cos(2φ) +
        0.5 * sin(5θ) .* cos(4φ) * cos(0.5π * t) +
        randn(size(θ)...) * 0.1
    )
    
    add_snapshot!(sts, field, t)
end

# Analysis
dominant_modes = dominant_mode_evolution(sts, 5)
for mode in dominant_modes
    println("Mode l=$(mode.l), m=$(mode.m): mean energy = $(mode.mean_energy)")
end

# Temporal spectrum of dominant mode
if length(dominant_modes) > 0
    l, m = dominant_modes[1].l, dominant_modes[1].m
    freqs, power = temporal_power_spectrum(sts, l, m)
    println("Temporal spectrum computed for mode ($l, $m)")
end

free_config(cfg)
```

## Custom Interpolation and Remapping

### Adaptive Mesh Refinement Interface

```julia
using SHTnsKit

struct AdaptiveMesh
    base_cfg::SHTnsConfig
    refined_regions::Vector{Dict{Symbol, Any}}
    global_field::Union{Vector{Float64}, Nothing}
end

function AdaptiveMesh(base_lmax::Int)
    base_cfg = create_gauss_config(base_lmax, base_lmax)
    AdaptiveMesh(base_cfg, Dict{Symbol, Any}[], nothing)
end

function add_refined_region!(mesh::AdaptiveMesh, 
                           θ_center::Float64, φ_center::Float64, 
                           radius::Float64, refinement_lmax::Int)
    
    refined_cfg = create_gauss_config(refinement_lmax, refinement_lmax)
    
    region = Dict(
        :center => (θ_center, φ_center),
        :radius => radius,
        :cfg => refined_cfg,
        :lmax => refinement_lmax,
        :local_field => nothing
    )
    
    push!(mesh.refined_regions, region)
end

function interpolate_to_refined_region!(mesh::AdaptiveMesh, region_idx::Int)
    if mesh.global_field === nothing
        error("No global field set")
    end
    
    region = mesh.refined_regions[region_idx]
    base_spatial = synthesize(mesh.base_cfg, mesh.global_field)
    
    # Extract region from global field (simplified interpolation)
    θ_global, φ_global = get_coordinates(mesh.base_cfg)
    θ_local, φ_local = get_coordinates(region[:cfg])
    
    # Simple nearest-neighbor interpolation (in practice, use proper interpolation)
    local_spatial = zeros(size(θ_local))
    
    for i in 1:size(θ_local, 1), j in 1:size(θ_local, 2)
        # Find nearest point in global grid
        distances = (θ_global .- θ_local[i,j]).^2 + (φ_global .- φ_local[i,j]).^2
        min_idx = argmin(distances)
        local_spatial[i,j] = base_spatial[min_idx]
    end
    
    # Analyze to get local spectral representation
    region[:local_field] = analyze(region[:cfg], local_spatial)
end

function project_refined_to_global!(mesh::AdaptiveMesh, region_idx::Int)
    region = mesh.refined_regions[region_idx]
    
    if region[:local_field] === nothing
        error("No refined field in region $region_idx")
    end
    
    # Convert refined solution back to global grid
    local_spatial = synthesize(region[:cfg], region[:local_field])
    
    # Project onto global spectral representation
    # This requires careful handling of overlapping regions
    
    global_spatial = synthesize(mesh.base_cfg, mesh.global_field)
    
    # Weighted blending (simplified)
    θ_center, φ_center = region[:center]
    radius = region[:radius]
    
    θ_global, φ_global = get_coordinates(mesh.base_cfg)
    θ_local, φ_local = get_coordinates(region[:cfg])
    
    # Apply refined solution in the local region
    # (Proper implementation would use overlap integrals)
    
    mesh.global_field = analyze(mesh.base_cfg, global_spatial)
end

function cleanup!(mesh::AdaptiveMesh)
    free_config(mesh.base_cfg)
    for region in mesh.refined_regions
        free_config(region[:cfg])
    end
end
```

## Memory-Mapped Large Dataset Processing

```julia
using SHTnsKit
using Mmap

struct MemoryMappedSpectralData
    file_path::String
    cfg::SHTnsConfig
    n_snapshots::Int
    nlm::Int
    mmap_array::Array{Float64, 2}  # nlm × n_snapshots
end

function create_mmap_spectral_data(file_path::String, cfg::SHTnsConfig, n_snapshots::Int)
    nlm = get_nlm(cfg)
    
    # Create memory-mapped file
    file_size = nlm * n_snapshots * sizeof(Float64)
    
    open(file_path, "w+") do io
        write(io, zeros(UInt8, file_size))
    end
    
    # Memory map the file
    mmap_array = Mmap.mmap(file_path, Array{Float64, 2}, (nlm, n_snapshots))
    
    MemoryMappedSpectralData(file_path, cfg, n_snapshots, nlm, mmap_array)
end

function add_snapshot!(mmsd::MemoryMappedSpectralData, 
                      spatial_field::Matrix{Float64}, 
                      snapshot_idx::Int)
    if snapshot_idx > mmsd.n_snapshots
        error("Snapshot index $snapshot_idx exceeds capacity $(mmsd.n_snapshots)")
    end
    
    # Analyze and store directly in memory-mapped array
    sh = analyze(mmsd.cfg, spatial_field)
    mmsd.mmap_array[:, snapshot_idx] = sh
end

function process_snapshots_streaming(mmsd::MemoryMappedSpectralData, 
                                   process_func::Function,
                                   chunk_size::Int=100)
    results = []
    
    n_chunks = div(mmsd.n_snapshots, chunk_size)
    
    for chunk in 1:n_chunks
        start_idx = (chunk - 1) * chunk_size + 1
        end_idx = min(chunk * chunk_size, mmsd.n_snapshots)
        
        # Process chunk
        chunk_data = mmsd.mmap_array[:, start_idx:end_idx]
        chunk_result = process_func(mmsd.cfg, chunk_data)
        push!(results, chunk_result)
        
        # Optional: trigger garbage collection
        if chunk % 10 == 0
            GC.gc()
        end
    end
    
    return results
end

function compute_temporal_statistics(mmsd::MemoryMappedSpectralData)
    # Compute statistics without loading all data into memory
    
    mean_spectrum = zeros(mmsd.nlm)
    var_spectrum = zeros(mmsd.nlm)
    
    # First pass: compute mean
    for i in 1:mmsd.n_snapshots
        mean_spectrum .+= mmsd.mmap_array[:, i]
    end
    mean_spectrum ./= mmsd.n_snapshots
    
    # Second pass: compute variance
    for i in 1:mmsd.n_snapshots
        diff = mmsd.mmap_array[:, i] - mean_spectrum
        var_spectrum .+= diff.^2
    end
    var_spectrum ./= (mmsd.n_snapshots - 1)
    
    return mean_spectrum, sqrt.(var_spectrum)
end

function cleanup!(mmsd::MemoryMappedSpectralData)
    # Close memory mapping and optionally remove file
    finalize(mmsd.mmap_array)
    # rm(mmsd.file_path)  # Uncomment to delete file
end

# Example: Process large climate dataset
# cfg = create_gauss_config(128, 128)
# n_years = 100
# n_snapshots_per_year = 365
# total_snapshots = n_years * n_snapshots_per_year

# mmsd = create_mmap_spectral_data("climate_data.bin", cfg, total_snapshots)

# # Add data (in practice, this would come from files)
# for i in 1:total_snapshots
#     synthetic_field = generate_climate_snapshot(i)  # User function
#     add_snapshot!(mmsd, synthetic_field, i)
# end

# # Compute statistics
# mean_spec, std_spec = compute_temporal_statistics(mmsd)

# cleanup!(mmsd)
# free_config(cfg)
```

## Integration with External Libraries

### Interfacing with Climate Models

```julia
using SHTnsKit
# using NCDatasets  # For NetCDF files

function read_climate_model_output(file_path::String, variable::String, time_index::Int)
    # Read data from NetCDF file (pseudo-code)
    # In practice, use NCDatasets.jl or similar
    
    # data = NCDatasets.Dataset(file_path) do ds
    #     ds[variable][:, :, time_index]
    # end
    
    # For demonstration, create synthetic data
    nlat, nlon = 96, 192  # Typical climate model resolution
    data = rand(nlat, nlon)
    
    return data
end

function climate_model_to_shtns(data::Matrix{Float64}, target_lmax::Int)
    # Convert climate model grid to SHTns format
    
    input_nlat, input_nlon = size(data)
    
    # Create appropriate configuration
    cfg = create_regular_config(target_lmax, target_lmax)
    target_nlat, target_nlon = get_nlat(cfg), get_nphi(cfg)
    
    # Interpolate to target grid (simplified)
    if (input_nlat, input_nlon) != (target_nlat, target_nlon)
        # Bilinear interpolation (in practice, use proper spherical interpolation)
        data_interpolated = imresize(data, (target_nlat, target_nlon))
    else
        data_interpolated = data
    end
    
    # Analyze
    sh = analyze(cfg, data_interpolated)
    
    return cfg, sh
end

function process_climate_ensemble(file_paths::Vector{String}, 
                                variable::String, 
                                target_lmax::Int)
    
    ensemble_spectra = []
    reference_cfg = nothing
    
    for file_path in file_paths
        println("Processing: $file_path")
        
        # Read multiple time steps
        n_time_steps = get_time_dimension_size(file_path)  # User function
        
        for t in 1:min(n_time_steps, 100)  # Limit for example
            data = read_climate_model_output(file_path, variable, t)
            cfg, sh = climate_model_to_shtns(data, target_lmax)
            
            if reference_cfg === nothing
                reference_cfg = cfg
            end
            
            push!(ensemble_spectra, sh)
        end
    end
    
    return reference_cfg, ensemble_spectra
end

# Example usage
# file_paths = ["model1_output.nc", "model2_output.nc", "model3_output.nc"]
# cfg, ensemble = process_climate_ensemble(file_paths, "temperature", 64)

# # Compute ensemble statistics
# n_members = length(ensemble)
# mean_spectrum = sum(ensemble) / n_members
# variance_spectrum = sum([(sp - mean_spectrum).^2 for sp in ensemble]) / (n_members - 1)

# println("Ensemble analysis complete")
# free_config(cfg)
```

This comprehensive advanced usage guide demonstrates sophisticated patterns for expert users of SHTnsKit.jl, covering everything from configuration management to large-scale data processing and integration with external scientific computing workflows.