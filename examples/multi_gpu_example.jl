#!/usr/bin/env julia
"""
Multi-GPU Spherical Harmonic Transforms Example

This example demonstrates how to use SHTnsKit.jl with multiple GPUs
for accelerated spherical harmonic transforms.

Before running this example, ensure you have:
1. Multiple NVIDIA GPUs available
2. GPU packages installed: CUDA.jl, GPUArrays.jl, KernelAbstractions.jl

Usage:
    julia --project=. examples/multi_gpu_example.jl [--gpus 0,1] [--strategy latitude]
"""

using SHTnsKit
using Printf
using BenchmarkTools

# Try to load GPU packages
try
    using CUDA, GPUArrays, KernelAbstractions
    println("Multi-GPU packages loaded successfully")
    GPU_AVAILABLE = true
catch e
    println("WARNING: GPU packages not available: $e")
    println("  Install with: julia -e 'using Pkg; Pkg.add([\"CUDA\", \"GPUArrays\", \"KernelAbstractions\"])'")
    GPU_AVAILABLE = false
end

"""
    parse_args()

Parse command line arguments for multi-GPU configuration.
"""
function parse_args()
    gpu_ids = nothing
    strategy = :latitude
    
    for i in 1:length(ARGS)
        if ARGS[i] == "--gpus" && i < length(ARGS)
            gpu_ids = [parse(Int, id) for id in split(ARGS[i+1], ",")]
        elseif ARGS[i] == "--strategy" && i < length(ARGS)
            strategy = Symbol(ARGS[i+1])
        end
    end
    
    return gpu_ids, strategy
end

"""
    run_multi_gpu_example()

Main multi-GPU demonstration function.
"""
function run_multi_gpu_example()
    println("=" ^ 60)
    println("Multi-GPU Spherical Harmonic Transforms")
    println("=" ^ 60)
    
    if !GPU_AVAILABLE
        println("ERROR: GPU packages not available. Cannot run multi-GPU example.")
        return
    end
    
    # Parse command line arguments
    gpu_ids, strategy = parse_args()
    
    # Problem configuration
    lmax = 64
    nlat = lmax + 2
    nlon = 2 * (2*lmax + 1)
    
    println("Problem size: lmax=$lmax, grid=$(nlat)×$(nlon)")
    println("Distribution strategy: $strategy")
    
    # Detect available GPUs
    println("\nGPU Detection:")
    println("-" ^ 40)
    
    available_gpus = get_available_gpus()
    if isempty(available_gpus)
        println("ERROR: No GPUs detected.")
        return
    end
    
    println("Available GPUs:")
    for gpu in available_gpus
        println("  $(gpu.device) $(gpu.id): $(gpu.name)")
    end
    
    # Create multi-GPU configuration
    try
        mgpu_config = create_multi_gpu_config(lmax, nlat; 
                                             nlon=nlon, 
                                             strategy=strategy,
                                             gpu_ids=gpu_ids)
        
        println("\\nMulti-GPU Configuration:")
        println("  Using $(length(mgpu_config.gpu_devices)) GPUs")
        println("  Strategy: $(mgpu_config.distribution_strategy)")
        
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            println("  GPU $i: $(gpu.device) $(gpu.id) - $(gpu.name)")
        end
        
    catch e
        println("ERROR: Failed to create multi-GPU configuration: $e")
        return
    end
    
    # Create test data
    println("\\nCreating Test Data:")
    println("-" ^ 40)
    
    θ, φ = mgpu_config.base_config.θ, mgpu_config.base_config.φ
    spatial_data = zeros(nlat, nlon)
    
    # Create bandlimited test function
    for i in 1:nlat, j in 1:nlon
        spatial_data[i,j] = (1.0 + 
                            0.5 * (3*cos(θ[i])^2 - 1) +    # Y_2^0
                            0.3 * sin(θ[i]) * cos(φ[j]) +   # Approximation of Y_4^1
                            0.2 * sin(θ[i])^2 * cos(2*φ[j])) # Approximation of Y_6^2
    end
    
    println("Created bandlimited test function")
    
    # Single GPU benchmark for comparison
    println("\\nPerformance Comparison:")
    println("-" ^ 40)
    
    # Single GPU timing
    single_gpu_config = create_gauss_config_gpu(lmax, nlat; nlon=nlon, device=:auto)
    
    print("Single GPU Analysis: ")
    single_gpu_time = @belapsed gpu_analysis_safe($single_gpu_config, $spatial_data)
    @printf "%.2f ms\\n" (single_gpu_time * 1000)
    
    # Multi-GPU timing
    print("Multi-GPU Analysis:  ")
    try
        multi_gpu_time = @belapsed multi_gpu_analysis($mgpu_config, $spatial_data)
        @printf "%.2f ms" (multi_gpu_time * 1000)
        speedup = single_gpu_time / multi_gpu_time
        @printf " (%.1f× speedup)\\n" speedup
        
        # Test accuracy
        println("\\nAccuracy Test:")
        println("-" ^ 40)
        
        single_coeffs = gpu_analysis_safe(single_gpu_config, spatial_data)
        multi_coeffs = multi_gpu_analysis(mgpu_config, spatial_data)
        
        coefficient_error = maximum(abs.(single_coeffs - multi_coeffs))
        @printf "Single vs Multi-GPU coefficient difference: %.2e\\n" coefficient_error
        
        # Test synthesis roundtrip
        multi_reconstructed = multi_gpu_synthesis(mgpu_config, multi_coeffs; real_output=true)
        roundtrip_error = maximum(abs.(spatial_data - multi_reconstructed))
        @printf "Multi-GPU roundtrip error: %.2e\\n" roundtrip_error
        
        # Memory usage analysis
        println("\\nMemory Distribution:")
        println("-" ^ 40)
        
        total_memory = estimate_memory_usage(mgpu_config.base_config, :analysis)
        memory_per_gpu = total_memory ÷ length(mgpu_config.gpu_devices)
        
        println("  Total estimated memory: $(total_memory÷(1024^2)) MB")
        println("  Memory per GPU: $(memory_per_gpu÷(1024^2)) MB")
        
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            set_gpu_device(gpu.device, gpu.id)
            try
                if gpu.device == :cuda
                    mem_info = CUDA.MemoryInfo()
                    println("  GPU $i: $(mem_info.free÷(1024^3)) GB free / $(mem_info.total÷(1024^3)) GB total")
                else
                    println("  GPU $i: Memory info not available for $(gpu.device)")
                end
            catch e
                println("  GPU $i: Could not get memory info: $e")
            end
        end
        
    catch e
        println("ERROR: Multi-GPU analysis failed: $e")
        if contains(string(e), "only supports")
            println("NOTE: Current implementation has limited distribution strategy support")
        end
    end
    
    # Distribution strategy demonstration
    println("\\nDistribution Strategy Details:")
    println("-" ^ 40)
    
    try
        distributed_data = distribute_spatial_array(spatial_data, mgpu_config)
        
        println("Data distribution across $(length(distributed_data)) GPUs:")
        for (i, chunk) in enumerate(distributed_data)
            lat_range, lon_range = chunk.indices
            chunk_size = (length(lat_range), length(lon_range))
            data_mb = prod(chunk_size) * 16 ÷ (1024^2)  # ComplexF64 = 16 bytes
            
            println("  GPU $i ($(chunk.gpu.device) $(chunk.gpu.id)):")
            println("    Latitude range: $(first(lat_range))-$(last(lat_range))")
            println("    Longitude range: $(first(lon_range))-$(last(lon_range))")
            println("    Chunk size: $(chunk_size)")
            println("    Data size: ~$(data_mb) MB")
        end
    catch e
        println("Could not demonstrate distribution: $e")
    end
    
    println("\\nMulti-GPU Example Completed!")
    
    # Usage recommendations
    println("\\nUsage Recommendations:")
    println("-" ^ 40)
    println("• Use :latitude strategy for most problems (currently implemented)")
    println("• Ensure problem size is large enough to benefit from multi-GPU")
    println("• Monitor GPU memory usage to avoid out-of-memory errors")
    println("• Consider MPI + GPU for distributed computing across nodes")
    
    # Cleanup
    try
        destroy_config(single_gpu_config)
    catch e
        # Config may not support destroy
    end
end

"""
    main()

Entry point for multi-GPU example.
"""
function main()
    try
        run_multi_gpu_example()
    catch e
        println("ERROR: Multi-GPU example failed: $e")
        if isa(e, MethodError) && contains(string(e), "get_available_gpus")
            println("\\nNOTE: Multi-GPU functionality requires GPU extension to be loaded.")
            println("Make sure CUDA.jl is installed and functional.")
        end
        rethrow(e)
    end
end

# Run example if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end