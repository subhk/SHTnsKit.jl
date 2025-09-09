#!/usr/bin/env julia
"""
GPU-Accelerated Spherical Harmonic Transforms Example

This example demonstrates how to use SHTnsKit.jl with GPU acceleration
using CUDA, AMDGPU, GPUArrays, and KernelAbstractions.

Before running this example, install the required GPU packages:
```bash
julia -e 'using Pkg; Pkg.add(["CUDA", "AMDGPU", "GPUArrays", "KernelAbstractions"])'
```

Usage:
    julia --project=. examples/gpu_acceleration.jl [--device cuda|amdgpu|cpu]
"""

using SHTnsKit
using Printf
using BenchmarkTools

# Try to load GPU packages - they're optional
try
    using CUDA, AMDGPU, GPUArrays, KernelAbstractions
    println("✓ GPU packages loaded successfully")
    GPU_AVAILABLE = true
catch e
    println("⚠ GPU packages not available: $e")
    println("  Install with: julia -e 'using Pkg; Pkg.add([\"CUDA\", \"AMDGPU\", \"GPUArrays\", \"KernelAbstractions\"])'"
    GPU_AVAILABLE = false
end

"""
    parse_device_arg()

Parse command line arguments for device selection.
"""
function parse_device_arg()
    if length(ARGS) >= 2 && ARGS[1] == "--device"
        device_str = lowercase(ARGS[2])
        if device_str == "cpu"
            return :cpu
        elseif device_str == "cuda"
            return :cuda
        elseif device_str == "amdgpu"
            return :amdgpu
        else
            @warn "Unknown device '$device_str', using auto-detection"
            return :auto
        end
    else
        return :auto
    end
end

"""
    run_gpu_comparison_example()

Compare CPU vs GPU performance for spherical harmonic transforms.
"""
function run_gpu_comparison_example()
    println("=" ^ 60)
    println("GPU-Accelerated Spherical Harmonic Transforms")
    println("=" ^ 60)
    
    # Problem configuration
    lmax = 64
    nlat = lmax + 2  
    nlon = 2 * (2*lmax + 1)
    
    println("Problem size: lmax=$lmax, grid=$(nlat)×$(nlon)")
    
    # Parse command line device preference
    device_pref = parse_device_arg()
    
    # Create configurations for different devices
    println("\n📋 Device Detection and Configuration:")
    println("-" ^ 40)
    
    # Auto-detect best available device
    if device_pref == :auto
        selected_device, gpu_available = select_compute_device()
        println("🔍 Auto-selected device: $selected_device")
    else
        selected_device = device_pref
        gpu_available = (device_pref != :cpu)
        println("👤 User-specified device: $selected_device") 
    end
    
    # CPU configuration (always available)
    cfg_cpu = create_gauss_config(lmax, nlat; nlon=nlon)
    set_config_device!(cfg_cpu, :cpu)
    println("✓ CPU configuration created")
    
    # GPU configuration (if available)
    cfg_gpu = nothing
    if GPU_AVAILABLE && selected_device != :cpu
        try
            cfg_gpu = create_gauss_config_gpu(lmax, nlat; nlon=nlon, device=selected_device)
            println("✓ GPU configuration created ($(get_config_device(cfg_gpu)))")
        catch e
            println("❌ Failed to create GPU configuration: $e")
            cfg_gpu = nothing
        end
    end
    
    # Create test data - bandlimited function for exact roundtrip
    println("\n🔧 Creating Test Data:")
    println("-" ^ 40)
    
    θ, φ = cfg_cpu.θ, cfg_cpu.φ
    spatial_data = zeros(cfg_cpu.nlat, cfg_cpu.nlon)
    
    # Create a bandlimited test function (representable by chosen lmax)
    for i in 1:cfg_cpu.nlat, j in 1:cfg_cpu.nlon
        # Combination of low-order harmonics: Y_0^0 + Y_2^0 + Y_4^1 + Y_6^2
        spatial_data[i,j] = (1.0 + 
                            0.5 * (3*cos(θ[i])^2 - 1) +  # Y_2^0
                            0.3 * sin(θ[i]) * cos(φ[j]) + # Y_4^1 (simplified)
                            0.2 * sin(θ[i])^2 * cos(2*φ[j])) # Y_6^2 (simplified)
    end
    
    println("Created bandlimited test function with max energy at low l")
    
    # CPU Benchmarking
    println("\n⚡ Performance Benchmarking:")
    println("-" ^ 40)
    
    println("CPU Performance:")
    print("  Analysis transform: ")
    cpu_analysis_time = @belapsed analysis($cfg_cpu, $spatial_data)
    @printf "%.2f ms\n" (cpu_analysis_time * 1000)
    
    # Get coefficients for synthesis test
    coeffs_cpu = analysis(cfg_cpu, spatial_data)
    
    print("  Synthesis transform: ")
    cpu_synthesis_time = @belapsed synthesis($cfg_cpu, $coeffs_cpu; real_output=true)
    @printf "%.2f ms\n" (cpu_synthesis_time * 1000)
    
    # GPU Benchmarking (if available)
    if cfg_gpu !== nothing
        println("\n$(uppercase(string(get_config_device(cfg_gpu)))) Performance:")
        
        try
            # Test memory-safe GPU functions first
            print("  GPU Analysis (safe): ")
            gpu_analysis_time = @belapsed gpu_analysis_safe($cfg_gpu, $spatial_data)
            @printf "%.2f ms" (gpu_analysis_time * 1000)
            speedup = cpu_analysis_time / gpu_analysis_time
            @printf " (%.1f× speedup)\n" speedup
            
            # Get GPU coefficients for synthesis
            coeffs_gpu = gpu_analysis_safe(cfg_gpu, spatial_data)
            
            print("  GPU Synthesis (safe): ")
            gpu_synthesis_time = @belapsed gpu_synthesis_safe($cfg_gpu, $coeffs_gpu; real_output=true)
            @printf "%.2f ms" (gpu_synthesis_time * 1000)
            speedup = cpu_synthesis_time / gpu_synthesis_time  
            @printf " (%.1f× speedup)\n" speedup
            
            # Also test direct GPU functions
            println("\n  Direct GPU functions (may use more memory):")
            
            # Check memory usage
            required_mem = estimate_memory_usage(cfg_gpu, :analysis)
            println("  Estimated memory usage: $(required_mem÷(1024^2)) MB")
            
            print("  GPU Analysis (direct): ")
            gpu_analysis_time = @belapsed gpu_analysis($cfg_gpu, $spatial_data)  
            @printf "%.2f ms\n" (gpu_analysis_time * 1000)
            
        catch e
            if contains(string(e), "not fully implemented") || contains(string(e), "GPU extension")
                println("⚠ GPU functions using CPU fallback (GPU packages not loaded)")
            else
                println("❌ GPU benchmarking failed: $e")
            end
        end
    end
    
    # Accuracy Testing
    println("\n🎯 Roundtrip Accuracy Test:")
    println("-" ^ 40)
    
    # CPU roundtrip
    coeffs_cpu = analysis(cfg_cpu, spatial_data)
    reconstructed_cpu = synthesis(cfg_cpu, coeffs_cpu; real_output=true)
    cpu_error = maximum(abs.(spatial_data - reconstructed_cpu))
    @printf "CPU roundtrip error:  %.2e\n" cpu_error
    
    # GPU roundtrip (if available)
    if cfg_gpu !== nothing
        try
            coeffs_gpu = gpu_analysis_safe(cfg_gpu, spatial_data)
            reconstructed_gpu = gpu_synthesis_safe(cfg_gpu, coeffs_gpu; real_output=true) 
            gpu_error = maximum(abs.(spatial_data - reconstructed_gpu))
            @printf "GPU roundtrip error:  %.2e\n" gpu_error
            
            # Compare CPU vs GPU results
            coeff_diff = maximum(abs.(coeffs_cpu - coeffs_gpu))
            @printf "CPU-GPU coefficient difference: %.2e\n" coeff_diff
            
            # Memory management example
            println("\n💾 Memory Management:")
            current_device = get_config_device(cfg_gpu)
            if current_device != :cpu
                try
                    mem_info = gpu_memory_info(current_device)
                    println("  GPU Memory: $(mem_info.free÷(1024^3)) GB free / $(mem_info.total÷(1024^3)) GB total")
                    gpu_clear_cache!(current_device)
                catch e
                    println("  Could not access GPU memory info: $e")
                end
            end
            
        catch e
            if contains(string(e), "GPU extension")
                println("⚠ GPU accuracy test skipped (GPU packages not loaded)")
            else
                println("❌ GPU accuracy test failed: $e")
            end
        end
    end
    
    # Vector field example
    println("\n🌀 Vector Field GPU Acceleration:")
    println("-" ^ 40)
    
    # Create test vector field
    u_wind = zeros(cfg_cpu.nlat, cfg_cpu.nlon) 
    v_wind = zeros(cfg_cpu.nlat, cfg_cpu.nlon)
    
    for i in 1:cfg_cpu.nlat, j in 1:cfg_cpu.nlon
        θ_val, φ_val = θ[i], φ[j]
        u_wind[i,j] = 20 * sin(2*θ_val) * cos(φ_val)  # Zonal wind
        v_wind[i,j] = 10 * cos(θ_val) * sin(φ_val)    # Meridional wind
    end
    
    # CPU vector transform
    print("CPU spheroidal-toroidal analysis: ")
    cpu_vector_time = @belapsed spat_to_SHsphtor($cfg_cpu, $u_wind, $v_wind)
    @printf "%.2f ms\n" (cpu_vector_time * 1000)
    
    # GPU vector transform (if available)
    if cfg_gpu !== nothing
        try
            print("GPU spheroidal-toroidal analysis: ")
            gpu_vector_time = @belapsed gpu_spat_to_SHsphtor($cfg_gpu, $u_wind, $v_wind)
            @printf "%.2f ms" (gpu_vector_time * 1000)
            speedup = cpu_vector_time / gpu_vector_time
            @printf " (%.1f× speedup)\n" speedup
            
            # Test vector field roundtrip
            sph_gpu, tor_gpu = gpu_spat_to_SHsphtor(cfg_gpu, u_wind, v_wind)
            u_recon_gpu, v_recon_gpu = gpu_SHsphtor_to_spat(cfg_gpu, sph_gpu, tor_gpu; real_output=true)
            
            vector_error = max(maximum(abs.(u_wind - u_recon_gpu)), maximum(abs.(v_wind - v_recon_gpu)))
            @printf "GPU vector roundtrip error: %.2e\n" vector_error
            
        catch e
            if contains(string(e), "GPU extension")
                println("⚠ GPU vector transform skipped (GPU packages not loaded)")
            else
                println("❌ GPU vector transform failed: $e")
            end
        end
    end
    
    # Device management examples
    println("\n⚙️  Device Management Examples:")
    println("-" ^ 40)
    
    # Show device switching
    println("Current CPU config device: $(get_config_device(cfg_cpu))")
    
    if cfg_gpu !== nothing
        println("Current GPU config device: $(get_config_device(cfg_gpu))")
        println("Is GPU config? $(is_gpu_config(cfg_gpu))")
        
        # Demonstrate device switching
        original_device = get_config_device(cfg_gpu)
        set_config_device!(cfg_gpu, :cpu)  
        println("After switching to CPU: $(get_config_device(cfg_gpu))")
        set_config_device!(cfg_gpu, original_device)
        println("After switching back: $(get_config_device(cfg_gpu))")
    end
    
    # Cleanup
    destroy_config(cfg_cpu)
    if cfg_gpu !== nothing
        destroy_config(cfg_gpu)
    end
    
    println("\n✨ GPU acceleration example completed!")
    
    if !GPU_AVAILABLE
        println("\n💡 To enable GPU acceleration:")
        println("   1. Install GPU packages: julia -e 'using Pkg; Pkg.add([\"CUDA\", \"AMDGPU\", \"GPUArrays\", \"KernelAbstractions\"])'"
        println("   2. Restart Julia and run this example again")
    end
end

"""
    main()

Main entry point for the GPU acceleration example.
"""
function main()
    try
        run_gpu_comparison_example()
    catch e
        println("❌ Example failed: $e")
        if isa(e, MethodError)
            println("\n💡 This may be due to incomplete GPU extension implementation.")
            println("   The current implementation provides the framework for GPU acceleration.")
        end
        rethrow(e)
    end
end

# Run the example if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end