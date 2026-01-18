# SHTnsKit.jl

```@raw html
<div style="background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); color: white; padding: 2.5rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;">
    <h1 style="font-size: 2.5rem; margin: 0 0 0.5rem 0; color: white; border: none;">SHTnsKit.jl</h1>
    <p style="font-size: 1.25rem; margin: 0; opacity: 0.95;">High-Performance Spherical Harmonic Transforms for Julia</p>
    <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">CPU | GPU | MPI Distributed | Auto-Differentiation</p>
</div>
```

[![Build Status](https://github.com/subhk/SHTnsKit.jl/workflows/CI/badge.svg)](https://github.com/subhk/SHTnsKit.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/SHTnsKit.jl/stable)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## Features

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; transition: transform 0.2s, box-shadow 0.2s;">
    <h3 style="color: #2563eb; margin-top: 0;">Core Transforms</h3>
    <ul style="margin: 0; padding-left: 1.25rem; color: #475569;">
        <li>Forward & backward spherical harmonic transforms</li>
        <li>Scalar, vector, and complex field support</li>
        <li>In-place operations for memory efficiency</li>
    </ul>
</div>

<div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem;">
    <h3 style="color: #7c3aed; margin-top: 0;">GPU Acceleration</h3>
    <ul style="margin: 0; padding-left: 1.25rem; color: #475569;">
        <li>CUDA/cuFFT for NVIDIA GPUs</li>
        <li>KernelAbstractions.jl backend</li>
        <li>Automatic CPU fallback</li>
    </ul>
</div>

<div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem;">
    <h3 style="color: #059669; margin-top: 0;">MPI Distributed</h3>
    <ul style="margin: 0; padding-left: 1.25rem; color: #475569;">
        <li>PencilArrays domain decomposition</li>
        <li>Scalable to thousands of cores</li>
        <li>Efficient communication patterns</li>
    </ul>
</div>

<div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem;">
    <h3 style="color: #dc2626; margin-top: 0;">Analysis Tools</h3>
    <ul style="margin: 0; padding-left: 1.25rem; color: #475569;">
        <li>Power spectrum analysis</li>
        <li>Energy diagnostics</li>
        <li>Wigner D-matrix rotations</li>
    </ul>
</div>

</div>
```

---

## Quick Start

### Installation

```julia
using Pkg

# Basic installation
Pkg.add("SHTnsKit")

# With GPU support
Pkg.add(["SHTnsKit", "CUDA", "KernelAbstractions"])

# With MPI support
Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs"])
```

### Hello World

```julia
using SHTnsKit

# Create configuration for lmax=16
cfg = create_gauss_config(16, 18)

# Create a test pattern on the sphere
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat
    x = cfg.x[i]  # cos(θ)
    spatial[i, :] .= (3*x^2 - 1)/2  # Y_2^0 pattern
end

# Transform: spatial → spectral → spatial
Alm = analysis(cfg, spatial)
recovered = synthesis(cfg, Alm)

# Verify roundtrip accuracy
println("Error: ", maximum(abs.(spatial - recovered)))  # ~1e-14
```

!!! tip "Pro Tip"
    For large problems (lmax > 64), use GPU acceleration with `gpu_analysis()` and `gpu_synthesis()` for significant speedup.

---

## GPU Example

```julia
using SHTnsKit, CUDA

cfg = create_gauss_config(128, 130)
spatial = rand(cfg.nlat, cfg.nlon)

# GPU-accelerated transforms
Alm = gpu_analysis(cfg, spatial)
recovered = gpu_synthesis(cfg, Alm)

# Safe version with automatic CPU fallback
Alm_safe = gpu_analysis_safe(cfg, spatial)
```

---

## Scientific Applications

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

<div style="background: #eff6ff; border-radius: 8px; padding: 1rem; border-left: 4px solid #2563eb;">
    <strong style="color: #1e40af;">Climate Science</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        Atmospheric dynamics, weather prediction, ocean circulation
    </p>
</div>

<div style="background: #fef3c7; border-radius: 8px; padding: 1rem; border-left: 4px solid #f59e0b;">
    <strong style="color: #92400e;">Geophysics</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        Gravitational fields, magnetic anomalies, Earth surface modeling
    </p>
</div>

<div style="background: #f3e8ff; border-radius: 8px; padding: 1rem; border-left: 4px solid #7c3aed;">
    <strong style="color: #5b21b6;">Astrophysics</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        CMB analysis, stellar dynamics, gravitational waves
    </p>
</div>

<div style="background: #ecfdf5; border-radius: 8px; padding: 1rem; border-left: 4px solid #10b981;">
    <strong style="color: #065f46;">Fluid Dynamics</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        Vorticity-divergence decomposition, turbulence on spheres
    </p>
</div>

</div>
```

---

## Documentation

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

<a href="quickstart/" style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s;">
    <div>
        <strong>Quick Start</strong>
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Get started in minutes</p>
    </div>
</a>

<a href="gpu/" style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s;">
    <div>
        <strong>GPU Guide</strong>
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">CUDA acceleration</p>
    </div>
</a>

<a href="distributed/" style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s;">
    <div>
        <strong>Distributed</strong>
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">MPI parallelization</p>
    </div>
</a>

<a href="api/" style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: white; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s;">
    <div>
        <strong>API Reference</strong>
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Complete documentation</p>
    </div>
</a>

</div>
```

```@contents
Pages = [
    "quickstart.md",
    "gpu.md",
    "distributed.md",
    "api/index.md",
    "examples/index.md",
    "performance.md",
    "advanced.md"
]
Depth = 1
```

---

## Installation Options

| Setup | Command | Use Case |
|-------|---------|----------|
| **Basic** | `Pkg.add("SHTnsKit")` | Single CPU, getting started |
| **GPU** | `+ CUDA, KernelAbstractions` | NVIDIA GPU acceleration |
| **MPI** | `+ MPI, PencilArrays, PencilFFTs` | Cluster computing |
| **Full** | All of the above | Maximum flexibility |

!!! note "Requirements"
    - Julia 1.10 or later
    - For GPU: NVIDIA GPU with CUDA support
    - For MPI: OpenMPI or MPICH installed

---

## Contributing

We welcome contributions! See our [GitHub repository](https://github.com/subhk/SHTnsKit.jl) for:
- Bug reports and feature requests
- Documentation improvements
- Pull requests

---

## Citation

If you use SHTnsKit.jl in your research, please cite:

```bibtex
@software{shtnskit,
  author = {Kar, Subhajit},
  title = {SHTnsKit.jl: High-Performance Spherical Harmonic Transforms},
  url = {https://github.com/subhk/SHTnsKit.jl},
  year = {2024}
}
```

---

## License

SHTnsKit.jl is released under the **GNU General Public License v3.0**.

```@raw html
<div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: #f8fafc; border-radius: 8px;">
    <p style="margin: 0; color: #64748b;">
        Made for the scientific computing community
    </p>
</div>
```
