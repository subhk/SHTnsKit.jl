# GPU Acceleration

SHTnsKit 2.0 uses vendor-neutral device dispatch. CUDA and AMDGPU extensions
implement the same scalar, vector, QST, packed, batch, planned, operator, and
diagnostic APIs for their device arrays.

## Install a backend

NVIDIA:

```julia
using Pkg
Pkg.add(["CUDA", "GPUArrays", "GPUArraysCore", "KernelAbstractions"])
```

AMD:

```julia
using Pkg
Pkg.add(["AMDGPU", "GPUArrays", "GPUArraysCore", "KernelAbstractions"])
```

Load `SHTnsKit` and the vendor package plus the shared GPU packages in the same
Julia session to activate the extension.

## Generic device-array workflow

The array type selects the vendor. Results remain on that device:

```julia
using SHTnsKit, CUDA, GPUArrays, GPUArraysCore, KernelAbstractions
CUDA.functional() || error("CUDA is not functional")

cfg = create_gauss_config(64, 66)
field = CUDA.rand(Float64, cfg.nlat, cfg.nlon)

coefficients = analysis(cfg, field)
recovered = synthesis(cfg, coefficients)

@assert coefficients isa CUDA.AnyCuArray
@assert recovered isa CUDA.AnyCuArray
```

The AMDGPU form differs only in allocation and the resulting array type:

```julia
using SHTnsKit, AMDGPU, GPUArrays, GPUArraysCore, KernelAbstractions
AMDGPU.functional() || error("AMDGPU is not functional")

cfg = create_gauss_config(64, 66)
field = AMDGPU.ROCArray(rand(Float64, cfg.nlat, cfg.nlon))
coefficients = analysis(cfg, field)
recovered = synthesis(cfg, coefficients)
```

The same inference applies to `analysis_sphtor`, `analysis_qst`, packed and
batch transforms, and their synthesis counterparts.

## Strict device dispatch

[`CPU`](@ref) and [`GPU`](@ref) make execution intent explicit:

```julia
host_coefficients = analysis(CPU(), cfg, host_field)
device_coefficients = analysis(GPU(), cfg, device_field)
```

`GPU()` is strict. If the requested backend is unavailable or cannot handle an
operation, SHTnsKit raises [`BackendUnavailableError`](@ref) instead of copying
to the CPU. If multiple functional adapters make selection ambiguous,
SHTnsKit raises `ArgumentError`; pass a vendor-specific prototype array to
select the intended adapter.

Use the following helpers for explicit placement:

```julia
set_device!(CPU())
host_copy = to_device(CPU(), device_field)

# With one functional adapter loaded:
device_copy = to_device(GPU(), host_field)

# With more than one functional adapter, select the vendor by prototype:
device_copy = to_device(GPU(), host_field, device_prototype)
@assert on_device(device_copy) isa GPU
```

`get_device()` returns the preferred typed device. `set_device!(GPU())`
validates that at least one loaded GPU adapter is functional.

## Reusable plans and in-place calls

An [`SHTPlan`](@ref) can own the reusable transform workspace while device
arrays supply device storage:

```julia
cfg = create_gauss_config(64, 66)
plan = SHTPlan(cfg; use_rfft=true)

field = CUDA.rand(Float64, cfg.nlat, cfg.nlon)
coefficients = CUDA.zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
recovered = CUDA.zeros(Float64, cfg.nlat, cfg.nlon)

analysis!(plan, coefficients, field)
synthesis!(plan, recovered, coefficients)
```

Output comes before input, matching the CPU in-place API. A single plan is
mutable and must not be used by concurrent tasks.

## CUDA host-returning compatibility wrappers

The CUDA extension retains `gpu_analysis` and `gpu_synthesis` for applications
that pass host arrays and expect host results:

```julia
using SHTnsKit, CUDA, GPUArrays, GPUArraysCore, KernelAbstractions

cfg = create_gauss_config(64, 66)
host_field = rand(cfg.nlat, cfg.nlon)
host_coefficients = gpu_analysis(cfg, host_field)
host_recovered = gpu_synthesis(cfg, host_coefficients)
```

These wrappers perform host/device transfers. Prefer generic device-array
dispatch inside GPU-resident pipelines.

`gpu_analysis_safe` and `gpu_synthesis_safe` are also CUDA compatibility
wrappers. They explicitly fall back to CPU execution when CUDA is not
functional, the memory estimate does not fit, or CUDA reports an out-of-memory
error:

```julia
host_coefficients = gpu_analysis_safe(cfg, host_field)
host_recovered = gpu_synthesis_safe(cfg, host_coefficients)
```

They do not turn strict `analysis(GPU(), ...)` calls into fallback calls.

## CUDA-specific utilities

The following helpers are provided by the CUDA extension and are not the
vendor-neutral transform interface:

- `get_available_gpus()` and `set_gpu_device(id)` enumerate/select CUDA devices.
- `gpu_memory_info()`, `check_gpu_memory(bytes)`, and `gpu_clear_cache!()` expose
  CUDA memory state.
- `estimate_memory_usage(cfg, operation)` estimates transform storage.
- `create_cufft_plan`, `gpu_fft!`, and `gpu_ifft!` provide reusable cuFFT plans.

```julia
plan = create_cufft_plan(cfg.nlat, cfg.nlon)
buffer = CUDA.rand(ComplexF64, cfg.nlat, cfg.nlon)
original = copy(buffer)
gpu_fft!(plan, buffer)
gpu_ifft!(plan, buffer)
@assert Array(buffer) ≈ Array(original)
```

## Distributed GPU arrays

For a `PencilArray` backed by CUDA or AMDGPU storage, the
`DistTransposePlan` scalar, spheroidal/toroidal, and QST bang transforms
are device-native. MPI communication uses the device directly when the MPI
stack is GPU-aware; otherwise SHTnsKit may use bounded pinned staging at that
communication boundary.

Other unsupported MPI+GPU mathematical entry points raise
`BackendUnavailableError` before whole-call host staging. See [Distributed
Computing](distributed.md) and the [SHTns 3.7 parity matrix](shtns37-parity.md)
for the exact tracked surface.

## Troubleshooting

- Check `CUDA.functional()` or `AMDGPU.functional()` before allocating.
- Load all dependencies named in the selected extension row in
  [Installation](installation.md).
- Keep inputs, outputs, scratch, and prototypes on the same vendor.
- If both vendors are loaded, pass a device-array prototype to disambiguate
  `to_device(GPU(), value, prototype)`.
- Use `gpu_analysis_safe` only when a host-returning CUDA fallback is intended;
  it can hide a transfer that is expensive inside a time-stepping loop.
