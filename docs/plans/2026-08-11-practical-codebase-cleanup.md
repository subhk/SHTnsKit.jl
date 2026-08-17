# Practical Codebase Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove legacy and speculative surfaces while retaining the tested serial, MPI, single-GPU, AD, and packed-transform core.

**Architecture:** Keep numerical code paths intact and simplify their boundaries. Use one typed device API, one supported PencilArrays API, and one implementation for each distributed analysis path; delete C-compatibility and multi-GPU layers rather than replacing them.

**Tech Stack:** Julia 1.10–1.12, FFTW.jl, CUDA.jl extension, MPI.jl, PencilArrays.jl 0.19, PencilFFTs.jl, Test, JET, Aqua.

### Task 1: Define the reduced API with failing removal tests

**Files:**
- Create: `test/serial/test_cleanup_contract.jl`
- Modify: `test/serial/runtests.jl`

**Step 1: Write the failing contract test**

Add a test that requires the retained typed device API and requires removed
surfaces to be absent:

```julia
using Test
using SHTnsKit

@testset "cleanup contract" begin
    @test CPU() isa ComputeDevice
    @test GPU() isa ComputeDevice
    @test all(isdefined(SHTnsKit, name) for name in
              (:get_device, :set_device!, :to_device, :on_device))

    removed = (
        :SHTDevice, :CPU_DEVICE, :CUDA_DEVICE,
        :set_backend!, :current_backend, :with_backend, :reset_backend!,
        :create_gauss_config_gpu, :set_config_device!, :get_config_device,
        :is_gpu_config, :MultiGPUConfig, :create_multi_gpu_config,
        :multi_gpu_analysis, :multi_gpu_synthesis,
        :shtns_init, :shtns_create, :shtns_set_grid, :shtns_malloc,
        :SHT_GAUSS, :SHT_ALLOW_GPU,
    )
    @test all(name -> !isdefined(SHTnsKit, name), removed)

    cfg = create_gauss_config(3, 4; nlon=7)
    @test !hasproperty(cfg, :compute_device)
    @test !hasproperty(cfg, :device_preference)
end
```

Include it immediately after `test_device_utils.jl`.

**Step 2: Run the test to verify RED**

Run:

```bash
julia --startup-file=no --project=. test/serial/test_cleanup_contract.jl
```

Expected: FAIL because `ComputeDevice` is currently an alias, legacy names are
defined, and `SHTConfig` still contains device fields.

**Step 3: Commit the executable specification**

```bash
git add test/serial/test_cleanup_contract.jl test/serial/runtests.jl
git commit -m "test: define reduced public API"
```

### Task 2: Collapse device management to typed CPU/GPU operations

**Files:**
- Modify: `src/devices.jl`
- Modify: `src/device_utils.jl`
- Modify: `src/config.jl`
- Modify: `src/sphtor_transforms.jl`
- Modify: `src/SHTnsKit.jl`
- Modify: `test/serial/test_device_utils.jl`
- Modify: `test/serial/test_config_copy.jl`

**Step 1: Replace the backend hierarchy**

Make `ComputeDevice` the real abstract type and keep only typed values:

```julia
abstract type ComputeDevice end
struct CPU <: ComputeDevice end
struct GPU <: ComputeDevice end
```

Delete `SHTBackend` and all symbol normalization methods.

**Step 2: Reduce the global device API**

Keep a single state cell and these public operations in `device_utils.jl`:

```julia
const _DEVICE_STATE = Ref{Union{Nothing,ComputeDevice}}(nothing)

function get_device()
    requested = _DEVICE_STATE[]
    requested === nothing && return _check_cuda_available() ? GPU() : CPU()
    requested isa GPU && !_check_cuda_available() && return CPU()
    return requested
end

function set_device!(device::ComputeDevice)
    device isa GPU && !_check_cuda_available() &&
        @warn "CUDA requested but not available; using CPU until CUDA is functional"
    _DEVICE_STATE[] = device
    return get_device()
end

to_device(arr::AbstractArray, ::CPU) = _to_cpu(arr)
to_device(arr::AbstractArray, ::GPU) = _to_gpu(arr)
to_device(arr::AbstractArray) = to_device(arr, get_device())
on_device(::AbstractArray) = CPU()
```

Retain CUDA discovery and extension hooks. Delete backend aliases, symbol
overloads, selection preferences, dispatch helpers, macros, config forwarding,
and device-info wrappers.

**Step 3: Remove execution policy from `SHTConfig`**

Delete `compute_device` and `device_preference` fields, constructor keywords,
copy logic, and GPU-config helper functions. Remove the `is_gpu_config` branch
from `analysis_sphtor`; GPU execution remains explicit through
`gpu_analysis_sphtor`.

**Step 4: Clean exports and base fallbacks**

Export only `ComputeDevice`, `CPU`, `GPU`, `get_device`, `set_device!`,
`to_device`, and `on_device` from the device layer. Delete the removed fallback
functions from `SHTnsKit.jl`.

**Step 5: Update focused tests and verify GREEN**

Rewrite `test_device_utils.jl` around typed devices only. Remove config-device
assertions from `test_config_copy.jl`.

Run:

```bash
julia --startup-file=no --project=. test/serial/test_device_utils.jl
julia --startup-file=no --project=. test/serial/test_cleanup_contract.jl
julia --startup-file=no --project=. test/serial/test_configuration.jl
julia --startup-file=no --project=. test/serial/test_config_copy.jl
```

Expected: all focused tests PASS.

**Step 6: Commit**

```bash
git add src/devices.jl src/device_utils.jl src/config.jl src/sphtor_transforms.jl src/SHTnsKit.jl test/serial
git commit -m "refactor: keep one typed device API"
```

### Task 3: Remove legacy and multi-GPU CUDA surfaces

**Files:**
- Modify: `ext/SHTnsKitGPUExt.jl`
- Modify: `docs/src/gpu.md`
- Modify: `examples/gpu_acceleration.jl`
- Delete: `examples/multi_gpu_example.jl`

**Step 1: Add a failing CUDA-extension smoke assertion**

Extend the external CUDA smoke command to assert that the extension does not
define `SHTDevice` or `MultiGPUConfig`. Run it before implementation and confirm
it fails on those definitions.

**Step 2: Delete the redundant device system**

Remove the `SHTDevice` enum, legacy `set_device!`/`to_device` overloads, symbol
device handling, and legacy exports. Make `_is_cpu_device` accept only `CPU()`
and `GPU()`.

**Step 3: Delete multi-GPU code**

Remove `MultiGPUConfig`, configuration/discovery orchestration, P2P setup,
latitude-subset configuration, analysis/synthesis splitting, and streaming
helpers. Keep `get_available_gpus()` and `set_gpu_device(id)` because they are
useful for selecting one CUDA device.

**Step 4: Update docs and examples**

Remove multi-GPU sections and the multi-GPU example. Update the remaining GPU
example to construct a normal `SHTConfig` and call typed GPU functions directly.

**Step 5: Verify GREEN**

Run the CUDA extension smoke in the existing CUDA test environment. Verify
extension precompilation, `gpu_analysis(...; device=CPU())`, typed `on_device`,
and absence of the removed extension names.

**Step 6: Commit**

```bash
git add ext/SHTnsKitGPUExt.jl docs/src/gpu.md examples
git commit -m "refactor: remove legacy and multi-GPU CUDA layers"
```

### Task 4: Remove the dedicated SHTns C compatibility layer

**Files:**
- Delete: `src/api_compat.jl`
- Delete: `test/serial/test_api_compat.jl`
- Delete: `test/serial/test_flags.jl`
- Modify: `src/SHTnsKit.jl`
- Modify: `test/serial/runtests.jl`
- Modify: `test/serial/test_configuration.jl`
- Modify: `test/runtests.jl`
- Modify: `docs/src/api/index.md`
- Modify: `docs/src/grids.md`

**Step 1: Verify the removal test is RED**

Run `test_cleanup_contract.jl` and confirm the `shtns_*` and integer-flag absence
assertions still fail.

**Step 2: Remove the layer and exports**

Delete the include of `api_compat.jl` and its export block. Delete the dedicated
compatibility tests and includes. Remove compatibility-only testsets from
`test/runtests.jl` and `test_configuration.jl`.

Do not remove rotation functions from `rotations.jl`; they are used by the
rotation and AD implementations. Do not rename them in this cleanup.

**Step 3: Update user documentation**

Describe grids through `create_gauss_config`, `create_regular_config`, and
`create_config`. Remove C-API entries from the API index.

**Step 4: Verify GREEN**

Run:

```bash
julia --startup-file=no --project=. test/serial/test_cleanup_contract.jl
julia --startup-file=no --project=. test/serial/runtests.jl
```

Expected: cleanup contract and serial suite PASS.

**Step 5: Commit**

```bash
git add src test docs/src
git commit -m "refactor: remove C-style SHTns compatibility layer"
```

### Task 5: Target the supported PencilArrays API directly

**Files:**
- Modify: `ext/SHTnsKitParallelExt.jl`
- Modify: `test/parallel/test_mpi_audit_fixes.jl`
- Modify: `docs/src/parallel_installation.md`
- Modify: `docs/TROUBLESHOOTING.md`

**Step 1: Write the failing structural regression**

Add MPI assertions that `communicator(A)` equals `PencilArrays.get_comm(A)` and
`globalindices(A, dim)` equals
`PencilArrays.range_local(PencilArrays.pencil(A))[dim]`. Also assert that the
extension no longer defines `pencilarray_version_info`; this last assertion is
RED before cleanup.

**Step 2: Run the four-rank audit to verify RED**

Use the existing MPI wrapper for `test_mpi_audit_fixes.jl`.

Expected: FAIL only because the obsolete version-inspection function exists.

**Step 3: Replace compatibility probes**

Use direct helpers:

```julia
communicator(A) = PencilArrays.get_comm(A)
globalindices(A, dim) = PencilArrays.range_local(PencilArrays.pencil(A))[dim]
```

Delete version detection, cached capability dictionaries, old function-name
probes, direct field access, and broad fallback catches. Update troubleshooting
text to state the supported `0.19` API.

**Step 4: Run the audit to verify GREEN**

Expected: 34 existing assertions plus the new helper assertions pass on every
rank.

**Step 5: Commit**

```bash
git add ext/SHTnsKitParallelExt.jl test/parallel/test_mpi_audit_fixes.jl docs
git commit -m "refactor: target PencilArrays 0.19 directly"
```

### Task 6: Flatten parallel cache state and remove fake strategies

**Files:**
- Modify: `ext/SHTnsKitParallelExt.jl`
- Modify: `ext/ParallelTransforms.jl`
- Modify: `ext/ParallelPlans.jl`
- Modify: `test/parallel/test_mpi_extended.jl`

**Step 1: Write failing absence checks**

In the MPI audit, require `_ParallelExtState`,
`dist_analysis_cache_blocked`, and `dist_analysis_fused_cache_blocked` to be
undefined. Confirm RED.

**Step 2: Flatten state**

Replace `_ParallelExtState` plus aliases with direct constants for the enabled
flag, caches, cap, and locks. Preserve public cache toggle behavior.

**Step 3: Remove fake strategies**

Delete the two forwarding strategy functions. Remove the ignored
`use_cache_blocking` and `use_loop_fusion` keywords from `dist_analysis` and
update plan call sites. Change the extended MPI test to compare planned and
canonical non-planned analysis only.

**Step 4: Verify GREEN**

Run the four-rank audit and extended suites. Expected: all assertions PASS.

**Step 5: Commit**

```bash
git add ext test/parallel
git commit -m "refactor: simplify distributed analysis state and dispatch"
```

### Task 7: Repository-wide cleanup and verification

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: affected files under `docs/` and `examples/`
- Modify: `Project.toml` only if removed code leaves unused dependencies

**Step 1: Find stale references**

Run `rg` for every removed export, enum, symbol device value, multi-GPU name,
and compatibility constant. Update or remove all user-facing references. Do not
rewrite historical planning documents outside the current cleanup plan.

**Step 2: Run the default package suite**

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: zero failures.

**Step 3: Run static quality checks**

```bash
env SHTNSKIT_RUN_JET_TESTS=1 SHTNSKIT_RUN_AQUA_TESTS=1 \
  julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: JET and Aqua PASS.

**Step 4: Run MPI suites**

Run `test_mpi_audit_fixes.jl`, `test_mpi_comprehensive.jl`, and
`test_mpi_extended.jl` with four ranks. Expected: all PASS without hangs.

**Step 5: Run CUDA smoke**

Load CUDA and the package in the existing CUDA environment. Verify extension
precompilation and the typed CPU fallback. State explicitly if no functional GPU
was available for kernel execution.

**Step 6: Verify the cleanup itself**

```bash
git diff --check
git diff --stat
git status --short
```

Expected: no whitespace errors, a substantial net source reduction, and no
unrelated files modified. Preserve the pre-existing `.DS_Store` deletions.

**Step 7: Commit**

```bash
git add README.md CHANGELOG.md docs examples Project.toml Manifest.toml
git commit -m "docs: align package surface with simplified core"
```
