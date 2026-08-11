# Practical Codebase Cleanup Design

## Goal

Make SHTnsKit smaller and easier to maintain by removing legacy and speculative
surfaces while preserving the tested serial, MPI, single-GPU, AD, and packed
transform capabilities.

## Product Boundary

The package remains a pure-Julia spherical-harmonic transform library with:

- scalar, vector, QST, packed, and complex transforms;
- Gauss, regular, pole-inclusive, and Driscoll-Healy grids;
- plans, diagnostics, operators, rotations, and batching;
- MPI/PencilArrays distributed transforms;
- single-device CUDA acceleration using `CPU()` and `GPU()`;
- the existing ForwardDiff, Zygote, ChainRules, and LoopVectorization extensions.

The cleanup removes:

- the C-style `shtns_*` configuration/memory/profiling compatibility layer and
  its integer flag constants;
- the legacy `SHTDevice`, `CPU_DEVICE`, and `CUDA_DEVICE` enum API;
- symbol device inputs such as `:cpu`, `:gpu`, and `:cuda`;
- duplicate backend/configuration device APIs and configuration device metadata;
- multi-GPU configuration, latitude splitting, streaming, and its example;
- runtime compatibility probes for PencilArrays versions excluded by
  `Project.toml`;
- internal distributed-analysis strategy names that only forward to the same
  implementation.

The `shtns_*` rotation functions are not part of `src/api_compat.jl`; they are
the implementation used by the rotation and AD code. They remain in this pass
to avoid replacing working mathematics with a naming-only rewrite.

## Architecture

Device selection has one public state API: `get_device()` and
`set_device!(CPU() | GPU())`. Array placement remains `to_device` and
`on_device`. GPU transform functions retain their explicit entry points and use
typed devices only. `SHTConfig` becomes mathematical configuration again; it no
longer stores execution policy, and CPU transforms do not silently jump into a
GPU extension.

The CUDA extension contains only CUDA discovery, transfer, single-GPU
transforms, FFT helpers, and memory inspection. It no longer carries a second
device type system or the sequential latitude-chunking code presented as
multi-GPU execution.

The MPI extension targets PencilArrays `0.19`, using `get_comm` and
`range_local` directly. Cache state is represented by direct module constants,
not a state struct immediately unpacked into aliases. One implementation owns
distributed scalar analysis; obsolete strategy aliases and ignored strategy
keywords are removed.

## Error Handling

Calls into an unloaded optional extension retain concise package-specific error
messages. `set_device!(GPU())` continues to resolve safely when CUDA is not
functional. Unsupported device values fail by dispatch rather than by a symbol
normalization table. MPI helper failures expose the supported dependency API
instead of falling through broad `try`/`catch` chains.

## Verification

Removal tests assert that deleted exports are absent and that the surviving
typed device API remains coherent. Focused serial tests cover configuration and
device behavior. CUDA verification loads and precompiles the extension and
checks the typed CPU fallback; real GPU kernels remain hardware-dependent.

The complete serial suite, JET, Aqua, and the four-rank MPI audit,
comprehensive, and extended suites must pass. Documentation and examples must
contain no references to removed APIs. `git diff --check` must be clean, and the
final source diff must show a meaningful net line reduction.

## Non-Goals

- Rewriting numerical kernels solely for style.
- Removing tested AD or packed-storage functionality without evidence that it
  duplicates another supported path.
- Supporting dependency releases outside `Project.toml` compatibility bounds.
- Adding traits or generalized backend abstractions for hypothetical devices.
