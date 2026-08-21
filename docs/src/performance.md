# Performance Guide

Measure on your workload before choosing an optimization. Transform cost and
the best backend depend on resolution, batch size, grid, coefficient type, and
how long data remains on a device or rank.

## Reuse serial plans

For repeated transforms on one grid, [`SHTPlan`](@ref) reuses FFT plans and
workspace:

```@example performance-plan
using SHTnsKit

cfg = create_gauss_config(64, 66)
plan = SHTPlan(cfg; use_rfft=true)
field = rand(cfg.nlat, cfg.nlon)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
field_out = zeros(cfg.nlat, cfg.nlon)

analysis!(plan, alm, field)
synthesis!(plan, field_out, alm)
nothing
```

Use `use_rfft=true` for real scalar inputs and outputs when
`cfg.mmax <= cfg.nlon ÷ 2`. Vector transforms can share the plan but retain the
complex workspace they require.

A plan is mutable and not concurrency-safe. Allocate one per simultaneous
worker:

```julia
plans = [SHTPlan(cfg; use_rfft=true) for _ in 1:Threads.nthreads()]
Threads.@threads for i in eachindex(fields)
    analysis!(plans[Threads.threadid()], outputs[i], fields[i])
end
```

## Reuse scratch without a plan

The configuration-form bang methods accept caller-owned FFT scratch:

```@example performance-scratch
using SHTnsKit

cfg = create_gauss_config(32, 34; nlon=129)
fft_scratch = scratch_fft(cfg)
field = rand(cfg.nlat, cfg.nlon)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
field_out = scratch_spatial(cfg)

analysis!(cfg, alm, field; fft_scratch)
synthesis!(cfg, field_out, alm; fft_scratch)
nothing
```

Output arguments precede inputs in every bang transform.

## Batch fields

Batch transforms reduce repeated setup and keep the field index in dimension
three:

```@example performance-batch
using SHTnsKit

cfg = create_gauss_config(32, 34)
fields = rand(cfg.nlat, cfg.nlon, 8)
alm = analysis_batch(cfg, fields; use_rfft=true)
recovered = synthesis_batch(cfg, alm; real_output=true, use_rfft=true)
size(recovered)
```

Use the vector and QST batch families when all components share the same grid.

## Legendre tables

The default Gauss constructor uses on-the-fly recurrence. For a fixed grid,
precomputed tables can exchange memory for less repeated recurrence work:

```julia
bytes = estimate_table_memory(cfg)
prepare_plm_tables!(cfg)       # also enables table use
disable_plm_tables!(cfg)       # releases the tables
```

Benchmark both modes for the target resolution. `create_gauss_fly_config`
forces the lower-memory on-the-fly configuration; regular grids precompute
tables by default unless `precompute_plm=false` is requested.

## FFTW and Julia threads

SHTnsKit does not expose the removed legacy thread-control wrappers. Start Julia
with the desired Julia thread count, and configure FFTW through FFTW.jl:

```julia
using FFTW
FFTW.set_num_threads(min(Threads.nthreads(), 4))
```

Avoid oversubscription when Julia tasks, BLAS, FFTW, and MPI ranks share the
same cores. More FFTW threads are not always faster for the short longitude
FFTs used by low-resolution transforms.

## GPU pipelines

Generic operations on CUDA or AMDGPU arrays return device arrays. Keep
intermediate fields and coefficients on the device; repeated host-returning
`gpu_analysis`/`gpu_synthesis` calls pay transfer costs.

Reuse [`SHTPlan`](@ref) with device outputs for repeated operations. CUDA users
who only need longitude FFTs can also reuse `create_cufft_plan`. See [GPU
Acceleration](gpu.md) for the distinction between vendor-neutral dispatch and
CUDA compatibility helpers.

## Distributed transforms

Choose between two output layouts:

- `dist_analysis` / `dist_synthesis` return or consume a replicated dense
  spectrum, convenient for global post-processing.
- `DistTransposePlan` keeps the spectrum distributed by `m`, avoiding a
  full dense spectrum on every rank.

For repeated dense-path operations, reuse `DistAnalysisPlan`, `DistPlan`,
`DistSphtorPlan`, or `DistQstPlan`. Construct every plan collectively with an
identical configuration on all ranks.

## Benchmark correctly

Use BenchmarkTools, interpolate setup values, and synchronize GPU operations
when timing device work:

```julia
using BenchmarkTools
@btime analysis!($plan, $alm, $field)
```

For MPI, report the maximum elapsed time across ranks and benchmark after plan
construction. For GPU, include transfers only when the production workflow
actually transfers on every iteration.

## Checklist

- Start with a band-limited correctness check at the chosen grid size.
- Reuse plans or scratch for repeated calls.
- Batch independent fields when their configuration matches.
- Keep GPU and distributed data in their native layouts between operations.
- Compare on-the-fly recurrence with precomputed tables under the real memory
  budget.
- Measure end-to-end throughput, not only one kernel.
