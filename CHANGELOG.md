# Changelog

## Unreleased (v1.2.18)

### Breaking changes

**`dist_synthesis!` / `dist_synthesis_sphtor!` reject `real_output=false` with real output arrays.**
`real_output=false` used to return the *real* field wrapped as complex (a real
buffer re-typed), so writing it into a `PencilArray{Float64}` happened to work.
It now performs a genuine complex synthesis — summing the m ≥ 0 half without the
Hermitian mirror — which is a different function, not the same field in a wider
type. On a typical config the complex-path result has `|imag|` up to 1.34 and a
real part differing from the real field by 1.31, against a field magnitude of
2.96, so no tolerance check can bridge the two.

*Porting:* if you passed `real_output=false` with a real output array and wanted
the real field, pass `real_output=true`. If you want the true complex synthesis,
pass a complex output `PencilArray`. The error message states both.

**`analysis_axisym` / `analysis_axisym_l` now return different values.**
Both omitted the φ quadrature factor `cfg.cphi * nlon = 2π`, so every returned
coefficient was `1/2π` too small — they inverted neither `synthesis_axisym` nor
the m=0 column of the full `analysis`. They now agree with both. Anything that
compensated for the old scale downstream must drop that compensation.

### Fixed

- **Silent precision loss in batch QST/sphtor transforms.** `analysis_qst_batch`,
  `_synthesis_qst_batch` and the sphtor batch pair derived their output element
  type from one input array instead of promoting across all of them, truncating
  double-precision components to a single-precision sibling's type (measured
  error 2.05e-8 instead of ~1e-17).
- **`load_config` silently downgraded `:driscoll_healy` grids to `:regular`**,
  changing both the θ nodes and the quadrature weights; analysis→synthesis error
  degraded from 8.7e-16 to ~3e-3. `save_config` now records `use_dh_weights`.
- **`shtns_set_grid` returned success for unrecognized grid codes** while
  producing a south-to-north grid still reported as north-pole-first. The
  fallback branch was missing the `reverse!` the real Gauss branch applies.
- **Pole-inclusive grids with `nlat == 1` produced an all-NaN config** (`π/0`)
  that returned NaN from every subsequent transform with no error raised. Now
  rejected with a message naming the cause.
- **`dist_SH_mul_mx!` crashed on every `mres > 1` config** — it walked all orders
  through `LM_index`, which requires multiples of `mres`.
- **`dist_SH_Yrotate` crashed on `mres > 1`.** A Y-rotation mixes orders and so
  cannot be represented in an `mres`-strided layout at all; it now says that
  up front instead of failing deep inside the rotation.
- **`device_transfer_arrays(cfg, ...)` rejected `:cuda`/`:amdgpu`** — the config
  vocabulary — because `to_device` accepted only `:cpu`/`:gpu`.
- **ForwardDiff could not flow through any plan-based batch transform.**
  `SHTPlan` is FFTW-backed and cannot hold `ForwardDiff.Dual`; the batch entry
  points now route non-FFTW element types through the plan-free `cfg`-form
  transforms.
- **Cached FFT plans could silently fall back to an O(n²) DFT.** The plan cache
  keys on shape and strides but not alignment, so reuse on a differently-aligned
  matrix threw and callers quietly downgraded. Plans are now built `UNALIGNED`.
- Legendre south-pole and normalization-comment corrections carried over from the
  orthonormal refactor; eleven comments prescribed conversions the code no longer
  performs.

### Performance

- **Distributed analysis: 3 collectives per call → 1 on first use, 0 after.**
  The `φ_is_local_all` / `θ_is_distributed` predicates are reduced together in a
  single `Allreduce` and cached per `(pencil, communicator)`.
- **`dist_synthesis_packed_cplx` is single-pass**, down from two full distributed
  syntheses. The negative-m φ bins are filled in the same θ/m traversal as the
  positive ones, reusing one Legendre row per `(m, θ)`. It now matches the serial
  reference exactly.
- **Legendre table memory halved.** `prepare_plm_tables!` was building
  `NP_tables`/`NdP_tables` bit-for-bit identical to `plm_tables`/`dplm_tables`;
  they now alias. `estimate_table_memory` previously reported half the true
  figure, so jobs sized by it allocated twice their budget.
- Batch FFT helpers reuse the shared plan cache instead of re-planning per call.

### Internal

- `pack_lm!`/`pack_lm`/`unpack_lm!`/`unpack_lm` in `src/layout.jl` replace six
  open-coded copies of the packed↔dense `(l,m)` mapping. The `m % mres` guard had
  to be fixed three separate times across those copies.
