# Legendre Overflow Fix (lmax ≥ 151) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SHTnsKit transforms finite and accurate for all lmax (verified to 1024), removing the hard NaN ceiling at lmax ≥ 151.

**Architecture:** SHTnsKit's kernels always run in the **orthonormal + Condon–Shortley** convention (the 7 user norms + CS-phase are applied separately to `alm` via `convert_alm_norm!`, which uses bounded O(1) scale factors). The bug: `Plm_row!` returns **un-normalized** `P_l^m`, whose magnitude reaches ~1e305 at m≈150 and overflows Float64 by lmax≈155; kernels then do `Nlm(tiny)·P(Inf)·alm = NaN`, poisoning every transform because the sum runs over all (l,m). The fix replaces the "un-normalized P × tiny Nlm" split with a **fully-normalized analytical recurrence** (Holmes–Featherstone) that computes the orthonormal `P̄_l^m = Nlm·P_l^m` directly, bounded by ~1 at every (l,m). Kernels then use `P̄` directly and drop the separate `Nlm[l]·` multiply. `cfg.Nlm` and `convert_alm_norm!` are untouched.

**Tech Stack:** Julia 1.11, FFTW; extensions MPI/PencilArrays/PencilFFTs (parallel), ChainRulesCore/Zygote/ForwardDiff (AD), CUDA/KernelAbstractions (GPU), LoopVectorization.

**Validated core recurrence (orthonormal P̄, proven 2026-05-31 to match current `Nlm·P` to 2e-12 at lmax≤64 and stay bounded — max|P̄|≈1.05 — to lmax=1024):**

```julia
s = sqrt(max(0, 1 - x*x))
P̄_0^0 = NORM00                                      # = sqrt(1/(4π)) (orthonormal); == cfg.Nlm[1,1]
P̄_m^m = -sqrt((2m+1)/(2m)) * s * P̄_{m-1}^{m-1}      # stable sectoral (carries CS phase)
P̄_{m+1}^m = sqrt(2m+3) * x * P̄_m^m
P̄_l^m = a*x*P̄_{l-1}^m - b*P̄_{l-2}^m       (l ≥ m+2)
  a = sqrt(((2l-1)*(2l+1)) / ((l-m)*(l+m)))
  b = sqrt(((2l+1)*(l-1-m)*(l-1+m)) / ((2l-3)*(l-m)*(l+m)))
```

`NORM00 = sqrt(1/(4π))`. Do NOT derive coefficients from `cfg.Nlm` ratios — `cfg.Nlm` itself underflows to 0 at high m (lmax≳256), giving 0/0. Coefficients must be analytical as above.

**Constraint:** No `git commit` without explicit user permission (project rule). Each task ends with a *staged-and-ready* note, not an automatic commit, unless the user has granted commit permission for execution.

**Environment note:** This machine's `julia` launcher (juliaup) is broken (root-owned `juliaup.json`). Use the direct binary:
`JL=~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia`. Serial tests run with `$JL --project=.`. MPI/JET tests need an env with the extra deps (a temp project that `dev`s SHTnsKit + adds the weakdeps), launched via MPI.jl's `mpiexec()` wrapper (system mpiexec is incompatible OpenMPI).

---

## File Structure

- `src/legendre.jl` — add normalized row functions `Plm_norm_row!`, `Plm_norm_and_dPdtheta_row!`, `Plm_norm_dPdtheta_over_sinth_row!` (and `_and_dPdx` if the tables path needs it). These RETURN orthonormal `P̄` (and its θ-derivatives) directly. The existing un-normalized functions stay until all callers migrate, then are removed in the final task.
- `src/kernels.jl` — switch scalar + sphtor kernels to consume `P̄` directly; delete the `Nlm[l+1,col]*` factor (it is now folded into `P̄`). Pole closed-forms (`_dPdtheta_at_pole`, `_P_over_sinth_at_pole`) must be re-expressed to include the orthonormal `N` consistently (they currently multiply by `N`).
- `src/config.jl` — `prepare_plm_tables!` builds the `NP`/`NdP` fused tables from `P̄` directly (bounded); the raw `plm`/`dplm` tables (if still exposed) likewise.
- `src/core_transforms.jl`, `src/sphtor_transforms.jl`, `src/batch_transforms.jl`, `src/complex_packed.jl`, `src/local.jl`, `src/api_compat.jl` — OTF call sites switch to the normalized row functions + drop the per-l `Nlm` multiply.
- `ext/ParallelTransforms.jl`, `ext/ParallelLocal.jl`, `ext/SHTnsKitLoopVecExt.jl`, `ext/SHTnsKitAdvancedADExt.jl`, `ext/SHTnsKitGPUExt.jl` — same migration in the parallel / loopvec / AD / GPU paths.
- `test/serial/test_legendre.jl` (extend) + new `test/serial/test_highlmax.jl` — correctness + high-lmax round-trip.
- `docs/src/*` and `GeoDynamo.jl/scripts/sht_scaling_benchmark.jl` — remove the "lmax ≥ 151 NaN" caveat once the suite passes at high lmax.

**Migration strategy:** Add normalized functions ALONGSIDE the existing ones (Task 1), migrate one transform path at a time with verification (Tasks 2–7), then delete the dead un-normalized functions (Task 9). This keeps the suite green throughout.

---

## Task 0: Baseline & verification harness

**Files:**
- Create: `test/serial/test_highlmax.jl`
- Create (scratch): `/tmp/legfix_baseline.jl`

- [ ] **Step 1: Capture golden baseline at lmax ≤ 150 (current code is correct here).**

Create `/tmp/legfix_baseline.jl`:

```julia
using SHTnsKit, Serialization
golden = Dict{Int,Tuple{Matrix{ComplexF64},Matrix{Float64}}}()
for lmax in (8, 32, 64, 128, 150)
    nlat = lmax + 2; nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    alm = zeros(ComplexF64, lmax+1, lmax+1)
    for m in 0:lmax, l in m:lmax
        s = 1/(1+l)^2
        alm[l+1,m+1] = m==0 ? complex(s) : complex(s, 0.5s)
    end
    f = SHTnsKit.synthesis(cfg, alm; real_output=true)
    a2 = SHTnsKit.analysis(cfg, f)
    golden[lmax] = (a2, f)
end
serialize("/tmp/legfix_golden.jls", golden)
println("golden saved: ", keys(golden))
```

- [ ] **Step 2: Run it (current code).**

Run: `$JL --project=. /tmp/legfix_baseline.jl`
Expected: `golden saved: ...` with no error (lmax ≤ 150 is finite under current code).

- [ ] **Step 3: Write the high-lmax round-trip test (currently failing — documents the bug and the target).**

Create `test/serial/test_highlmax.jl`:

```julia
using Test, SHTnsKit

@testset "High-lmax transforms are finite and accurate" begin
    for lmax in (150, 160, 256, 512)
        nlat = lmax + 2; nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        # band-limited field from a decaying spectrum
        alm0 = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax, l in m:lmax
            sc = 1/(1+l)^2
            alm0[l+1, m+1] = m==0 ? complex(sc) : complex(sc, 0.5sc)
        end
        f = SHTnsKit.synthesis(cfg, alm0; real_output=true)
        @test all(isfinite, f)
        a = SHTnsKit.analysis(cfg, f)
        @test all(isfinite, a)
        @test isapprox(a, alm0; rtol=1e-9, atol=1e-11)
    end
end
```

- [ ] **Step 4: Run it to confirm it FAILS now (documents the bug).**

Run: `$JL --project=. test/serial/test_highlmax.jl`
Expected: FAIL at lmax=160 (`all(isfinite, f)` is false).

- [ ] **Step 5: Stage.** `git add test/serial/test_highlmax.jl` (do not commit without permission).

---

## Task 1: Normalized scalar recurrence `Plm_norm_row!`

**Files:**
- Modify: `src/legendre.jl` (add new function after `Plm_row!`, ~line 186)
- Test: `test/serial/test_legendre.jl` (extend) and `/tmp/legfix_validate.jl`

- [ ] **Step 1: Write the standalone validation (failing — function not defined).**

Create `/tmp/legfix_validate.jl`:

```julia
using SHTnsKit
function check(lmax; x=0.3137)
    nlat=lmax+2; cfg=create_gauss_config(lmax,nlat;nlon=2*lmax+1); N=cfg.Nlm
    P=Vector{Float64}(undef,lmax+1); g=Vector{Float64}(undef,lmax+1)
    maxrel=0.0; fin=true; mx=0.0
    for m in 0:lmax
        SHTnsKit.Plm_row!(P,x,lmax,m)
        SHTnsKit.Plm_norm_row!(g,x,lmax,m)
        fin &= all(isfinite, @view g[m+1:end]); mx=max(mx, maximum(abs, @view g[m+1:end]))
        for l in m:lmax
            ref = N[l+1,m+1]*P[l+1]
            (isfinite(ref) && abs(ref) > 1e-10) && (maxrel = max(maxrel, abs(g[l+1]-ref)/abs(ref)))
        end
    end
    println("lmax=$lmax rel-err=$(round(maxrel,sigdigits=3)) finite=$fin max|P̄|=$(round(mx,sigdigits=3))")
    @assert fin "non-finite at lmax=$lmax"
    @assert maxrel < 1e-8 "mismatch at lmax=$lmax (only valid where current Nlm*P is accurate, lmax<=128)"
end
check(64); check(128); check(256); check(512); check(1024)
```

- [ ] **Step 2: Run to verify it fails (function undefined).**

Run: `$JL --project=. /tmp/legfix_validate.jl`
Expected: `UndefVarError: Plm_norm_row!`.

- [ ] **Step 3: Implement `Plm_norm_row!` in `src/legendre.jl`.**

```julia
const _INV_SQRT_4PI = 0.28209479177387814  # sqrt(1/(4π)) = orthonormal P̄_0^0

"""
    Plm_norm_row!(P, x, lmax, m)

Fill `P[l+1] = P̄_l^m(x)` for l = m..lmax with the ORTHONORMAL + Condon–Shortley
associated Legendre functions (i.e. exactly `cfg.Nlm[l,m] * (raw P_l^m)`), via an
analytical fully-normalized recurrence that stays bounded (|P̄| ≲ 1) at all l, m —
no overflow at high lmax. Entries l < m are zeroed.
"""
function Plm_norm_row!(P::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    @inbounds fill!(P, zero(T))
    m < 0 && throw(ArgumentError("m must be ≥ 0"))
    lmax >= m || return P
    s = sqrt(max(zero(T), one(T) - x*x))
    # sectoral P̄_m^m via stable recurrence from P̄_0^0
    pmm = T(_INV_SQRT_4PI)
    @inbounds for k in 1:m
        pmm = -sqrt(T(2k + 1) / T(2k)) * s * pmm
    end
    P[m+1] = pmm
    lmax == m && return P
    P[m+2] = sqrt(T(2m + 3)) * x * pmm
    @inbounds for l in (m+2):lmax
        a = sqrt((T(2l - 1) * T(2l + 1)) / (T(l - m) * T(l + m)))
        b = sqrt((T(2l + 1) * T(l - 1 - m) * T(l - 1 + m)) / (T(2l - 3) * T(l - m) * T(l + m)))
        P[l+1] = a * x * P[l] - b * P[l-1]
    end
    return P
end
```

Export note: add `Plm_norm_row!` to the export list near the other `Plm_*` exports in `src/SHTnsKit.jl` only if the existing `Plm_row!` is exported (match the existing visibility; if `Plm_row!` is internal, keep `Plm_norm_row!` internal too).

- [ ] **Step 4: Run the validation; expect PASS.**

Run: `$JL --project=. /tmp/legfix_validate.jl`
Expected: all lines `finite=true`, `max|P̄|` ≈ 0.6–1.1, `rel-err < 1e-8` (the ~1.6e-10 plateau above lmax≈150 is the *current* `Nlm*P` losing precision, not the new code; the assert tolerance 1e-8 accommodates it).

- [ ] **Step 5: Add a permanent unit test in `test/serial/test_legendre.jl`.**

```julia
@testset "Plm_norm_row! bounded + matches Nlm*P (lmax<=128)" begin
    for lmax in (16, 64, 128), x in (-0.9, 0.13, 0.77)
        cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1); N = cfg.Nlm
        P = zeros(lmax+1); g = zeros(lmax+1)
        for m in 0:lmax
            SHTnsKit.Plm_row!(P, x, lmax, m)
            SHTnsKit.Plm_norm_row!(g, x, lmax, m)
            @test all(isfinite, @view g[m+1:end])
            for l in m:lmax
                ref = N[l+1,m+1]*P[l+1]
                abs(ref) > 1e-9 && @test isapprox(g[l+1], ref; rtol=1e-9)
            end
        end
    end
    # high lmax: just bounded + finite
    cfg = create_gauss_config(512, 514; nlon=1025); g = zeros(513)
    for m in 0:512
        SHTnsKit.Plm_norm_row!(g, 0.31, 512, m)
        @test all(isfinite, @view g[m+1:end]) && maximum(abs, @view g[m+1:end]) < 2.0
    end
end
```

- [ ] **Step 6: Run the legendre tests.** Run: `$JL --project=. test/serial/test_legendre.jl` → Expected: PASS.
- [ ] **Step 7: Stage** `src/legendre.jl test/serial/test_legendre.jl`.

---

## Task 2: Scalar OTF kernels use `P̄` (the unblock)

**Files:**
- Modify: `src/kernels.jl` (`_scalar_analysis_kernel_otf!` ~line 33, `_scalar_synthesis_kernel_otf` ~line 57)

Current synthesis OTF kernel (kernels.jl:57):
```julia
@inline function _scalar_synthesis_kernel_otf(cfg, alm, P, i, col, m, lmax)
    Plm_row!(P, cfg.x[i], lmax, m)
    acc = zero(eltype(alm))
    Nlm = cfg.Nlm
    @inbounds for l in m:lmax
        acc += (Nlm[l+1, col] * P[l+1]) * alm[l+1, col]
    end
    return acc
end
```

- [ ] **Step 1: Add failing scalar high-lmax round-trip test (orthonormal).**

Add to `test/serial/test_highlmax.jl`:
```julia
@testset "scalar OTF round-trip lmax 256/512 orthonormal" begin
    for lmax in (256, 512)
        cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)  # on_the_fly default
        alm0 = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax, l in m:lmax; sc=1/(1+l)^2; alm0[l+1,m+1]= m==0 ? complex(sc) : complex(sc,0.5sc); end
        f = SHTnsKit.synthesis(cfg, alm0; real_output=true); @test all(isfinite, f)
        a = SHTnsKit.analysis(cfg, f); @test isapprox(a, alm0; rtol=1e-9, atol=1e-11)
    end
end
```

- [ ] **Step 2: Run → FAIL (NaN at lmax 256).** Run: `$JL --project=. test/serial/test_highlmax.jl`.

- [ ] **Step 3: Rewire the two OTF kernels to use `Plm_norm_row!` and drop the `Nlm` multiply.**

Replace `_scalar_synthesis_kernel_otf` body:
```julia
@inline function _scalar_synthesis_kernel_otf(cfg, alm, P, i, col, m, lmax)
    Plm_norm_row!(P, cfg.x[i], lmax, m)   # P now holds orthonormal P̄ (= Nlm·rawP)
    acc = zero(eltype(alm))
    @inbounds for l in m:lmax
        acc += P[l+1] * alm[l+1, col]     # Nlm already folded into P̄
    end
    return acc
end
```

Replace `_scalar_analysis_kernel_otf!` (kernels.jl:33) likewise:
```julia
@inline function _scalar_analysis_kernel_otf!(alm, cfg, Fph, P, i, col, m, lmax, scale_phi)
    Plm_row_norm = Plm_norm_row!(P, cfg.x[i], lmax, m)
    Fi = Fph[i, col]
    wi = cfg.w[i]
    @inbounds for l in m:lmax
        alm[l+1, col] += (wi * P[l+1] * scale_phi) * Fi
    end
end
```
(Drop the `Nlm = cfg.Nlm` hoist and the `Nlm[l+1,col]*` factor in both.)

- [ ] **Step 4: Run high-lmax test → PASS.** Run: `$JL --project=. test/serial/test_highlmax.jl` → Expected: PASS at 256/512.

- [ ] **Step 5: Regression — golden match at lmax ≤ 150.**

Create `/tmp/legfix_regress.jl`:
```julia
using SHTnsKit, Serialization
golden = deserialize("/tmp/legfix_golden.jls")
for (lmax,(a_ref,f_ref)) in sort(collect(golden))
    nlat=lmax+2; cfg=create_gauss_config(lmax,nlat;nlon=2*lmax+1)
    alm=zeros(ComplexF64,lmax+1,lmax+1)
    for m in 0:lmax,l in m:lmax; s=1/(1+l)^2; alm[l+1,m+1]= m==0 ? complex(s) : complex(s,0.5s); end
    f=SHTnsKit.synthesis(cfg,alm;real_output=true); a2=SHTnsKit.analysis(cfg,f)
    println("lmax=$lmax  synth Δ=$(maximum(abs,f.-f_ref))  analysis Δ=$(maximum(abs,a2.-a_ref))")
    @assert maximum(abs, f .- f_ref) < 1e-10
    @assert maximum(abs, a2 .- a_ref) < 1e-10
end
println("REGRESSION OK")
```
Run: `$JL --project=. /tmp/legfix_regress.jl` → Expected: `REGRESSION OK` (Δ ≲ 1e-12; OTF scalar path is now exercised by golden too).

- [ ] **Step 6: Run existing scalar tests.** Run: `$JL --project=. test/serial/test_basic_transforms.jl` and `test/serial/test_truncated_transforms.jl` → Expected: PASS.
- [ ] **Step 7: Stage** `src/kernels.jl test/serial/test_highlmax.jl`.

---

## Task 3: Precomputed tables path uses `P̄`

**Files:**
- Modify: `src/config.jl` `prepare_plm_tables!` (~line 1157–1200); `src/kernels.jl` table kernels `_scalar_analysis_kernel!` (line 24) and `_scalar_synthesis_kernel` (line 48).

The fused tables `NP[l+1,i] = Nlm·P_l^m(x_i)` are built (config.jl:1199) as `Nlm_lm * tbl[l+1,i]` where `tbl` comes from `Plm_and_dPdx_row!` (raw P). That overflows the same way at high lmax. The table kernels already consume `NP` (the fused product), so once `NP` is built from `P̄` they are correct unchanged — but verify.

- [ ] **Step 1: Add failing test (tables path, high lmax).**

Add to `test/serial/test_highlmax.jl`:
```julia
@testset "tables path round-trip lmax 256" begin
    lmax=256; cfg=create_gauss_config(lmax,lmax+2;nlon=2*lmax+1)
    SHTnsKit.prepare_plm_tables!(cfg)
    alm0=zeros(ComplexF64,lmax+1,lmax+1)
    for m in 0:lmax,l in m:lmax; sc=1/(1+l)^2; alm0[l+1,m+1]= m==0 ? complex(sc) : complex(sc,0.5sc); end
    f=SHTnsKit.synthesis(cfg,alm0;real_output=true); @test all(isfinite,f)
    a=SHTnsKit.analysis(cfg,f); @test isapprox(a,alm0;rtol=1e-9,atol=1e-11)
end
```

- [ ] **Step 2: Run → FAIL (NaN tables at 256).**

- [ ] **Step 3: Build `NP` from `P̄` directly in `prepare_plm_tables!`.**

In `src/config.jl` around line 1178–1199, replace the per-row fill so that `NP[l+1,i] = P̄_l^m(x_i)` directly. Use `Plm_norm_row!` for the value table; for the θ-derivative-fused `NdP` table use `Plm_norm_and_dPdtheta_row!` from Task 4 (sequence: do Task 4 before the `NdP` part of this step, or split — the value table `NP` can be done now, `NdP` after Task 4). Concretely for `NP`:
```julia
g = Vector{Float64}(undef, lmax + 1)
for m in 0:mmax
    tblNP = NP[m+1]
    for i in 1:nlat
        Plm_norm_row!(g, cfg.x[i], lmax, m)
        @inbounds for l in m:lmax
            tblNP[l+1, i] = g[l+1]
        end
    end
end
```
(Drop the separate `Nlm_lm *` multiply — `g` already carries it. Match the existing table container shape/orientation in the current code.)

- [ ] **Step 4: Run → PASS.** Then golden regression `/tmp/legfix_regress.jl` with tables (add a tables variant) → Δ < 1e-10.

- [ ] **Step 5: Run** `test/serial/test_kernels.jl`, `test/serial/test_plan.jl` → Expected: PASS.
- [ ] **Step 6: Stage** `src/config.jl src/kernels.jl test/serial/test_highlmax.jl`.

---

## Task 4: Normalized θ-derivative rows (vector transforms)

**Files:**
- Modify: `src/legendre.jl` (add `Plm_norm_and_dPdtheta_row!`, `Plm_norm_dPdtheta_over_sinth_row!`)

**Recurrence to use (orthonormal, validate-first).** With `P̄` from Task 1, the θ-derivative satisfies the stable l-coupled relation
`sinθ · dP̄_l^m/dθ = l·d_{l+1}^m·P̄_{l+1}^m − (l+1)·d_l^m·P̄_{l-1}^m`, with `d_l^m = sqrt((l^2−m^2)/(4l^2−1))`.
So compute `P̄` up to lmax+1, then `dP̄_l^m/dθ = (l·d_{l+1}^m·P̄_{l+1}^m − (l+1)·d_l^m·P̄_{l-1}^m)/sinθ` for interior points, and `P̄_l^m/sinθ` from `P̄`. At the poles use the existing closed forms (`_dPdtheta_at_pole`, `_P_over_sinth_at_pole` in kernels.jl) but seeded with the orthonormal normalization consistent with `P̄` (those helpers currently take `N`; pass the orthonormal `N` value `Plm_norm`-consistently, i.e. fold `N` so they return the `P̄`-scaled limit).

- [ ] **Step 1: Standalone validation harness (failing).**

Create `/tmp/legfix_dtheta.jl`:
```julia
using SHTnsKit
function check(lmax; x=0.4273)
    cfg=create_gauss_config(lmax,lmax+2;nlon=2*lmax+1); N=cfg.Nlm
    P=zeros(lmax+1); dP=zeros(lmax+1)                 # current (raw) θ-deriv
    g=zeros(lmax+1); dg=zeros(lmax+1)                 # normalized
    maxrel=0.0; fin=true
    for m in 0:lmax
        SHTnsKit.Plm_and_dPdtheta_row!(P,dP,x,lmax,m)
        SHTnsKit.Plm_norm_and_dPdtheta_row!(g,dg,x,lmax,m)
        fin &= all(isfinite,@view dg[m+1:end])
        for l in max(1,m):lmax
            ref = N[l+1,m+1]*dP[l+1]
            (isfinite(ref) && abs(ref)>1e-9) && (maxrel=max(maxrel,abs(dg[l+1]-ref)/abs(ref)))
        end
    end
    println("lmax=$lmax dθ rel-err=$(round(maxrel,sigdigits=3)) finite=$fin")
    @assert fin; @assert maxrel < 1e-7
end
check(64); check(128); check(256); check(512)
```

- [ ] **Step 2: Run → FAIL (function undefined).**

- [ ] **Step 3: Implement `Plm_norm_and_dPdtheta_row!` and `Plm_norm_dPdtheta_over_sinth_row!` in `src/legendre.jl`** using the recurrence above. Compute `P̄` up to `lmax+1` internally (one extra term) for the `dθ` formula; handle `|x|→1` (sinθ→0) via the pole closed forms. (Full code is written during this task after the standalone harness in Step 1 passes; the harness is the acceptance gate. Mirror the structure of the existing `Plm_and_dPdtheta_row!` / `Plm_dPdtheta_over_sinth_row!` for argument order and pole handling.)

- [ ] **Step 4: Run validation → PASS** (rel-err < 1e-7 at lmax ≤ 128; finite to 512).

- [ ] **Step 5: Permanent unit test** in `test/serial/test_legendre.jl` mirroring the harness (lmax 16/64/128 vs `N·dP`, plus finite/bounded at 512).
- [ ] **Step 6: Stage** `src/legendre.jl test/serial/test_legendre.jl`.

---

## Task 5: Sphtor + QST kernels use `P̄` / `dP̄`

**Files:**
- Modify: `src/kernels.jl` (`_sphtor_synthesis_kernel*`, `_sphtor_analysis_kernel*` — the OTF variants and the pole branches), `src/sphtor_transforms.jl`, `src/qst_transforms.jl` call sites.

The sphtor kernels currently use `Nlm = cfg.Nlm` and `N = Nlm[l+1,col]` with raw `dPdtheta`/`P_over_sinth`. Switch the OTF path to `Plm_norm_dPdtheta_over_sinth_row!` (returns `P̄`, `dP̄/dθ`, `P̄/sinθ`) and drop the `N *` multiplies; the pole closed-forms return the `P̄`-scaled limits.

- [ ] **Step 1: Failing vector high-lmax test** in `test/serial/test_highlmax.jl`:
```julia
@testset "sphtor round-trip lmax 256" begin
    lmax=256; cfg=create_gauss_config(lmax,lmax+2;nlon=2*lmax+1)
    S0=zeros(ComplexF64,lmax+1,lmax+1); T0=zeros(ComplexF64,lmax+1,lmax+1)
    for m in 0:lmax, l in max(1,m):lmax; sc=1/(1+l)^2; S0[l+1,m+1]=complex(sc, m==0 ? 0.0 : 0.5sc); T0[l+1,m+1]=complex(0.7sc, m==0 ? 0.0 : -0.3sc); end
    Vt,Vp = SHTnsKit.synthesis_sphtor(cfg,S0,T0;real_output=true)
    @test all(isfinite,Vt) && all(isfinite,Vp)
    S,T = SHTnsKit.analysis_sphtor(cfg,Vt,Vp)
    @test isapprox(S,S0;rtol=1e-8,atol=1e-10) && isapprox(T,T0;rtol=1e-8,atol=1e-10)
end
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Rewire sphtor OTF kernels + pole branches** to `dP̄`/`P̄/sinθ`, dropping `N*`.
- [ ] **Step 4: Run → PASS;** golden vector regression at lmax ≤ 150 (extend the golden file with a sphtor case in Task 0 if not present, else add a serial reference here) → Δ < 1e-9.
- [ ] **Step 5: Run** `test/serial/test_vector_transforms.jl`, `test/serial/test_sphtor_extended.jl`, `test/serial/test_qst_transforms.jl` → PASS.
- [ ] **Step 6: Stage** the modified files + test.

---

## Task 6: Migrate remaining serial call sites

**Files:** `src/batch_transforms.jl` (4 `Plm_row!`), `src/complex_packed.jl` (3), `src/local.jl` (rows + pole), `src/api_compat.jl` (2), `src/core_transforms.jl` (adjoint path `Plm_row!` at ~line 491 area).

- [ ] **Step 1:** For each call site, replace `Plm_row!` → `Plm_norm_row!` (and the derivative variants), and delete the accompanying `Nlm[...]*` multiply in that site's accumulation loop. One file per sub-step.
- [ ] **Step 2:** After each file, run its test (`test/serial/test_batch_transforms.jl`, `test_complex_packed.jl`, `test_local.jl`, `test_api_compat.jl`) + the golden regression → PASS / Δ < 1e-9.
- [ ] **Step 3: Run the full serial suite** `test/serial/runtests.jl` (or each file) → PASS, including `test_highlmax.jl`.
- [ ] **Step 4: Stage** all modified files.

---

## Task 7: Migrate extension call sites

**Files:** `ext/ParallelTransforms.jl`, `ext/ParallelLocal.jl`, `ext/SHTnsKitLoopVecExt.jl`, `ext/SHTnsKitAdvancedADExt.jl`, `ext/SHTnsKitGPUExt.jl`.

- [ ] **Step 1: Parallel (`ParallelTransforms.jl`, `ParallelLocal.jl`).** Replace the ~10 `Plm_row!`/derivative call sites + drop `Nlm` multiplies (the distributed analysis loops `_analysis_loop_no_tables!`, sphtor/qst loops, local eval). Build/keep the env that `dev`s SHTnsKit + adds MPI/PencilArrays/PencilFFTs. Verify with the targeted MPI round-trip at lmax 64 AND 256 (θ-decomposition), 2 ranks, via `mpiexec()` wrapper: dist round-trip Δ < 1e-8 and finite. (Pre-fix, the distributed path also NaNs at lmax≥151.)
- [ ] **Step 2: LoopVec (`SHTnsKitLoopVecExt.jl`).** 4 `Plm_row!` → `Plm_norm_row!`, drop `Nlm`. Verify with LoopVectorization loaded: scalar round-trip lmax 256 finite + matches non-loopvec.
- [ ] **Step 3: AD (`SHTnsKitAdvancedADExt.jl`).** The `Plm_dPdtheta_over_sinth_row!` call → normalized variant. Verify the recurrence runs on the AD element type (the analytical coefficients are `Float64` constants; `sqrt` of constants is fine; `x` may be `Dual`). Run the AD/gradient tests (`test/serial/test_gradients.jl`) at a modest lmax → PASS, finite.
- [ ] **Step 4: GPU (`SHTnsKitGPUExt.jl`).** Inspect whether the GPU path computes its own Legendre (device kernels) or reuses host tables. If it has its own recurrence, port the same normalized recurrence (it is overflow-prone on GPU too). If CUDA is unavailable on this machine, mark as "code-reviewed + ported, hardware-verify on a GPU node" and add a CPU-fallback correctness check if the ext supports it.
- [ ] **Step 5: Stage** the modified ext files.

---

## Task 8: Other norms, CS-phase, and end-to-end suite

**Files:** none expected (norm conversion is in `convert_alm_norm!`, untouched). Verification only.

- [ ] **Step 1: Norm/CS round-trips at high lmax.**

Create `/tmp/legfix_norms.jl`:
```julia
using SHTnsKit
for norm in (:orthonormal, :fourpi, :schmidt), cs in (true, false)
    lmax=256; cfg=create_gauss_config(lmax,lmax+2;nlon=2*lmax+1, norm=norm, cs_phase=cs)
    alm0=zeros(ComplexF64,lmax+1,lmax+1)
    for m in 0:lmax,l in m:lmax; s=1/(1+l)^2; alm0[l+1,m+1]= m==0 ? complex(s) : complex(s,0.5s); end
    f=SHTnsKit.synthesis(cfg,alm0;real_output=true); a=SHTnsKit.analysis(cfg,f)
    ok = all(isfinite,f) && isapprox(a,alm0;rtol=1e-8,atol=1e-10)
    println("norm=$norm cs=$cs lmax=256 -> ", ok ? "OK" : "FAIL  Δ=$(maximum(abs,a.-alm0))")
    @assert ok
end
println("ALL NORMS OK")
```
Run: `$JL --project=. /tmp/legfix_norms.jl` → Expected: `ALL NORMS OK`. (Confirm `create_gauss_config` accepts `norm`/`cs_phase` kwargs; if the API differs, set them via the documented path.)

- [ ] **Step 2: JET type-stability** — `Plm_norm_row!` must be type-stable (it is straight-line `Float64` arithmetic). Run `test/test_jet.jl` in an env with JET → 30/30 (no new instabilities).
- [ ] **Step 3: Full MPI suite** `test/test_mpi_pencil.jl` at 2 ranks (after Task 7) → "All tests PASSED!" including a high-lmax case if added.
- [ ] **Step 4: Stage** any test additions.

---

## Task 9: Remove dead code, update docs/benchmark, finalize

**Files:** `src/legendre.jl`, `docs/src/*`, `GeoDynamo.jl/scripts/sht_scaling_benchmark.jl`, memory note.

- [ ] **Step 1:** Confirm no remaining callers of the old un-normalized `Plm_row!`/derivative functions: `grep -rn "Plm_row!\|Plm_and_dPdtheta_row!\|Plm_dPdtheta_over_sinth_row!\|Plm_over_sinth_row!\|Plm_and_dPdx_row!" src/ ext/ | grep -v norm`. Expected: empty (except defs).
- [ ] **Step 2:** Delete the now-unused un-normalized row functions from `src/legendre.jl` (or keep one, `Plm_row!`, marked deprecated if it is part of the public API — check `src/SHTnsKit.jl` exports). Run full serial suite → PASS.
- [ ] **Step 3:** Remove the "lmax ≥ 151 NaN" caveat + the default lmax cap in `GeoDynamo.jl/scripts/sht_scaling_benchmark.jl` (raise default `SHT_LMAX_LIST` to include 256, 512) and any SHTnsKit docs that mention the ceiling.
- [ ] **Step 4:** Update memory `plm-overflow-lmax150.md` → mark RESOLVED with the commit/PR ref.
- [ ] **Step 5: Stage everything; request user permission to commit / open PR.**

---

## Self-Review Notes

- **Spec coverage:** overflow root cause (Task 1), scalar OTF unblock (Task 2), tables (Task 3), vector θ-derivatives (Tasks 4–5), all serial + ext call sites (Tasks 6–7), 7 norms + CS via untouched `convert_alm_norm!` (Task 8), cleanup/docs (Task 9). ✔
- **Key risk — derivative recurrences (Task 4):** the `dP̄/dθ` and `P̄/sinθ` normalized forms are given as standard formulas but are *gated by a standalone validation harness* (Task 4 Step 1) that must match the current `N·dP` to 1e-7 at lmax≤128 before integration. If the formula needs a sign/normalization tweak to match SHTnsKit's exact convention, that is discovered there, not in the transforms.
- **Convention anchor:** `P̄_0^0 = sqrt(1/4π) = cfg.Nlm[1,1]` (verified). All higher (l,m) follow analytically; matched current `Nlm·P` to 2e-12 at lmax≤64.
- **Determinism:** kernels lose the `Nlm[l]·` multiply uniformly; `cfg.Nlm` stays a public field (built orthonormal) for back-compat and for the `convert_alm_norm!` scale matrix derivation — confirm nothing else depends on the raw un-normalized `P`.
