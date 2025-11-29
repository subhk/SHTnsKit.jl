# Next Steps Summary

## Immediate Actions (Do This First!)

### 1. Debug the DH Test Failure
The test is failing with the exact same error (2.13), which suggests either:
- The DH weights aren't being applied
- The phi_scale isn't being used correctly
- There's a normalization mismatch

**Run this to diagnose**:
```bash
cd /Users/subhajitkar/Documents/GitHub/SHTnsKit.jl
VERBOSE=1 julia --project=. test/runtests.jl
```

Look for the debug output:
```
[ Info: DH Config grid_type=:driscoll_healy phi_scale=:dft sum_weights=2.828... first_weight=0.0 last_weight=0.0
```

**Expected values**:
- `grid_type` = `:driscoll_healy`
- `phi_scale` = `:dft`
- `sum_weights` ≈ 2.828 (which is 2√2)
- `first_weight` = 0.0 (north pole)
- `last_weight` ≈ 0.0 (should be near zero)

### 2. If DH Still Fails, Use Oversampling Fallback

If the DH approach continues to have issues, revert to the proven method:

```julia
# In test/runtests.jl, replace the DH test with:
lmax = 8
nlat = max(2 * (lmax + 1), 8 * lmax)  # Heavy oversampling: 64 points
nlon = 2 * (2 * lmax + 1)
cfg_reg = create_regular_config(lmax, nlat; nlon=nlon, precompute_plm=true, include_poles=false)
# ... rest of test
```

This uses many more latitude points to achieve accuracy without special quadrature.

---

## Quick Wins (1-2 Days)

### 3. Add Unit Test for DH Weights
Create `test/test_dh_weights.jl`:

```julia
using Test
using SHTnsKit

@testset "Driscoll-Healy Weights" begin
    # Test n=4
    w4 = SHTnsKit.driscoll_healy_weights(4)
    @test length(w4) == 4
    @test w4[1] ≈ 0.0 atol=1e-15  # North pole
    @test w4[end] ≈ 0.0 atol=1e-15  # Near south pole
    @test sum(w4) ≈ 2*sqrt(2) rtol=1e-10

    # Test n=18 (lmax=8 case)
    w18 = SHTnsKit.driscoll_healy_weights(18)
    @test sum(w18) ≈ 2*sqrt(2) rtol=1e-10
    @test all(w18 .>= 0.0)

    # Test error conditions
    @test_throws ArgumentError SHTnsKit.driscoll_healy_weights(1)  # n < 2
    @test_throws ArgumentError SHTnsKit.driscoll_healy_weights(5)  # odd n
end
```

### 4. Document Current Status
Update the main README with:

```markdown
## Grid Types

SHTnsKit supports multiple grid types:

- **Gauss-Legendre** (`:gauss`): Optimal for spectral accuracy, uses Gaussian quadrature
- **Regular Equiangular** (`:regular`): Uniform spacing in θ, simpler but requires oversampling
- **Regular with Poles** (`:regular_poles`): Includes both poles
- **Driscoll-Healy** (`:driscoll_healy`): Exact transforms on regular grid (experimental)

### φ-Scaling

The `phi_scale` parameter controls longitude integration:
- `:dft` - For Gauss and Driscoll-Healy grids (default for these)
- `:quad` - For regular grids (default for these)
- `:auto` - Automatic selection based on grid type

See `docs/phi_scaling.md` for details.
```

---

## Important Improvements (1 Week)

### 5. Refactor Configuration System

The current system has too many parameters. Consider:

```julia
# Proposed new API (keep old for backward compat)
struct GridSpec
    type::Symbol  # :gauss, :regular, :driscoll_healy
    nlat::Int
    nlon::Int
    include_poles::Bool
    phi_scale::Symbol  # :auto, :dft, :quad
end

function create_config(lmax::Int, grid::GridSpec; kwargs...)
    # Single unified configuration function
end

# Convenience constructors
gauss_grid(nlat, nlon) = GridSpec(:gauss, nlat, nlon, true, :dft)
regular_grid(nlat, nlon; poles=false) = GridSpec(:regular, nlat, nlon, poles, :quad)
dh_grid(lmax) = GridSpec(:driscoll_healy, 2*(lmax+1), 2*(2*lmax+1), true, :dft)
```

### 6. Add Performance Benchmarks

Create `benchmark/transforms.jl`:

```julia
using BenchmarkTools
using SHTnsKit

suite = BenchmarkGroup()

for lmax in [8, 16, 32, 64, 128]
    for grid_type in [:gauss, :regular]
        cfg = if grid_type == :gauss
            create_gauss_config(lmax, lmax+1)
        else
            create_regular_config(lmax, 8*lmax)  # Oversampled
        end

        alm = randn(ComplexF64, lmax+1, lmax+1)
        suite["synthesis"][grid_type][lmax] = @benchmarkable synthesis($cfg, $alm)

        f = synthesis(cfg, alm)
        suite["analysis"][grid_type][lmax] = @benchmarkable analysis($cfg, $f)
    end
end

results = run(suite)
```

---

## Larger Projects (1 Month+)

### 7. Stabilize API for v1.0
- Audit all exported functions
- Mark experimental features clearly
- Write deprecation warnings for old API
- Freeze public interface

### 8. Comprehensive Documentation
- Tutorial: Getting Started
- Tutorial: Choosing Grid Types
- Tutorial: Performance Optimization
- API Reference (complete)
- Theory/Math background

### 9. Community Engagement
- Submit package to Julia registry (if not already)
- Write blog post: "Fast Spherical Harmonic Transforms in Julia"
- Create examples: weather data, gravitational potential, etc.
- Engage with geophysics/astronomy communities

---

## Files to Review/Clean Up

### High Priority
1. `src/core_transforms.jl` - Large file, consider splitting
2. `test/runtests.jl` - Should be split into multiple test files
3. `docs/` - Needs organization and completion

### Technical Debt
1. Remove or document why `cphi = 2π / nlon` is hardcoded vs using `phi_inv_scale`
2. Clarify relationship between `w` (quadrature weights) and `wlat` (alias?)
3. Document all grid types in one place
4. Unify error messages and validation

---

## Resources Needed

### Documentation
- [ ] Mathematical foundation of DH quadrature
- [ ] Performance comparison: Gauss vs Regular vs DH
- [ ] Memory usage guide for large lmax

### Testing
- [ ] Test datasets with known solutions
- [ ] Cross-validation with SHTOOLS
- [ ] Accuracy vs performance benchmarks

### Community
- [ ] Contributing guide
- [ ] Code of conduct
- [ ] Issue templates
- [ ] PR templates

---

## Success Criteria

**Week 1**: All tests passing, DH working or documented why not
**Month 1**: Test coverage >80%, basic docs complete
**Quarter 1**: v1.0 release candidate, comprehensive docs
**Quarter 2**: v1.0 stable, 5+ example packages, 50+ GitHub stars
