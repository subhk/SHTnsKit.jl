# Grid Types

Grid constructors validate sampling constraints immediately. All grids store
latitude in the first spatial dimension and longitude in the second.

## Gauss–Legendre

[`create_gauss_config`](@ref) is the default choice for numerical transforms:

```@example grids-gauss
using SHTnsKit

lmax = 32
cfg = create_gauss_config(lmax, lmax + 1; nlon=2lmax + 1)
(cfg.grid_type, cfg.nlat, cfg.nlon, sum(cfg.w))
```

Constraints:

- `nlat >= lmax + 1`
- `0 <= mmax <= lmax`
- `mres >= 1`
- `nlon >= 2*mmax + 1`

`create_gauss_fly_config` uses the same nodes while keeping Legendre recurrence
on the fly. `create_gauss_config_spf` creates the south-pole-first variant.

## Regular Fejér grid

The default regular grid uses midpoint colatitudes
`theta_i = (i + 1/2)π/nlat`, excluding both poles:

```@example grids-regular
using SHTnsKit

cfg = create_regular_config(32, 34; include_poles=false)
(cfg.grid_type, first(cfg.θ) > 0, last(cfg.θ) < π)
```

It requires `nlat >= lmax + 2` and `nlon >= 2*mmax + 1`. Associated Legendre
tables are precomputed by default; pass `precompute_plm=false` to favor a lower
initial memory footprint.

## Pole-inclusive regular grid

```@example grids-poles
using SHTnsKit

cfg = create_regular_config(32, 33; include_poles=true)
(first(cfg.θ), last(cfg.θ))
```

This grid includes both endpoints, requires `nlat >= lmax + 1`, and always
requires at least two latitudes. Pole-safe scalar and vector recurrences handle
the endpoint rows.

## Driscoll–Healy weights

Enable Driscoll–Healy quadrature on a pole grid:

```@example grids-dh
using SHTnsKit

lmax = 15
nlat = 2 * (lmax + 1)
cfg = create_regular_config(
    lmax, nlat;
    include_poles=true,
    use_dh_weights=true,
)
(cfg.grid_type, cfg.nlat)
```

`use_dh_weights=true` requires `include_poles=true` and an even `nlat`. Exact
sampling uses `nlat == 2*(lmax + 1)`; another even value is accepted with a
warning because it no longer has that exactness guarantee. The DH nodes include
the north pole and exclude the south pole.

## Common constructor

[`create_config`](@ref) selects a grid through `grid_type` and forwards the
normalization and sampling options. Prefer the specific constructors in user
code when the chosen grid should be obvious at the call site.

## Latitude order

New configurations are north-pole-first. Use `set_south_pole_first!` or
`set_north_pole_first!` before allocating/filling fields when an external data
source uses the opposite order. Query with `is_south_pole_first(cfg)`.

Do not reverse only the field: the field, nodes, quadrature weights, and tables
must use one consistent ordering.
