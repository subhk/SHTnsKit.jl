# Grid Types

SHTnsKit uses equally spaced longitudes for every grid. The important choice is
how points are placed in colatitude ``\theta`` from the north pole
(``\theta = 0``) to the south pole (``\theta = \pi``). The comparison below
uses the same 12 latitude samples and 16 longitudes in every panel, so the
different latitude patterns are directly comparable.

```@raw html
<figure class="grid-pattern-figure">
  <picture>
    <source media="(max-width: 700px)"
            srcset="assets/grid-patterns-stacked.svg">
    <img src="assets/grid-patterns.svg"
         alt="Four globes comparing Gauss–Legendre, regular midpoint, regular with poles, and Driscoll–Healy sampling grids.">
  </picture>
  <figcaption>Faint dots lie on the far side of each globe. Driscoll–Healy dot size reflects quadrature weight.</figcaption>
</figure>
```

At a pole, all longitudes describe the same physical point, so their markers
overlap in the figure.

## Compare the sampling patterns

| Grid | Colatitudes | Pole samples | Good choice when… |
|:-----|:-------------|:-------------|:------------------|
| **Gauss–Legendre** | Roots of the degree-`nlat` Legendre polynomial | Neither | You want the default grid and the most accurate quadrature for a given number of latitudes. |
| **Regular midpoint** | ``\theta_i = (i + 1/2)\pi/n_\mathrm{lat}`` | Neither | Your data are cell-centred or already live on an equiangular image-like grid. |
| **Regular with poles** | ``\theta_i = i\pi/(n_\mathrm{lat}-1)`` | Both | Your external format requires explicit values at both poles. |
| **Driscoll–Healy** | ``\theta_i = i\pi/n_\mathrm{lat}`` | North only | You want equiangular sampling with Driscoll–Healy quadrature for band-limited transforms. |

Gauss–Legendre quadrature with `nlat` points integrates polynomials in
``x = \cos\theta`` through degree `2*nlat - 1` exactly. Driscoll–Healy
quadrature is exact at the intended band limit when `nlat = 2*(lmax + 1)`;
`nlat` must be even.

## Create each grid

The common [`create_config`](@ref) entry point makes it easy to switch grid
types without changing the rest of a program:

```julia
using SHTnsKit

lmax = 5
nlat = 2 * (lmax + 1)  # 12: exact Driscoll–Healy sampling for this lmax
nlon = 16              # must be at least 2*lmax + 1

gauss = create_config(lmax; nlat, nlon, grid_type=:gauss)
midpoint = create_config(lmax; nlat, nlon, grid_type=:regular)
with_poles = create_config(lmax; nlat, nlon, grid_type=:regular_poles)
dh = create_config(lmax; nlat, nlon, grid_type=:driscoll_healy)
```

The lower-level constructors are also available when their intent reads more
clearly in an application:

```julia
gauss = create_gauss_config(lmax, nlat; nlon)
midpoint = create_regular_config(lmax, nlat; nlon, include_poles=false)
with_poles = create_regular_config(lmax, nlat; nlon, include_poles=true)
dh = create_regular_config(
    lmax, nlat; nlon, include_poles=true, use_dh_weights=true,
)
```

!!! tip "Which grid should I choose?"
    Start with Gauss–Legendre unless you need to exchange data with a particular
    equiangular layout. Use regular midpoint for cell-centred data, regular with
    poles for formats that store both endpoints, and Driscoll–Healy when its
    sampling theorem and quadrature weights are part of your numerical method.

Regular and Driscoll–Healy configurations precompute Legendre tables by default.
For a memory-saving setup, pass `precompute_plm=false` and evaluate the
polynomials on demand.
