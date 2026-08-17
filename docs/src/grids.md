## Grid Types

SHTnsKit supports several latitude grids through its configuration constructors:

- Gauss (Gaussian quadrature): Exact for integrals up to degree `2*nlat-1`.
  Use `create_gauss_config`; suggested `nlat = lmax+1` and
  `nlon ≥ 2*mmax+1`.

- Regular equiangular without poles (reg_fast/reg_dct/quick_init):
  Midpoint latitudes `θ_i = (i+0.5)π/nlat`. Fast to set up and compatible with
  FFT-friendly sampling. Use `create_regular_config(...; include_poles=false)`.

- Regular equiangular with poles (reg_poles):
  `θ_i = i π/(nlat-1)` including poles. Use
  `create_regular_config(...; include_poles=true)`.

`create_config` provides a common entry point through its `grid_type` keyword.
For best numerical exactness, prefer Gauss. For image-like sampling, use a
regular grid with precomputed Legendre tables.
