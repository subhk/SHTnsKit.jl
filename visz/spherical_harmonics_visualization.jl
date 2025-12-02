### A Pluto.jl notebook ###
# v0.19.38

#> [frontmatter]
#> title = "Spherical Harmonic Visualization"
#> description = "Interactive visualization of spherical harmonics Y_l^m using SHTnsKit.jl"

using Markdown
using InteractiveUtils

# ╔═╡ Cell order:
# ╟─notebook_header
# ╟─introduction
# ╠═setup_packages
# ╠═helper_functions
# ╟─y10_section
# ╠═y10_plot
# ╟─y11_section
# ╠═y11_plot
# ╟─y21_section
# ╠═y21_plot
# ╟─y22_section
# ╠═y22_plot
# ╟─combined_view
# ╠═all_plots

# ╔═╡ notebook_header ═╡
md"""
# Spherical Harmonic Visualization with SHTnsKit.jl

This notebook demonstrates the visualization of spherical harmonics using SHTnsKit.jl and CairoMakie.
We'll visualize several important spherical harmonic modes to understand their spatial structure.
"""

# ╔═╡ introduction ═╡
md"""
## What are Spherical Harmonics?

Spherical harmonics ``Y_\ell^m(\theta, \phi)`` are the angular part of solutions to Laplace's equation
in spherical coordinates. They form a complete orthonormal basis on the sphere, making them essential
for representing functions on spherical surfaces.

### Notation
- ``\ell`` (degree): Controls the overall complexity/scale
- ``m`` (order): Controls the azimuthal wave number
- ``\theta`` (colatitude): Angle from north pole [0, π]
- ``\phi`` (longitude): Azimuthal angle [0, 2π)

We'll visualize:
- ``Y_1^0``: Zonal mode (varies only with latitude)
- ``Y_1^1``: Sectoral mode (tilted dipole pattern)
- ``Y_2^1``: Tesseral mode (mixed pattern)
- ``Y_2^2``: Sectoral quadrupole mode
"""

# ╔═╡ setup_packages ═╡
begin
    using SHTnsKit
    using CairoMakie
    using Printf

    CairoMakie.activate!(type = "png")

    println("Packages loaded successfully!")
end

# ╔═╡ helper_functions ═╡
begin
    """
        create_spherical_harmonic(lmax, l, m; nlat=128, nlon=256)

    Create a spatial field containing a single spherical harmonic Y_l^m.
    """
    function create_spherical_harmonic(lmax, l, m; nlat=128, nlon=256)
        # Create Gauss-Legendre grid configuration
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Initialize coefficients to zero
        alm = zeros(ComplexF64, lmax+1, lmax+1)

        # Set single coefficient for Y_l^m
        # Convention: alm[l+1, m+1] for degree l, order m
        alm[l+1, m+1] = 1.0 + 0.0im

        # Synthesize to spatial domain
        f = synthesis(cfg, alm; real_output=true)

        return f, cfg
    end

    """
        plot_spherical_harmonic(f, cfg, l, m; title="")

    Create a beautiful contour plot of a spherical harmonic on the sphere.
    """
    function plot_spherical_harmonic(f, cfg, l, m; title="",
                                     colormap=:RdBu, levels=20)
        # Create figure with good resolution
        fig = Figure(resolution = (800, 500), fontsize = 14)
        ax = Axis(fig[1, 1],
                  xlabel = "Longitude φ (degrees)",
                  ylabel = "Latitude (degrees)",
                  title = title,
                  aspect = 2.0)

        # Convert angles to degrees for better readability
        lons_deg = rad2deg.(cfg.φ)
        lats_deg = 90 .- rad2deg.(cfg.θ)  # Convert colatitude to latitude

        # Create contour plot
        cf = contourf!(ax, lons_deg, lats_deg, f',
                       levels = levels,
                       colormap = colormap,
                       extendlow = :auto,
                       extendhigh = :auto)

        # Add colorbar
        Colorbar(fig[1, 2], cf, label = "Amplitude")

        # Add gridlines
        hlines!(ax, [-60, -30, 0, 30, 60], color = :black,
                linewidth = 0.5, alpha = 0.3, linestyle = :dash)
        vlines!(ax, [0, 90, 180, 270, 360], color = :black,
                linewidth = 0.5, alpha = 0.3, linestyle = :dash)

        # Stats annotation
        max_val = maximum(abs.(f))
        min_val = minimum(f)
        max_pos = maximum(f)

        text!(ax, 10, -80,
              text = @sprintf("Range: [%.3f, %.3f]\nMax |value|: %.3f",
                              min_val, max_pos, max_val),
              fontsize = 10, align = (:left, :bottom),
              color = :black, backgroundcolor = (:white, 0.7))

        return fig
    end

    """
        plot_multiple_harmonics(harmonics_data; title="")

    Create a 2x2 grid showing multiple spherical harmonics.
    """
    function plot_multiple_harmonics(harmonics_data;
                                      title="Spherical Harmonics Overview",
                                      colormap=:RdBu, levels=20)
        fig = Figure(resolution = (1400, 1000), fontsize = 12)

        # Main title
        Label(fig[0, :], title, fontsize = 20, font = :bold)

        for (idx, (f, cfg, l, m, subtitle)) in enumerate(harmonics_data)
            row = (idx - 1) ÷ 2 + 1
            col = (idx - 1) % 2 + 1

            ax = Axis(fig[row, 2*col-1],
                     xlabel = "Longitude φ (°)",
                     ylabel = "Latitude (°)",
                     title = subtitle,
                     aspect = 2.0)

            lons_deg = rad2deg.(cfg.φ)
            lats_deg = 90 .- rad2deg.(cfg.θ)

            cf = contourf!(ax, lons_deg, lats_deg, f',
                          levels = levels,
                          colormap = colormap)

            Colorbar(fig[row, 2*col], cf, label = "Amplitude")

            # Add gridlines
            hlines!(ax, [-60, -30, 0, 30, 60], color = :black,
                   linewidth = 0.5, alpha = 0.2)
            vlines!(ax, [0, 90, 180, 270], color = :black,
                   linewidth = 0.5, alpha = 0.2)
        end

        return fig
    end

    println("Helper functions defined!")
end

# ╔═╡ y10_section ═╡
md"""
## Y₁⁰: Zonal Dipole (l=1, m=0)

This is the simplest non-constant spherical harmonic. It represents a **zonal** pattern
(varies only with latitude, not longitude). This mode is proportional to cos(θ),
so it's positive in the northern hemisphere and negative in the southern hemisphere.

**Physical interpretation**: Climate zones, dipole magnetic field component
"""

# ╔═╡ y10_plot ═╡
begin
    # Create Y_1^0
    f_10, cfg_10 = create_spherical_harmonic(4, 1, 0; nlat=128, nlon=256)

    # Plot
    fig_10 = plot_spherical_harmonic(f_10, cfg_10, 1, 0;
                                     title = "Spherical Harmonic Y₁⁰ (Zonal Dipole)",
                                     colormap = :balance)

    fig_10
end

# ╔═╡ y11_section ═╡
md"""
## Y₁¹: Sectoral Dipole (l=1, m=1)

This is a **sectoral** mode where |m| = l. It creates a pattern that varies with both
latitude and longitude, showing one full wavelength around the equator.

**Physical interpretation**: Tilted dipole field, first-order wave pattern
"""

# ╔═╡ y11_plot ═╡
begin
    # Create Y_1^1
    f_11, cfg_11 = create_spherical_harmonic(4, 1, 1; nlat=128, nlon=256)

    # Plot
    fig_11 = plot_spherical_harmonic(f_11, cfg_11, 1, 1;
                                     title = "Spherical Harmonic Y₁¹ (Sectoral Dipole)",
                                     colormap = :RdBu)

    fig_11
end

# ╔═╡ y21_section ═╡
md"""
## Y₂¹: Tesseral Mode (l=2, m=1)

This is a **tesseral** mode where 0 < m < l. These modes have mixed
latitude-longitude patterns that are neither purely zonal nor purely sectoral.

**Physical interpretation**: Complex atmospheric or oceanic wave patterns
"""

# ╔═╡ y21_plot ═╡
begin
    # Create Y_2^1
    f_21, cfg_21 = create_spherical_harmonic(4, 2, 1; nlat=128, nlon=256)

    # Plot
    fig_21 = plot_spherical_harmonic(f_21, cfg_21, 2, 1;
                                     title = "Spherical Harmonic Y₂¹ (Tesseral Mode)",
                                     colormap = :RdBu)

    fig_21
end

# ╔═╡ y22_section ═╡
md"""
## Y₂²: Sectoral Quadrupole (l=2, m=2)

Another sectoral mode (|m| = l) but at degree 2, showing a quadrupole pattern
with two full wavelengths around the equator.

**Physical interpretation**: Quadrupole magnetic field, higher-order oscillations
"""

# ╔═╡ y22_plot ═╡
begin
    # Create Y_2^2
    f_22, cfg_22 = create_spherical_harmonic(4, 2, 2; nlat=128, nlon=256)

    # Plot
    fig_22 = plot_spherical_harmonic(f_22, cfg_22, 2, 2;
                                     title = "Spherical Harmonic Y₂² (Sectoral Quadrupole)",
                                     colormap = :RdBu)

    fig_22
end

# ╔═╡ combined_view ═╡
md"""
## Combined View: All Four Harmonics

Here's a side-by-side comparison showing how the complexity increases with degree l
and how the azimuthal structure changes with order m.

**Key observations**:
- **Degree l**: Controls the number of nodal lines (zeros) - higher l means more structure
- **Order m**: Controls azimuthal wave number - |m| full waves around equator
- **Zonal (m=0)**: No longitudinal variation
- **Sectoral (|m|=l)**: Maximum longitudinal variation
- **Tesseral (0<m<l)**: Mixed patterns
"""

# ╔═╡ all_plots ═╡
begin
    # Collect all harmonics data
    harmonics_data = [
        (f_10, cfg_10, 1, 0, "Y₁⁰: Zonal Dipole"),
        (f_11, cfg_11, 1, 1, "Y₁¹: Sectoral Dipole"),
        (f_21, cfg_21, 2, 1, "Y₂¹: Tesseral Mode"),
        (f_22, cfg_22, 2, 2, "Y₂²: Sectoral Quadrupole")
    ]

    # Create combined plot
    fig_combined = plot_multiple_harmonics(harmonics_data;
                                           title = "Spherical Harmonics: Pattern Comparison",
                                           colormap = :RdBu,
                                           levels = 25)

    fig_combined
end
