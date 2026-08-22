module GridPatternPlots

using Plots
using SHTnsKit

export generate_grid_patterns, grid_specs, projected_points

const LMAX = 5
const NLAT = 2 * (LMAX + 1)
const NLON = 16
const CAMERA_AZIMUTH = deg2rad(-34.0)
const CAMERA_ELEVATION = deg2rad(22.0)

"""Return the four documented grids with identical sampling dimensions."""
function grid_specs()
    common = (; nlat=NLAT, nlon=NLON, precompute_plm=false)
    return [
        (; key=:gauss, title="Gauss–Legendre",
         detail="nonuniform rings · no poles", color="#2563EB",
         config=create_config(LMAX; common..., grid_type=:gauss)),
        (; key=:regular, title="Regular midpoint",
         detail="uniform rings · no poles", color="#7C3AED",
         config=create_config(LMAX; common..., grid_type=:regular)),
        (; key=:regular_poles, title="Regular with poles",
         detail="uniform rings · both poles", color="#0891B2",
         config=create_config(LMAX; common..., grid_type=:regular_poles)),
        (; key=:driscoll_healy, title="Driscoll–Healy",
         detail="uniform rings · north pole only", color="#EA580C",
         config=create_config(LMAX; common..., grid_type=:driscoll_healy)),
    ]
end

@inline function _project_point(θ::Real, φ::Real)
    x = sin(θ) * cos(φ)
    y = sin(θ) * sin(φ)
    z = cos(θ)

    ca, sa = cos(CAMERA_AZIMUTH), sin(CAMERA_AZIMUTH)
    ce, se = cos(CAMERA_ELEVATION), sin(CAMERA_ELEVATION)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    yp = ce * z - se * yr
    depth = se * z + ce * yr
    return xr, yp, depth
end

"""Orthographically project every public `(θ, φ)` sample in `cfg`."""
function projected_points(cfg::SHTConfig)
    npoints = cfg.nlat * cfg.nlon
    xs = Vector{Float64}(undef, npoints)
    ys = similar(xs)
    depths = similar(xs)
    weights = similar(xs)

    k = 1
    for (i, θ) in pairs(cfg.θ), φ in cfg.φ
        xs[k], ys[k], depths[k] = _project_point(θ, φ)
        weights[k] = cfg.w[i]
        k += 1
    end

    return (; x=xs, y=ys, depth=depths, front=depths .>= 0, weight=weights)
end

function _project_curve(θs, φs)
    n = length(θs)
    xs = Vector{Float64}(undef, n)
    ys = similar(xs)
    depths = similar(xs)
    for i in eachindex(θs, φs)
        xs[i], ys[i], depths[i] = _project_point(θs[i], φs[i])
    end
    return (; x=xs, y=ys, depth=depths)
end

function _masked(values, keep)
    result = copy(values)
    result[.!keep] .= NaN
    return result
end

function _guide_curves()
    t = collect(range(0, 2π; length=361))
    return (
        _project_curve(fill(π / 2, length(t)), t),
        _project_curve(t, fill(0.0, length(t))),
        _project_curve(t, fill(π / 2, length(t))),
    )
end

function _marker_sizes(spec, points)
    spec.key === :driscoll_healy || return fill(3.7, length(points.x))
    maximum_weight = maximum(points.weight)
    return 2.5 .+ 2.5 .* sqrt.(max.(points.weight, 0) ./ maximum_weight)
end

function _grid_panel(spec)
    p = plot(
        ; aspect_ratio=:equal,
        xlims=(-1.08, 1.08),
        ylims=(-1.18, 1.08),
        axis=false,
        ticks=false,
        grid=false,
        framestyle=:none,
        legend=false,
        background_color="#FFFFFF",
        title=spec.title,
        titlefontsize=15,
        titlefontcolor="#0F172A",
    )

    outline_t = range(0, 2π; length=361)
    globe = Shape(cos.(outline_t), sin.(outline_t))
    plot!(p, globe; color="#F8FAFC", linecolor="#CBD5E1", linewidth=1.4, label=false)

    for curve in _guide_curves()
        back = curve.depth .< 0
        front = .!back
        plot!(p, _masked(curve.x, back), _masked(curve.y, back);
              color="#CBD5E1", alpha=0.38, linewidth=0.7, linestyle=:dash,
              label=false)
        plot!(p, _masked(curve.x, front), _masked(curve.y, front);
              color="#94A3B8", alpha=0.62, linewidth=0.8, label=false)
    end

    points = projected_points(spec.config)
    sizes = _marker_sizes(spec, points)
    back = .!points.front
    scatter!(p, points.x[back], points.y[back];
             color=spec.color, markersize=sizes[back], markeralpha=0.22,
             markerstrokewidth=0, label=false)
    scatter!(p, points.x[points.front], points.y[points.front];
             color=spec.color, markersize=sizes[points.front], markeralpha=0.94,
             markerstrokecolor="#FFFFFF", markerstrokewidth=0.45, label=false)

    plot!(p, cos.(outline_t), sin.(outline_t);
          color="#64748B", linewidth=1.25, label=false)
    annotate!(p, 0, -1.115, text(spec.detail, 9, "#475569", :center))
    return p
end

function _add_accessibility_metadata!(path::AbstractString)
    svg = read(path, String)
    svg_range = findfirst("<svg", svg)
    svg_range === nothing && error("Plots did not produce an SVG root element")
    tag_end = findnext('>', svg, first(svg_range))
    tag_end === nothing && error("Plots produced an incomplete SVG root element")

    opening_tag = String(SubString(svg, firstindex(svg), tag_end))
    opening_tag = replace(
        opening_tag,
        "<svg" => "<svg role=\"img\" aria-labelledby=\"grid-patterns-title grid-patterns-description\"";
        count=1,
    )
    metadata = """

<title id="grid-patterns-title">SHTnsKit spherical sampling grids</title>
<desc id="grid-patterns-description">Four globes compare Gauss–Legendre, regular midpoint, regular with poles, and Driscoll–Healy sample locations.</desc>"""
    remainder_start = nextind(svg, tag_end)
    updated = string(opening_tag, metadata, SubString(svg, remainder_start))
    write(path, updated)
    return path
end

"""Generate the responsive four-panel grid comparison as an SVG asset."""
function generate_grid_patterns(output::AbstractString)
    gr()
    panels = _grid_panel.(grid_specs())
    figure = plot(
        panels...;
        layout=(2, 2),
        size=(1120, 940),
        background_color="#F8FAFC",
        plot_title="SHTnsKit spherical sampling grids",
        plot_titlefontsize=20,
        plot_titlefontcolor="#0F172A",
    )

    mkpath(dirname(output))
    savefig(figure, output)
    _add_accessibility_metadata!(output)
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_grid_patterns(joinpath(
        @__DIR__, "..", "src", "assets", "grid-patterns.svg",
    ))
end

end
