# Grid Pattern Plots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a polished, sphere-only comparison of every supported SHTnsKit grid to the deployed web documentation.

**Architecture:** A docs-only Julia module will construct four real `SHTConfig` objects, orthographically project their spherical sample points, and render deterministic 2-by-2 desktop and one-column mobile SVGs with Plots.jl. The committed assets are selected by a responsive `picture` in `grids.md`, so Documenter never has to generate graphics during a normal build; a focused docs test verifies geometry, generation, accessibility metadata, and page wiring.

**Tech Stack:** Julia 1.10+, SHTnsKit public configuration API, Plots.jl/GR SVG output, Documenter.jl, Julia `Test`, CSS.

---

### Task 1: Specify the grid plot contract with a failing test

**Files:**
- Create: `docs/test/test_grid_patterns.jl`
- Create later: `docs/scripts/generate_grid_patterns.jl`

**Step 1: Write the failing test**

Create a focused test that includes the future generator module and checks the actual grid semantics, projected point count, output metadata, and docs wiring:

```julia
using Test

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "docs", "scripts", "generate_grid_patterns.jl"))
using .GridPatternPlots

@testset "grid-pattern documentation figure" begin
    specs = grid_specs()
    @test getproperty.(specs, :key) ==
          [:gauss, :regular, :regular_poles, :driscoll_healy]
    @test getproperty.(getproperty.(specs, :config), :grid_type) ==
          [:gauss, :regular, :regular_poles, :driscoll_healy]

    by_key = Dict(spec.key => spec for spec in specs)
    @test first(by_key[:gauss].config.θ) > 0
    @test last(by_key[:gauss].config.θ) < π
    @test first(by_key[:regular].config.θ) > 0
    @test last(by_key[:regular].config.θ) < π
    @test first(by_key[:regular_poles].config.θ) == 0
    @test last(by_key[:regular_poles].config.θ) ≈ π
    @test first(by_key[:driscoll_healy].config.θ) == 0
    @test last(by_key[:driscoll_healy].config.θ) < π

    for spec in specs
        points = projected_points(spec.config)
        @test length(points.x) == spec.config.nlat * spec.config.nlon
        @test all(points.x .^ 2 .+ points.y .^ 2 .<= 1 + 100eps())
        @test count(points.front) > 0
        @test count(.!points.front) > 0
    end

    mktempdir() do dir
        output = joinpath(dir, "grid-patterns.svg")
        @test generate_grid_patterns(output) == output
        @test isfile(output)
        svg = read(output, String)
        @test occursin("<svg", svg)
        @test occursin("<title id=\"grid-patterns-title\">", svg)
        @test occursin("<desc id=\"grid-patterns-description\">", svg)
        @test filesize(output) > 10_000
    end
end

@testset "grid page wiring" begin
    page = read(joinpath(ROOT, "docs", "src", "grids.md"), String)
    makefile = read(joinpath(ROOT, "docs", "make.jl"), String)
    @test occursin("assets/grid-patterns.svg", page)
    @test occursin("Grid Types\" => \"grids.md", makefile)
end
```

**Step 2: Run the test to verify it fails**

Run:

```bash
julia --project=docs docs/test/test_grid_patterns.jl
```

Expected: `LoadError` because `docs/scripts/generate_grid_patterns.jl` does not exist.

### Task 2: Implement the plot generator from real configurations

**Files:**
- Create: `docs/scripts/generate_grid_patterns.jl`
- Test: `docs/test/test_grid_patterns.jl`

**Step 1: Add the minimal generator module**

Implement `module GridPatternPlots` with these constants and public helpers:

```julia
module GridPatternPlots

using Plots
using SHTnsKit

export generate_grid_patterns, grid_specs, projected_points

const LMAX = 5
const NLAT = 2 * (LMAX + 1)
const NLON = 16

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
```

`projected_points(cfg)` must iterate over the public `cfg.θ`, `cfg.φ`, and
`cfg.w` values, convert each pair to Cartesian coordinates, rotate every point
with a fixed azimuth/elevation, and return vectors `x`, `y`, `depth`, `front`,
and `weight`. Keep the projection pure so it can be tested without rendering.

**Step 2: Render one consistent globe card per spec**

Use a 2D orthographic globe rather than backend-dependent 3D axes:

- Draw a pale unit-circle sphere, an outline, and three faint great-circle guides.
- Draw rear points first at low opacity and front points second at high opacity.
- Use identical bounds, camera, latitude/longitude counts, and point ordering.
- Keep fixed marker sizes for the first three grids; scale Driscoll–Healy markers
  gently by `cfg.w / maximum(cfg.w)` to reveal its quadrature weighting while
  retaining a sphere-only visualization.
- Put `title` above each panel and `detail` directly beneath it.
- Combine four panels with `layout=(2, 2)`, `size=(1120, 940)`, a light neutral
  card background, and a single plot title.

`generate_grid_patterns(output)` must create the parent directory, select GR,
save SVG, then insert accessible `<title>` and `<desc>` elements immediately
inside the root `<svg>` tag. Return `output`. Guard the command-line entry point:

```julia
if abspath(PROGRAM_FILE) == @__FILE__
    generate_grid_patterns(joinpath(@__DIR__, "..", "src", "assets",
                                    "grid-patterns.svg"))
end

end
```

**Step 3: Run the focused test and observe the remaining expected failure**

Run: `julia --project=docs docs/test/test_grid_patterns.jl`

Expected: generator and geometry tests pass; `grid page wiring` fails because the
page does not yet embed the asset and `docs/make.jl` does not list the page.

**Step 4: Generate the committed asset**

Run: `julia --project=docs docs/scripts/generate_grid_patterns.jl`

Expected: `docs/src/assets/grid-patterns.svg` exists, contains four panels, and
has the accessible title and description.

**Step 5: Commit the generator, test, and SVG**

```bash
git add docs/scripts/generate_grid_patterns.jl docs/test/test_grid_patterns.jl docs/src/assets/grid-patterns.svg
git commit -m "docs: generate spherical grid comparison"
```

### Task 3: Integrate the comparison into the web documentation

**Files:**
- Modify: `docs/src/grids.md`
- Modify: `docs/src/assets/custom.css`
- Modify: `docs/make.jl`
- Modify: `docs/plans/2026-08-22-grid-pattern-plots-design.md`
- Test: `docs/test/test_grid_patterns.jl`

**Step 1: Replace the terse grid page with visual guidance**

Give `docs/src/grids.md` an H1 title, a concise orientation paragraph, the SVG
inside an accessible figure, and a four-row comparison table. The image markup
must use this stable path and useful alt text:

```html
<figure class="grid-pattern-figure">
  <img src="assets/grid-patterns.svg"
       alt="Four globes comparing Gauss–Legendre, regular midpoint, regular with poles, and Driscoll–Healy sampling grids.">
  <figcaption>All panels use 12 latitude rings and 16 longitudes. Faint dots are on the far side of each globe; Driscoll–Healy dot size reflects quadrature weight.</figcaption>
</figure>
```

Document the actual geometry accurately: regular-poles includes both poles,
whereas the implemented Driscoll–Healy convention samples `θⱼ = πj/nlat` and
includes the north pole but excludes the south pole. Provide runnable
`create_config(...; grid_type=...)` examples for all four grids and a compact
selection guide that recommends Gauss by default.

**Step 2: Add responsive figure styling**

Append `.grid-pattern-figure` rules to `docs/src/assets/custom.css`: centered
layout, subtle border/radius/shadow, responsive `img { width: 100%; height: auto; }`,
and a muted readable caption. Include a dark-theme-safe background/border rule
using Documenter's theme selectors already present in the stylesheet.

**Step 3: Expose the page in navigation**

Add `"Grid Types" => "grids.md"` in the `User Guide` list in `docs/make.jl`,
before the performance pages.

**Step 4: Correct the approved design note**

Update the design note's Driscoll–Healy wording to reflect the real package
geometry (north pole included, south pole excluded) rather than saying it shares
all regular-poles locations. Retain weight-scaled markers as the subtle numerical
cue.

**Step 5: Run the focused test to verify all contracts pass**

Run: `julia --project=docs docs/test/test_grid_patterns.jl`

Expected: all `grid-pattern documentation figure` and `grid page wiring` tests pass.

**Step 6: Commit the page integration**

```bash
git add docs/src/grids.md docs/src/assets/custom.css docs/make.jl docs/plans/2026-08-22-grid-pattern-plots-design.md
git commit -m "docs: explain supported spherical grids visually"
```

### Task 4: Keep the generated figure checked in by CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Add a documentation-plot test step**

After `Configure doc environment` and before `julia-buildpkg`, add:

```yaml
      - name: Test documentation plots
        run: julia --project=docs docs/test/test_grid_patterns.jl
```

The focused test generates only into a temporary directory, so CI does not
mutate the checked-in asset.

**Step 2: Validate workflow syntax structurally**

Run:

```bash
sed -n '345,375p' .github/workflows/ci.yml
```

Expected: the new step is aligned with sibling `steps` entries and occurs after
the docs environment is instantiated.

**Step 3: Commit the CI check**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: verify documentation grid figure"
```

### Task 5: Regenerate, build, and visually verify the delivered page

**Files:**
- Verify: `docs/src/assets/grid-patterns.svg`
- Verify generated output: `docs/build/grids.html`

**Step 1: Verify deterministic regeneration**

Record the asset hash, run the generator, and compare the hash again. Expected:
the file is byte-for-byte unchanged. If GR embeds nondeterministic identifiers or
timestamps, normalize them in `generate_grid_patterns` and add a repeat-generation
assertion to the focused test.

**Step 2: Run the focused docs test**

Run: `julia --project=docs docs/test/test_grid_patterns.jl`

Expected: all tests pass.

**Step 3: Build the Documenter site**

Run: `julia --project=docs docs/make.jl`

Expected: exit 0, `docs/build/grids.html` exists, the page is in navigation, and
there are no new Documenter errors.

**Step 4: Inspect the rendered page at wide and narrow widths**

Open the local `docs/build/grids.html` in the in-app browser. Check desktop and
mobile-width screenshots for legible labels, complete globes, responsive image
scaling, caption readability, and no horizontal overflow. Iterate on the
generator or CSS if any plot is clipped or visually ambiguous.

**Step 5: Run regression verification**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: the baseline 67,239 serial checks and 1,673 parallel-grid checks pass;
optional MPI, JET, and Aqua groups remain skipped unless explicitly enabled.

**Step 6: Review the final diff and commit any visual refinements**

```bash
git diff --check
git status --short
git log --oneline -5
```

Expected: no whitespace errors, only intentional feature files, and a clean
worktree after the final refinement commit.
