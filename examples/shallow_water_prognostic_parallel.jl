#!/usr/bin/env julia

using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit
using Printf
using PencilArrays: PencilArray
import SHTnsKitParallelExt

const DEFAULT_OPTS = (
    lmax = 63,
    dt = 600.0,
    steps = 360,
    diag_stride = 12,
    radius = 6.371e6,
    omega = 7.2921159e-5,
    g = 9.80616,
    H = 8000.0,
    νζ = 5e5,
    νδ = 2e5,
    νh = 2e5,
)

struct SWParams
    dt::Float64
    steps::Int
    diag_stride::Int
    radius::Float64
    omega::Float64
    g::Float64
    H::Float64
    νζ::Float64
    νδ::Float64
    νh::Float64
end

mutable struct SWState
    ζlm::Matrix{ComplexF64}
    δlm::Matrix{ComplexF64}
    hlm::Matrix{ComplexF64}
end

mutable struct SWWorkspace
    ζ_field::PencilArray
    δ_field::PencilArray
    h_field::PencilArray
    abs_vort::PencilArray
    rhs_ζ::PencilArray
    rhs_δ::PencilArray
    rhs_h::PencilArray
    tmp_scalar1::PencilArray
    tmp_scalar2::PencilArray
    vecθ::PencilArray
    vecφ::PencilArray
    rotθ::PencilArray
    rotφ::PencilArray
    massθ::PencilArray
    massφ::PencilArray
    pot_field::PencilArray
    ke_field::PencilArray
    Vt::PencilArray
    Vp::PencilArray
    Slm_buf::Matrix{ComplexF64}
    Tlm_buf::Matrix{ComplexF64}
    stage_ζ::Matrix{ComplexF64}
    stage_δ::Matrix{ComplexF64}
    stage_h::Matrix{ComplexF64}
    N1_ζ::Matrix{ComplexF64}
    N1_δ::Matrix{ComplexF64}
    N1_h::Matrix{ComplexF64}
    N2_ζ::Matrix{ComplexF64}
    N2_δ::Matrix{ComplexF64}
    N2_h::Matrix{ComplexF64}
end

struct LinearCache
    expL::Matrix{Float64}
    phi1::Matrix{Float64}
    phi2::Matrix{Float64}
end

mutable struct PlanSet
    aplan::DistAnalysisPlan
    splan::DistPlan
    vplan::DistSphtorPlan
end

function parse_cli(args)
    opts = Dict{Symbol,Any}(pairs(DEFAULT_OPTS))
    i = 1
    while i <= length(args)
        arg = args[i]
        key, val = if occursin('=', arg)
            split(arg, '=', limit=2)
        else
            (arg, nothing)
        end
        function take_value()
            if val !== nothing
                return val
            elseif i + 1 <= length(args)
                i += 1
                return args[i]
            else
                error("Flag $arg requires a value")
            end
        end
        if startswith(key, "--")
            flag = key[3:end]
            value_str = take_value()
            if flag == "lmax"
                opts[:lmax] = parse(Int, value_str)
            elseif flag in ("steps", "diag", "diag_stride")
                opts[:steps] = flag == "steps" ? parse(Int, value_str) : opts[:steps]
                opts[:diag_stride] = parse(Int, value_str)
            elseif flag in ("dt", "radius", "omega", "g", "H", "νζ", "νδ", "νh")
                opts[Symbol(flag)] = parse(Float64, value_str)
            else
                error("Unknown flag $flag")
            end
        else
            error("Unexpected argument $arg")
        end
        i += 1
    end
    params = SWParams(
        opts[:dt],
        opts[:steps],
        opts[:diag_stride],
        opts[:radius],
        opts[:omega],
        opts[:g],
        opts[:H],
        opts[:νζ],
        opts[:νδ],
        opts[:νh],
    )
    return opts[:lmax], params
end

function proc_grid(p::Integer)
    best = (1, p); diff = p - 1
    for d in 1:p
        p % d == 0 || continue
        d2 = div(p, d)
        if abs(d - d2) < diff
            best = (d, d2); diff = abs(d - d2)
        end
    end
    return best
end

paalloc(topo; eltype=Float64) = begin
    try
        return PencilArrays.zeros(topo; eltype=eltype)
    catch
        try
            return PencilArrays.zeros(topo)
        catch
            return SHTnsKitParallelExt.allocate(topo; eltype=eltype)
        end
    end
end

function build_linear_cache(cfg::SHTConfig, dt::Float64, ν::Float64, radius::Float64)
    lmax, mmax = cfg.lmax, cfg.mmax
    expL = ones(Float64, lmax + 1, mmax + 1)
    phi1 = ones(Float64, lmax + 1, mmax + 1)
    phi2 = fill(0.5, lmax + 1, mmax + 1)
    if ν == 0
        return LinearCache(expL, phi1, phi2)
    end
    invR2 = 1 / (radius^2)
    for m in 0:mmax, l in m:lmax
        row, col = l + 1, m + 1
        λ = -ν * (l * (l + 1)) * invR2
        z = dt * λ
        ez = exp(z)
        expL[row, col] = ez
        if abs(z) < 1e-12
            phi1[row, col] = 1.0
            phi2[row, col] = 0.5
        else
            phi1[row, col] = (ez - 1) / z
            phi2[row, col] = (ez - 1 - z) / (z^2)
        end
    end
    return LinearCache(expL, phi1, phi2)
end

@inline function linear_stage!(dest::Matrix{ComplexF64}, state::Matrix{ComplexF64},
                               nonlin::Matrix{ComplexF64}, cache::LinearCache,
                               lmax::Int, mmax::Int, dt::Float64)
    expL = cache.expL; phi1 = cache.phi1
    @inbounds for m in 0:mmax, l in m:lmax
        row = l + 1; col = m + 1
        dest[row, col] = expL[row, col] * state[row, col] + dt * phi1[row, col] * nonlin[row, col]
    end
    return dest
end

@inline function finalize_linear!(state::Matrix{ComplexF64}, stage::Matrix{ComplexF64},
                                  N1::Matrix{ComplexF64}, N2::Matrix{ComplexF64},
                                  cache::LinearCache, lmax::Int, mmax::Int, dt::Float64)
    phi2 = cache.phi2
    @inbounds for m in 0:mmax, l in m:lmax
        row = l + 1; col = m + 1
        state[row, col] = stage[row, col] + dt * phi2[row, col] * (N2[row, col] - N1[row, col])
    end
    return state
end

function global_max_abs(field::PencilArray, comm)
    local_max = maximum(abs, field)
    return MPI.Allreduce(local_max, MPI.MAX, comm)
end

function initialize_state!(cfg::SHTConfig, plans::PlanSet, state::SWState,
                           work::SWWorkspace, params::SWParams, fgrid::PencilArray)
    ζ_amp = 4e-5
    h_amp = 200.0
    σ = 0.3
    θ_idx = collect(globalindices(work.ζ_field, 1))
    φ_idx = collect(globalindices(work.ζ_field, 2))
    for (iθ, gθ) in enumerate(θ_idx)
        θ = cfg.θ[gθ]
        lat = π / 2 - θ
        base_vort = ζ_amp * sin(lat) * cos(lat)
        h_bump = h_amp * exp(-((lat - 0.5)^2 + (0.0)^2) / σ^2)
        for (iφ, gφ) in enumerate(φ_idx)
            φ = cfg.φ[gφ]
            work.ζ_field[iθ, iφ] = base_vort + 1e-5 * sin(2φ) * cos(lat)^2
            work.δ_field[iθ, iφ] = 0.0
            work.h_field[iθ, iφ] = h_bump * cos(φ)
        end
    end
    dist_analysis!(plans.aplan, state.ζlm, work.ζ_field)
    dist_analysis!(plans.aplan, state.δlm, work.δ_field)
    dist_analysis!(plans.aplan, state.hlm, work.h_field)
    work.abs_vort .= work.ζ_field
    work.abs_vort .+= fgrid
end

function synthesize_scalar!(out::PencilArray, matrix::Matrix{ComplexF64}, splan::DistPlan)
    Alm_p = PencilArray(matrix)
    SHTnsKit.dist_synthesis!(splan, out, Alm_p; real_output=true)
    return out
end

function compute_velocity!(cfg::SHTConfig, plans::PlanSet, work::SWWorkspace,
                           ζlm::Matrix{ComplexF64}, δlm::Matrix{ComplexF64})
    SHTnsKit.spheroidal_from_divergence!(cfg, work.Slm_buf, δlm)
    SHTnsKit.toroidal_from_vorticity!(cfg, work.Tlm_buf, ζlm)
    SHTnsKit.dist_synthesis_sphtor!(plans.vplan, work.Vt, work.Vp, work.Slm_buf, work.Tlm_buf; real_output=true)
    return work.Vt, work.Vp
end

function compute_rhs!(cfg::SHTConfig, plans::PlanSet, work::SWWorkspace,
                      params::SWParams, fgrid::PencilArray,
                      ζlm::Matrix{ComplexF64}, δlm::Matrix{ComplexF64}, hlm::Matrix{ComplexF64},
                      outζ::Matrix{ComplexF64}, outδ::Matrix{ComplexF64}, outh::Matrix{ComplexF64})
    synthesize_scalar!(work.ζ_field, ζlm, plans.splan)
    synthesize_scalar!(work.δ_field, δlm, plans.splan)
    synthesize_scalar!(work.h_field, hlm, plans.splan)
    compute_velocity!(cfg, plans, work, ζlm, δlm)

    work.abs_vort .= work.ζ_field
    work.abs_vort .+= fgrid

    work.vecθ .= work.abs_vort .* work.Vt
    work.vecφ .= work.abs_vort .* work.Vp
    div_abs = SHTnsKit.dist_spatial_divergence(cfg, work.vecθ, work.vecφ; prototype_θφ=work.tmp_scalar1,
                                               use_rfft=true, real_output=true)
    copyto!(work.tmp_scalar1, div_abs)
    work.rhs_ζ .= .-work.tmp_scalar1
    if params.νζ > 0
        lapζ = SHTnsKit.dist_scalar_laplacian(cfg, work.ζ_field; prototype_θφ=work.tmp_scalar2,
                                              use_rfft=true, real_output=true)
        copyto!(work.tmp_scalar2, lapζ)
        @. work.rhs_ζ += (params.νζ / params.radius^2) * work.tmp_scalar2
    end

    work.rotθ .= .-work.abs_vort .* work.Vp
    work.rotφ .= work.abs_vort .* work.Vt
    div_rot = SHTnsKit.dist_spatial_divergence(cfg, work.rotθ, work.rotφ; prototype_θφ=work.tmp_scalar1,
                                               use_rfft=true, real_output=true)
    copyto!(work.tmp_scalar1, div_rot)

    work.ke_field .= 0.5 .* (work.Vt .* work.Vt .+ work.Vp .* work.Vp)
    work.pot_field .= params.g .* work.h_field
    work.pot_field .+= work.ke_field
    lap_phi = SHTnsKit.dist_scalar_laplacian(cfg, work.pot_field; prototype_θφ=work.tmp_scalar2,
                                             use_rfft=true, real_output=true)
    copyto!(work.tmp_scalar2, lap_phi)
    work.rhs_δ .= .-(work.tmp_scalar1 .+ work.tmp_scalar2)
    if params.νδ > 0
        lapδ = SHTnsKit.dist_scalar_laplacian(cfg, work.δ_field; prototype_θφ=work.tmp_scalar2,
                                              use_rfft=true, real_output=true)
        copyto!(work.tmp_scalar2, lapδ)
        @. work.rhs_δ += (params.νδ / params.radius^2) * work.tmp_scalar2
    end

    work.massθ .= (params.H .+ work.h_field) .* work.Vt
    work.massφ .= (params.H .+ work.h_field) .* work.Vp
    div_mass = SHTnsKit.dist_spatial_divergence(cfg, work.massθ, work.massφ; prototype_θφ=work.tmp_scalar1,
                                                use_rfft=true, real_output=true)
    copyto!(work.tmp_scalar1, div_mass)
    work.rhs_h .= .-work.tmp_scalar1
    if params.νh > 0
        laph = SHTnsKit.dist_scalar_laplacian(cfg, work.h_field; prototype_θφ=work.tmp_scalar2,
                                              use_rfft=true, real_output=true)
        copyto!(work.tmp_scalar2, laph)
        @. work.rhs_h += (params.νh / params.radius^2) * work.tmp_scalar2
    end

    dist_analysis!(plans.aplan, outζ, work.rhs_ζ)
    dist_analysis!(plans.aplan, outδ, work.rhs_δ)
    dist_analysis!(plans.aplan, outh, work.rhs_h)
end

function etdrk2_step!(cfg::SHTConfig, plans::PlanSet, work::SWWorkspace,
                      params::SWParams, fgrid::PencilArray,
                      caches::NamedTuple, state::SWState)
    lmax, mmax = cfg.lmax, cfg.mmax
    compute_rhs!(cfg, plans, work, params, fgrid,
                 state.ζlm, state.δlm, state.hlm,
                 work.N1_ζ, work.N1_δ, work.N1_h)

    linear_stage!(work.stage_ζ, state.ζlm, work.N1_ζ, caches.ζ, lmax, mmax, params.dt)
    linear_stage!(work.stage_δ, state.δlm, work.N1_δ, caches.δ, lmax, mmax, params.dt)
    linear_stage!(work.stage_h, state.hlm, work.N1_h, caches.h, lmax, mmax, params.dt)

    compute_rhs!(cfg, plans, work, params, fgrid,
                 work.stage_ζ, work.stage_δ, work.stage_h,
                 work.N2_ζ, work.N2_δ, work.N2_h)

    finalize_linear!(state.ζlm, work.stage_ζ, work.N1_ζ, work.N2_ζ, caches.ζ, lmax, mmax, params.dt)
    finalize_linear!(state.δlm, work.stage_δ, work.N1_δ, work.N2_δ, caches.δ, lmax, mmax, params.dt)
    finalize_linear!(state.hlm, work.stage_h, work.N1_h, work.N2_h, caches.h, lmax, mmax, params.dt)
    return state
end

function update_grids!(cfg::SHTConfig, plans::PlanSet, work::SWWorkspace, state::SWState)
    synthesize_scalar!(work.ζ_field, state.ζlm, plans.splan)
    synthesize_scalar!(work.δ_field, state.δlm, plans.splan)
    synthesize_scalar!(work.h_field, state.hlm, plans.splan)
    compute_velocity!(cfg, plans, work, state.ζlm, state.δlm)
end

function diagnostics!(cfg::SHTConfig, plans::PlanSet, work::SWWorkspace,
                      state::SWState, params::SWParams, step::Int, comm)
    update_grids!(cfg, plans, work, state)
    max_h = global_max_abs(work.h_field, comm)
    max_u = max(global_max_abs(work.Vt, comm), global_max_abs(work.Vp, comm))
    mass = global_mean(cfg, work.h_field, comm)
    if MPI.Comm_rank(comm) == 0
        @printf("step %5d : max|h| %8.3f m  max|u| %8.3f m/s  mean(h) %8.3f m\n",
                step, max_h, max_u, mass)
    end
end

function global_mean(cfg::SHTConfig, field::PencilArray, comm)
    θloc = axes(field, 1)
    glθ = collect(globalindices(field, 1))
    dφ = cfg.cphi
    local = 0.0
    for (ii, ilat) in enumerate(θloc)
        w = cfg.w[glθ[ii]]
        local += w * sum(view(field, ilat, :)) * dφ
    end
    total = MPI.Allreduce(local, MPI.SUM, comm)
    return total / (4π)
end

function allocate_workspace(cfg::SHTConfig, topo::PencilArrays.Pencil)
    ζ_field = paalloc(topo; eltype=Float64)
    δ_field = paalloc(topo; eltype=Float64)
    h_field = paalloc(topo; eltype=Float64)
    abs_vort = similar(ζ_field)
    rhs_ζ = similar(ζ_field)
    rhs_δ = similar(ζ_field)
    rhs_h = similar(ζ_field)
    tmp_scalar1 = similar(ζ_field)
    tmp_scalar2 = similar(ζ_field)
    vecθ = similar(ζ_field)
    vecφ = similar(ζ_field)
    rotθ = similar(ζ_field)
    rotφ = similar(ζ_field)
    massθ = similar(ζ_field)
    massφ = similar(ζ_field)
    pot_field = similar(ζ_field)
    ke_field = similar(ζ_field)
    Vt = similar(ζ_field)
    Vp = similar(ζ_field)
    lmax, mmax = cfg.lmax, cfg.mmax
    Slm_buf = zeros(ComplexF64, lmax+1, mmax+1)
    Tlm_buf = zeros(ComplexF64, lmax+1, mmax+1)
    stage_ζ = similar(Slm_buf)
    stage_δ = similar(Slm_buf)
    stage_h = similar(Slm_buf)
    N1_ζ = similar(Slm_buf)
    N1_δ = similar(Slm_buf)
    N1_h = similar(Slm_buf)
    N2_ζ = similar(Slm_buf)
    N2_δ = similar(Slm_buf)
    N2_h = similar(Slm_buf)
    return SWWorkspace(ζ_field, δ_field, h_field, abs_vort,
                       rhs_ζ, rhs_δ, rhs_h, tmp_scalar1, tmp_scalar2,
                       vecθ, vecφ, rotθ, rotφ, massθ, massφ,
                       pot_field, ke_field, Vt, Vp,
                       Slm_buf, Tlm_buf,
                       stage_ζ, stage_δ, stage_h,
                       N1_ζ, N1_δ, N1_h, N2_ζ, N2_δ, N2_h)
end

function run_sim(lmax::Int, params::SWParams)
    MPI.Init()
    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)

        nlat = lmax + 2
        nlon = 2 * (2 * lmax + 1)  # factor 2 for better longitudinal resolution
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        enable_plm_tables!(cfg)

        pθ, pφ = proc_grid(nprocs)
        topo = Pencil((nlat, nlon), (pθ, pφ), comm)
        work = allocate_workspace(cfg, topo)

        fgrid = similar(work.ζ_field)
        θ_idx = collect(globalindices(fgrid, 1))
        for (iθ, gθ) in enumerate(θ_idx)
            θ = cfg.θ[gθ]
            fval = 2 * params.omega * cos(θ)
            fgrid[iθ, :] .= fval
        end

        ζlm = zeros(ComplexF64, lmax+1, lmax+1)
        δlm = zeros(ComplexF64, lmax+1, lmax+1)
        hlm = zeros(ComplexF64, lmax+1, lmax+1)
        state = SWState(ζlm, δlm, hlm)

        Vt = work.Vt
        vplan = DistSphtorPlan(cfg, Vt; use_rfft=true, with_spatial_scratch=true)
        aplan = DistAnalysisPlan(cfg, work.ζ_field; use_rfft=true)
        splan = DistPlan(cfg, work.ζ_field; use_rfft=true)
        plans = PlanSet(aplan, splan, vplan)

        initialize_state!(cfg, plans, state, work, params, fgrid)

        caches = (
            ζ = build_linear_cache(cfg, params.dt, params.νζ, params.radius),
            δ = build_linear_cache(cfg, params.dt, params.νδ, params.radius),
            h = build_linear_cache(cfg, params.dt, params.νh, params.radius),
        )

        if rank == 0
            @info "Shallow-water run" lmax nlat nlon dt=params.dt steps=params.steps νζ=params.νζ νδ=params.νδ νh=params.νh
        end

        for step in 1:params.steps
            etdrk2_step!(cfg, plans, work, params, fgrid, caches, state)
            if step % params.diag_stride == 0
                diagnostics!(cfg, plans, work, state, params, step, comm)
            end
        end
    finally
        MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    lmax, params = parse_cli(ARGS)
    run_sim(lmax, params)
end
