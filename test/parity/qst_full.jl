using Test
using SHTnsKit

isdefined(@__MODULE__, :ScalarParityAdapter) || include("scalar_full.jl")
isdefined(@__MODULE__, :VectorParityAdapter) || include("sphtor_full.jl")

abstract type QSTParityAdapter end

struct CPUQSTAdapter <: QSTParityAdapter end

qst_place(::CPUQSTAdapter, ::SHTConfig, value, ::Symbol) = value
qst_collect(::CPUQSTAdapter, value, ::SHTConfig) = Array(value)
qst_resident(::CPUQSTAdapter, value) = @test on_device(value) isa CPU
qst_analysis(::CPUQSTAdapter, cfg, Vr, Vt, Vp; use_rfft=false) =
    analysis_qst(CPU(), cfg, Vr, Vt, Vp; use_rfft)
qst_analysis_cplx(::CPUQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst_cplx(CPU(), cfg, Vr, Vt, Vp)
qst_synthesis(::CPUQSTAdapter, cfg, Q, S, Tlm, _prototype;
              real_output=true, use_rfft=false) =
    synthesis_qst(CPU(), cfg, Q, S, Tlm; real_output, use_rfft)
qst_synthesis_cplx(::CPUQSTAdapter, cfg, Q, S, Tlm, _prototype) =
    synthesis_qst_cplx(CPU(), cfg, Q, S, Tlm)

function _qst_modes(cfg, ::Type{T}) where {T<:AbstractFloat}
    CT = Complex{T}
    Q = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    Q[1, 1] = CT(T(0.31), 0)
    Q[2, 1] = CT(T(-0.17), 0)
    Q[3, 3] = CT(T(0.07), T(-0.04))
    cfg.mres == 1 && (Q[2, 2] = CT(T(0.11), T(0.05)))
    S, Tlm = _vector_modes(cfg, T)
    return Q, S, Tlm
end

function _qst_external(cfg, canonical)
    external = zero(canonical)
    SHTnsKit.convert_alm_norm!(external, canonical, cfg; to_internal=false)
    return external
end

function _qst_references(cfg, Q, S_can, T_can; real_output::Bool)
    Vr = _direct_scalar_sum(cfg, Q; real_output)
    Vt, Vp = _direct_low_vector(cfg, S_can, T_can; real_output)
    return Vr, Vt, Vp
end

function _test_qst_case(adapter::QSTParityAdapter, cfg, ::Type{T}) where {T<:AbstractFloat}
    tol = _vector_tol(T)
    Q_can, S_can, T_can = _qst_modes(cfg, T)
    Q = _qst_external(cfg, Q_can)
    S = _qst_external(cfg, S_can)
    Tlm = _qst_external(cfg, T_can)
    Vr_ref, Vt_ref, Vp_ref = _qst_references(
        cfg, Q, S_can, T_can; real_output=true,
    )
    prototype = qst_place(adapter, cfg, Vr_ref, :spatial)
    Qd = qst_place(adapter, cfg, Q, :spectral)
    Sd = qst_place(adapter, cfg, S, :spectral)
    Td = qst_place(adapter, cfg, Tlm, :spectral)

    Vr, Vt, Vp = qst_synthesis(
        adapter, cfg, Qd, Sd, Td, prototype; real_output=true,
    )
    for value in (Vr, Vt, Vp)
        qst_resident(adapter, value)
    end
    for (got, expected) in zip((Vr, Vt, Vp), (Vr_ref, Vt_ref, Vp_ref))
        host = qst_collect(adapter, got, cfg)
        @test eltype(host) === T
        @test host ≈ expected atol=tol.atol rtol=tol.rtol
    end

    Qa, Sa, Ta = qst_analysis(
        adapter, cfg,
        qst_place(adapter, cfg, Vr_ref, :spatial),
        qst_place(adapter, cfg, Vt_ref, :spatial),
        qst_place(adapter, cfg, Vp_ref, :spatial),
    )
    for (got, expected) in zip((Qa, Sa, Ta), (Q, S, Tlm))
        qst_resident(adapter, got)
        host = qst_collect(adapter, got, cfg)
        @test eltype(host) === Complex{T}
        @test host ≈ expected atol=tol.atol rtol=tol.rtol
    end

    Q_complex = copy(Q)
    Q_complex[1, 1] = Complex{T}(T(0.31), T(0.08))
    Vrc_ref, Vtc_ref, Vpc_ref = _qst_references(
        cfg, Q_complex, S_can, T_can; real_output=false,
    )
    Qcd = qst_place(adapter, cfg, Q_complex, :spectral)
    Vrc, Vtc, Vpc = qst_synthesis_cplx(
        adapter, cfg, Qcd, Sd, Td, prototype,
    )
    for (got, expected) in zip((Vrc, Vtc, Vpc), (Vrc_ref, Vtc_ref, Vpc_ref))
        host = qst_collect(adapter, got, cfg)
        @test eltype(host) === Complex{T}
        @test host ≈ expected atol=tol.atol rtol=tol.rtol
    end
    Qac, Sac, Tac = qst_analysis_cplx(
        adapter, cfg,
        qst_place(adapter, cfg, Vrc_ref, :spatial),
        qst_place(adapter, cfg, Vtc_ref, :spatial),
        qst_place(adapter, cfg, Vpc_ref, :spatial),
    )
    for (got, expected) in zip((Qac, Sac, Tac), (Q_complex, S, Tlm))
        @test qst_collect(adapter, got, cfg) ≈ expected atol=tol.atol rtol=tol.rtol
    end
    return nothing
end

function _test_qst_isolated_components(adapter::QSTParityAdapter)
    cfg = _vector_config(:regular_poles, 2, 7)
    Q_can, S_can, T_can = _qst_modes(cfg, Float64)
    zero_modes = zero(Q_can)
    prototype = qst_place(
        adapter, cfg, zeros(Float64, cfg.nlat, cfg.nlon), :spatial,
    )
    components = (
        (Q_can, zero_modes, zero_modes),
        (zero_modes, S_can, zero_modes),
        (zero_modes, zero_modes, T_can),
    )
    for (index, canonical) in pairs(components)
        Q, S, Tlm = map(value -> _qst_external(cfg, value), canonical)
        got = qst_synthesis(
            adapter, cfg,
            qst_place(adapter, cfg, Q, :spectral),
            qst_place(adapter, cfg, S, :spectral),
            qst_place(adapter, cfg, Tlm, :spectral),
            prototype; real_output=true,
        )
        expected = _qst_references(
            cfg, Q, canonical[2], canonical[3]; real_output=true,
        )
        for component in 1:3
            host = qst_collect(adapter, got[component], cfg)
            @test host ≈ expected[component] atol=4e-11 rtol=4e-11
            is_decoupled = index == 1 ? component != 1 : component == 1
            is_decoupled && @test host ≈ zero(host) atol=4e-11 rtol=4e-11
        end
    end
    return nothing
end

function _test_qst_mres_leakage(adapter::QSTParityAdapter)
    cfg = _vector_config(:gauss, 3, 8; mres=2)
    unsupported = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    unsupported[2, 2] = 0.3 - 0.2im
    zero_modes = zero(unsupported)
    prototype = qst_place(
        adapter, cfg, zeros(Float64, cfg.nlat, cfg.nlon), :spatial,
    )
    placed = qst_place(adapter, cfg, unsupported, :spectral)
    zero_placed = qst_place(adapter, cfg, zero_modes, :spectral)
    for canonical in ((placed, zero_placed, zero_placed),
                      (zero_placed, placed, zero_placed),
                      (zero_placed, zero_placed, placed))
        fields = qst_synthesis(adapter, cfg, canonical..., prototype;
                               real_output=true)
        @test all(value -> all(iszero, qst_collect(adapter, value, cfg)), fields)
    end

    c = sqrt(3 / (8pi))
    Vr = [c * sqrt(max(0.0, 1 - cfg.x[i]^2)) * cos(cfg.φ[j])
          for i in 1:cfg.nlat, j in 1:cfg.nlon]
    Vt = [-2c * cfg.x[i] * sqrt(max(0.0, 1 - cfg.x[i]^2)) * cos(cfg.φ[j])
          for i in 1:cfg.nlat, j in 1:cfg.nlon]
    Vp = [2c * sqrt(max(0.0, 1 - cfg.x[i]^2)) * sin(cfg.φ[j])
          for i in 1:cfg.nlat, j in 1:cfg.nlon]
    analyzed = qst_analysis(
        adapter, cfg,
        qst_place(adapter, cfg, Vr, :spatial),
        qst_place(adapter, cfg, Vt, :spatial),
        qst_place(adapter, cfg, Vp, :spatial),
    )
    for value in analyzed
        @test all(iszero, qst_collect(adapter, value, cfg)[:, 2])
    end
    return nothing
end

function run_qst_full_parity(adapter::QSTParityAdapter;
                             grid_kinds=_VECTOR_GRID_KINDS,
                             precisions=(Float32, Float64),
                             mres_values=(1, 2),
                             norms=(:orthonormal, :fourpi, :schmidt),
                             real_norm_values=(false, true),
                             cs_phase_values=(false, true),
                             robert_values=(false, true),
                             pole_orders=(false, true))
    @testset "QST full-grid parity $(nameof(typeof(adapter)))" begin
        for kind in grid_kinds, T in precisions, mres in mres_values,
            norm in norms, real_norm in real_norm_values,
            cs_phase in cs_phase_values, robert_form in robert_values,
            south_pole_first in pole_orders
            cfg = _vector_config(
                kind, 3, 8; mres, norm, real_norm, cs_phase,
                robert_form, south_pole_first,
            )
            _test_qst_case(adapter, cfg, T)
        end
        _test_qst_isolated_components(adapter)
        _test_qst_mres_leakage(adapter)
    end
    return nothing
end

function run_cpu_qst_full_parity()
    @testset "QST public contract" begin
        @test :dist_analysis_qst in names(SHTnsKit)
        @test :dist_synthesis_qst in names(SHTnsKit)
    end
    run_qst_full_parity(CPUQSTAdapter())
    return nothing
end
