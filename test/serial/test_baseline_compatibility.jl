using Test
using TOML
using SHTnsKit

const _BASELINE_COMPATIBILITY_FIXTURE = joinpath(
    @__DIR__, "..", "fixtures", "compatibility", "e2ce9027.toml"
)

function _compatibility_coefficients(cfg)
    Q = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    S = similar(Q)
    T = similar(Q)
    Q[2, 1] = 0.75
    Q[3, 2] = 0.125 - 0.25im
    S[2, 1] = -0.5
    S[3, 2] = 0.2 + 0.1im
    T[2, 1] = 0.375
    T[3, 2] = -0.15 + 0.05im
    return Q, S, T
end

function _run_baseline_probe(id::String)
    if id == "gauss_configuration_defaults"
        @test applicable(create_gauss_config, 3, 4)
        cfg = create_gauss_config(3, 4)
        @test (cfg.mmax, cfg.mres, cfg.nlon) == (3, 1, 8)
        @test (cfg.norm, cfg.cs_phase, cfg.real_norm, cfg.robert_form) ==
              (:orthonormal, true, false, false)
    elseif id == "regular_configuration_defaults"
        @test applicable(create_regular_config, 3, 5)
        cfg = create_regular_config(3, 5)
        @test (cfg.mmax, cfg.mres, cfg.nlon, cfg.grid_type) == (3, 1, 8, :regular)
        @test !cfg.real_norm
        @test !cfg.robert_form
    elseif id == "scalar_transform_keywords"
        cfg = create_gauss_config(3, 4)
        Q, _, _ = _compatibility_coefficients(cfg)
        @test applicable(synthesis, cfg, Q)
        field = synthesis(cfg, Q)
        explicit = synthesis(cfg, Q; real_output=true, fft_scratch=nothing, use_rfft=false)
        @test field isa Matrix{Float64}
        @test field ≈ explicit
        @test applicable(analysis, cfg, field)
        @test analysis(cfg, field) ≈
              analysis(cfg, field; fft_scratch=nothing, use_rfft=false)
        @test synthesis(cfg, Q; real_output=false) isa Matrix{ComplexF64}
    elseif id == "vector_transform_tuple_order"
        cfg = create_gauss_config(3, 4)
        _, S, T = _compatibility_coefficients(cfg)
        @test applicable(synthesis_sphtor, cfg, S, T)
        vector_field = synthesis_sphtor(cfg, S, T)
        @test vector_field isa Tuple && length(vector_field) == 2
        sph = synthesis_sphtor(cfg, S, zero(T))
        tor = synthesis_sphtor(cfg, zero(S), T)
        @test vector_field[1] ≈ sph[1] + tor[1]
        @test vector_field[2] ≈ sph[2] + tor[2]
        recovered = analysis_sphtor(cfg, vector_field...)
        @test recovered isa Tuple && length(recovered) == 2
        @test recovered[1] ≈ analysis_sphtor(cfg, sph...)[1]
        @test recovered[2] ≈ analysis_sphtor(cfg, tor...)[2]
    elseif id == "qst_transform_tuple_order"
        cfg = create_gauss_config(3, 4)
        Q, S, T = _compatibility_coefficients(cfg)
        @test applicable(synthesis_qst, cfg, Q, S, T)
        qst = synthesis_qst(cfg, Q, S, T)
        @test qst isa Tuple && length(qst) == 3
        @test qst[1] ≈ synthesis(cfg, Q)
        sphtor = synthesis_sphtor(cfg, S, T)
        @test qst[2] ≈ sphtor[1]
        @test qst[3] ≈ sphtor[2]
        recovered = analysis_qst(cfg, qst...)
        @test recovered isa Tuple && length(recovered) == 3
        @test recovered[1] ≈ analysis(cfg, qst[1])
        vector_recovered = analysis_sphtor(cfg, qst[2], qst[3])
        @test recovered[2] ≈ vector_recovered[1]
        @test recovered[3] ≈ vector_recovered[2]
    elseif id == "batch_transform_keywords"
        cfg = create_gauss_config(3, 4)
        Q, _, _ = _compatibility_coefficients(cfg)
        coefficients = cat(Q, 2Q; dims=3)
        @test applicable(synthesis_batch, cfg, coefficients)
        fields = synthesis_batch(cfg, coefficients)
        @test fields isa Array{Float64,3}
        @test fields ≈ synthesis_batch(
            cfg, coefficients; real_output=true, use_rfft=false,
        )
        @test applicable(analysis_batch, cfg, fields)
        @test analysis_batch(cfg, fields) ≈
              analysis_batch(cfg, fields; use_rfft=false)
    elseif id == "packed_transform_signatures"
        cfg = create_gauss_config(3, 4)
        Q, _, _ = _compatibility_coefficients(cfg)
        packed = zeros(ComplexF64, cfg.nlm)
        for m in 0:cfg.mmax, l in m:cfg.lmax
            packed[LM_index(cfg.lmax, cfg.mres, l, m) + 1] = Q[l + 1, m + 1]
        end
        @test applicable(synthesis_packed, cfg, packed)
        field = synthesis_packed(cfg, packed)
        @test applicable(analysis_packed, cfg, vec(field))
        @test length(analysis_packed(cfg, vec(field))) == cfg.nlm
        @test applicable(synthesis_point, cfg, Q, 0.25, 0.5)
        @test synthesis_point(cfg, Q, 0.25, 0.5) isa Number
    elseif id == "rotation_mutation_contract"
        rotation = shtns_rotation_create(3, 3, 0)
        @test applicable(shtns_rotation_set_angles_ZYZ, rotation, 0.0, 0.0, 0.0)
        @test shtns_rotation_set_angles_ZYZ(rotation, 0.0, 0.0, 0.0) === nothing
        input = zeros(ComplexF64, nlm_calc(3, 3, 1))
        input[1] = 1
        output = similar(input)
        @test applicable(shtns_rotation_apply_real, rotation, input, output)
        @test shtns_rotation_apply_real(rotation, input, output) === output
        @test output ≈ input
    elseif id == "device_value_first_order"
        values = [1.0, 2.0]
        @test applicable(to_device, values, CPU())
        @test set_device!(CPU()) isa CPU
        @test to_device(values, CPU()) === values
        @test to_device(values) === values
        @test on_device(values) isa CPU
    else
        error("unknown cleanup-baseline compatibility probe: $id")
    end
end

@testset "Cleanup-baseline public API compatibility" begin
    @test isfile(_BASELINE_COMPATIBILITY_FIXTURE)

    if isfile(_BASELINE_COMPATIBILITY_FIXTURE)
        fixture = TOML.parsefile(_BASELINE_COMPATIBILITY_FIXTURE)
        baseline_exports = Set(Symbol.(fixture["exports"]))
        final_exports = Set(names(SHTnsKit; all=false, imported=false))

        @test fixture["baseline"]["commit"] == "e2ce9027"
        @test fixture["baseline"]["commit_full"] ==
              "e2ce9027c94a40b29c6582bc0616d22e26a94ad7"
        @test fixture["baseline"]["source_blob"] ==
              "f5ec0b731b891260a4c44f94951ddf118086f44b"
        @test fixture["baseline"]["export_count"] == length(baseline_exports)
        @test baseline_exports ⊆ final_exports

        probes = fixture["probe"]
        expected_probe_ids = Set((
            "gauss_configuration_defaults",
            "regular_configuration_defaults",
            "scalar_transform_keywords",
            "vector_transform_tuple_order",
            "qst_transform_tuple_order",
            "batch_transform_keywords",
            "packed_transform_signatures",
            "rotation_mutation_contract",
            "device_value_first_order",
        ))
        @test Set(probe["id"] for probe in probes) == expected_probe_ids
        for probe in probes
            @test haskey(probe, "signature")
            @test haskey(probe, "defaults")
            @test haskey(probe, "return_contract")
            _run_baseline_probe(probe["id"])
        end
    end
end
