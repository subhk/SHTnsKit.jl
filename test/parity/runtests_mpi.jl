using Test
using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit

MPI.Init()

include("scalar_full.jl")
include("scalar_variants.jl")

struct MPIScalarAdapter <: ScalarParityAdapter
    comm
end

function place(adapter::MPIScalarAdapter, cfg::SHTConfig, value::AbstractMatrix, kind::Symbol)
    if kind === :spectral
        return matrix_to_spectral_pencil(cfg, value; comm=adapter.comm)
    end
    pen = create_spatial_pencil(cfg; comm=adapter.comm)
    placed = PencilArray{eltype(value)}(undef, pen)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        parent(placed)[i, j] = value[iglobal, jglobal]
    end
    return placed
end

collect_result(::MPIScalarAdapter, value::PencilArray, cfg::SHTConfig) =
    size_global(value) == (cfg.lmax + 1, cfg.mmax + 1) ?
        spectral_pencil_to_matrix(cfg, value) : _collect_spatial(value, cfg)

function _collect_spatial(value::PencilArray, cfg::SHTConfig)
    result = zeros(eltype(value), cfg.nlat, cfg.nlon)
    pen = pencil(value)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        result[iglobal, jglobal] = parent(value)[i, j]
    end
    MPI.Allreduce!(result, +, PencilArrays.get_comm(value))
    return result
end

function _collect_distributed_vector(value::PencilArray)
    global_size = size_global(value)
    global_size[2] == 1 || error("expected a singleton-column PencilArray")
    result = zeros(eltype(value), global_size[1])
    globals = collect(Int, PencilArrays.range_local(pencil(value))[1])
    @inbounds for (i, iglobal) in pairs(globals)
        result[iglobal] = parent(value)[i, 1]
    end
    MPI.Allreduce!(result, +, PencilArrays.get_comm(value))
    return result
end

function _place_distributed_vector(values::AbstractVector, comm)
    pen = Pencil((length(values), 1), (1,), comm)
    result = PencilArray{eltype(values)}(undef, pen)
    globals = collect(Int, PencilArrays.range_local(pen)[1])
    @inbounds for (i, iglobal) in pairs(globals)
        parent(result)[i, 1] = values[iglobal]
    end
    return result
end

analysis_call(::MPIScalarAdapter, cfg, field; use_rfft=false) =
    analysis(cfg, field; use_rfft)
synthesis_call(::MPIScalarAdapter, cfg, coefficients, prototype;
               real_output, use_rfft=false) =
    synthesis(cfg, coefficients; prototype_θφ=prototype, real_output, use_rfft)
synthesis_cplx_call(::MPIScalarAdapter, cfg, coefficients, prototype) =
    synthesis_cplx(cfg, coefficients; prototype_θφ=prototype)
assert_resident(::MPIScalarAdapter, value) = @test value isa PencilArray

function test_phi_split_complex_float32(adapter::MPIScalarAdapter)
    cfg = _scalar_config(:gauss, 3, 8)
    coefficients = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
    coefficients[1, 1] = ComplexF32(0.25, -0.15)
    coefficients[3, 3] = ComplexF32(-0.2, 0.1)
    field = _direct_scalar_sum(cfg, coefficients; real_output=false)

    pen = Pencil((cfg.nlat, cfg.nlon), adapter.comm)
    distributed = PencilArray{ComplexF32}(undef, pen)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        parent(distributed)[i, j] = field[iglobal, jglobal]
    end

    analyzed = analysis(cfg, distributed; return_pencil=true)
    @test analyzed isa PencilArray
    analyzed_host = spectral_pencil_to_matrix(cfg, analyzed)
    @test eltype(analyzed_host) === ComplexF32
    @test analyzed_host ≈ analysis(CPU(), cfg, field) atol=2f-4 rtol=2f-4

    reconstructed = synthesis(
        cfg, analyzed; prototype_θφ=distributed, real_output=false,
    )
    @test reconstructed isa PencilArray
    reconstructed_host = _collect_spatial(reconstructed, cfg)
    @test eltype(reconstructed_host) === ComplexF32
    @test reconstructed_host ≈ field atol=2f-4 rtol=2f-4
    return nothing
end

function test_pencil_native_path(adapter::MPIScalarAdapter)
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    cfg = _scalar_config(:gauss, 5, 10)
    coefficients, field = _closed_form_low_order(cfg, Float64)
    spatial = place(adapter, cfg, field, :spatial)
    spectral = place(adapter, cfg, coefficients, :spectral)

    extension._reset_pencil_scalar_stats!()
    analyzed = analysis(cfg, spatial)
    after_analysis = extension._pencil_scalar_stats()
    @test after_analysis.full_matrix_helper_calls == 0
    max_spectral_local = MPI.Allreduce(length(parent(analyzed)), max, adapter.comm)
    @test after_analysis.analysis_max_message_elements <= max_spectral_local

    extension._reset_pencil_scalar_stats!()
    reconstructed = synthesis(cfg, spectral; prototype_θφ=spatial)
    after_synthesis = extension._pencil_scalar_stats()
    @test after_synthesis.full_matrix_helper_calls == 0
    max_theta_local = MPI.Allreduce(size(parent(spatial), 1), max, adapter.comm)
    @test after_synthesis.synthesis_max_message_elements <= max_theta_local * cfg.nlon
    @test _collect_spatial(reconstructed, cfg) ≈ field atol=3e-12 rtol=3e-12

    dense_compat = analysis(cfg, spatial; return_pencil=false)
    @test dense_compat isa Matrix
    @test dense_compat ≈ coefficients atol=3e-12 rtol=3e-12
    local_compat = dist_synthesis(cfg, dense_compat; prototype_θφ=spatial)
    compat = PencilArray{eltype(local_compat)}(undef, pencil(spatial))
    copyto!(parent(compat), local_compat)
    @test _collect_spatial(compat, cfg) ≈ field atol=3e-12 rtol=3e-12

    pen2 = try
        Pencil((cfg.nlat, cfg.nlon), (1, 2), adapter.comm)
    catch
        nothing
    end
    if pen2 !== nothing
        ranges = PencilArrays.range_local(pen2)
        both_split = MPI.Allreduce(
            length(ranges[1]) < cfg.nlat && length(ranges[2]) < cfg.nlon,
            &, adapter.comm,
        )
        if both_split
            spatial2 = PencilArray{Float64}(undef, pen2)
            for (j, jglobal) in pairs(ranges[2]), (i, iglobal) in pairs(ranges[1])
                parent(spatial2)[i, j] = field[iglobal, jglobal]
            end
            extension._reset_pencil_scalar_stats!()
            spectral2 = analysis(cfg, spatial2)
            reconstructed2 = synthesis(cfg, spectral2; prototype_θφ=spatial2)
            @test extension._pencil_scalar_stats().full_matrix_helper_calls == 0
            @test _collect_spatial(reconstructed2, cfg) ≈ field atol=3e-12 rtol=3e-12
        else
            @test true
        end
    else
        @test true
    end
    return nothing
end

function _all_ranks_catch(call, comm)
    message = try
        call()
        ""
    catch err
        "$(typeof(err)): $(sprint(showerror, err))"
    end
    reference = MPI.bcast(message, 0, comm)
    caught_same = !isempty(message) && message == reference
    total = MPI.Allreduce(caught_same ? 1 : 0, +, comm)
    MPI.Barrier(comm)
    return total == MPI.Comm_size(comm)
end

function test_collective_validation(adapter::MPIScalarAdapter)
    cfg = _scalar_config(:gauss, 3, 8)
    _, field = _closed_form_low_order(cfg, Float64)
    spatial = place(adapter, cfg, field, :spatial)
    coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    spectral = place(adapter, cfg, coefficients, :spectral)
    @test spectral_pencil_to_matrix(cfg, spectral; comm=adapter.comm) == coefficients

    malformed_pen = Pencil((cfg.lmax + 2, cfg.mmax + 1), adapter.comm)
    malformed = PencilArray{ComplexF64}(undef, malformed_pen)
    fill!(parent(malformed), 0)
    @test _all_ranks_catch(adapter.comm) do
        synthesis(cfg, malformed; prototype_θφ=spatial)
    end

    rank = MPI.Comm_rank(adapter.comm)
    malformed_matrix = rank == 0 ? coefficients :
        zeros(ComplexF64, cfg.lmax + 2, cfg.mmax + 1)
    @test _all_ranks_catch(adapter.comm) do
        matrix_to_spectral_pencil(cfg, malformed_matrix; comm=adapter.comm)
    end

    rank_varying_matrix = copy(coefficients)
    rank_varying_matrix[1, 1] = rank
    @test _all_ranks_catch(adapter.comm) do
        matrix_to_spectral_pencil(cfg, rank_varying_matrix; comm=adapter.comm)
    end

    helper_shape = rank == 0 ?
        (cfg.lmax + 1, cfg.mmax + 1) : (cfg.lmax + 2, cfg.mmax + 1)
    helper_malformed_pen = Pencil(helper_shape, adapter.comm)
    helper_malformed = PencilArray{ComplexF64}(undef, helper_malformed_pen)
    fill!(parent(helper_malformed), 0)
    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, helper_malformed)
    end

    varying_precision_matrix = rank == 0 ?
        zeros(ComplexF32, size(coefficients)) : coefficients
    @test _all_ranks_catch(adapter.comm) do
        matrix_to_spectral_pencil(cfg, varying_precision_matrix; comm=adapter.comm)
    end

    varying_precision_spectral = if rank == 0
        PencilArray{ComplexF32}(undef, pencil(spectral))
    else
        spectral
    end
    fill!(parent(varying_precision_spectral), 0)
    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, varying_precision_spectral)
    end

    wrong_decomposition_pen = Pencil(
        (cfg.lmax + 1, cfg.mmax + 1), (1,), adapter.comm,
    )
    wrong_decomposition = PencilArray{ComplexF64}(undef, wrong_decomposition_pen)
    fill!(parent(wrong_decomposition), 0)
    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, wrong_decomposition)
    end

    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, spectral; comm=MPI.COMM_SELF)
    end

    optional_comm = rank == 0 ? nothing : adapter.comm
    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, spectral; comm=optional_comm)
    end

    rank_varying = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    rank_varying[1, 1] = rank
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis(cfg, rank_varying; prototype_θφ=spatial)
    end

    optional_minus = rank == 0 ? copy(coefficients) : nothing
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis(
            cfg, coefficients; prototype_θφ=spatial,
            real_output=false, Aminus=optional_minus,
        )
    end

    malformed_spatial_pen = Pencil((cfg.nlat + 1, cfg.nlon), (1,), adapter.comm)
    malformed_spatial = PencilArray{Float64}(undef, malformed_spatial_pen)
    fill!(parent(malformed_spatial), 0)
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis(cfg, coefficients; prototype_θφ=malformed_spatial)
    end
    return nothing
end

adapter = MPIScalarAdapter(MPI.COMM_WORLD)
# Exercise every mathematical axis without taking their full Cartesian product:
# each Pencil construction owns an MPI Cartesian communicator, and a giant
# product needlessly exhausts MPI implementations with a 2048-context limit.
run_scalar_full_parity(
    adapter;
    grid_kinds=_SCALAR_GRID_KINDS,
    precisions=(Float32, Float64),
    mres_values=(1, 2),
    norms=(:orthonormal,),
    real_norm_values=(false,),
    cs_phase_values=(true,),
    pole_orders=(false,),
)
run_scalar_full_parity(
    adapter;
    grid_kinds=(:gauss,),
    precisions=(Float64,),
    mres_values=(1,),
    norms=(:orthonormal, :fourpi, :schmidt),
    real_norm_values=(false, true),
    cs_phase_values=(false, true),
    pole_orders=(false, true),
)
@testset "scalar full-grid parity φ-split complex Float32" begin
    test_phi_split_complex_float32(adapter)
end
@testset "Pencil-native scalar path" begin
    test_pencil_native_path(adapter)
end
@testset "collective scalar validation" begin
    test_collective_validation(adapter)
end

@testset "distributed scalar variant parity" begin
    cfg = create_gauss_config(
        5, 8; nlon=14, mres=2, norm=:schmidt,
        real_norm=true, cs_phase=false,
    )
    coefficients = _variant_coefficients(cfg, Float64)
    packed = SHTnsKit.pack_lm(cfg, coefficients)
    field = reshape(synthesis_packed(cfg, packed), cfg.nlat, cfg.nlon)
    spatial = place(adapter, cfg, field, :spatial)

    @testset "same-name packed and truncated Pencil APIs" begin
        packed_pencil = analysis_packed(cfg, spatial)
        @test packed_pencil isa PencilArray
        @test size_global(packed_pencil) == (cfg.nlm, 1)
        @test _collect_distributed_vector(packed_pencil) ≈ packed atol=3e-11 rtol=3e-11
        packed_field = synthesis_packed(
            cfg, packed_pencil; prototype_θφ=spatial,
        )
        @test packed_field isa PencilArray
        @test _collect_spatial(packed_field, cfg) ≈ field atol=3e-11 rtol=3e-11

        ltr_same_name = 3
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        extension._reset_pencil_scalar_stats!()
        truncated_pencil = analysis_packed_l(cfg, spatial, ltr_same_name)
        @test truncated_pencil isa PencilArray
        @test _collect_distributed_vector(truncated_pencil) ≈
              analysis_packed_l(cfg, vec(field), ltr_same_name) atol=3e-11 rtol=3e-11
        local_active_count = count(PencilArrays.range_local(pencil(truncated_pencil))[1]) do packed_index
            any(packed_index == LM_index(cfg.lmax, cfg.mres, l, m) + 1
                for m in 0:cfg.mres:min(cfg.mmax, ltr_same_name)
                for l in m:ltr_same_name)
        end
        @test extension._pencil_scalar_stats().analysis_packed_max_message_elements ==
              MPI.Allreduce(local_active_count, max, adapter.comm)
        noisy_pencil = _place_distributed_vector(copy(packed), adapter.comm)
        noisy_globals = collect(Int, PencilArrays.range_local(pencil(noisy_pencil))[1])
        for (i, packed_index) in pairs(noisy_globals)
            for m in 0:cfg.mres:cfg.mmax, l in max(m, ltr_same_name + 1):cfg.lmax
                if packed_index == LM_index(cfg.lmax, cfg.mres, l, m) + 1
                    parent(noisy_pencil)[i, 1] = 100 + 20im
                end
            end
        end
        extension._reset_pencil_scalar_stats!()
        low_field = synthesis_packed_l(
            cfg, noisy_pencil, ltr_same_name; prototype_θφ=spatial,
        )
        @test low_field isa PencilArray
        @test vec(_collect_spatial(low_field, cfg)) ≈
              synthesis_packed_l(cfg, packed, ltr_same_name) atol=3e-11 rtol=3e-11
        spectral_ranges = PencilArrays.range_local(
            create_spectral_pencil(cfg; comm=adapter.comm),
        )[2]
        local_synthesis_active = sum((
            ltr_same_name - m + 1 for m_index in spectral_ranges
            for m in (m_index - 1,)
            if m ≤ ltr_same_name && m % cfg.mres == 0
        ); init=0)
        @test extension._pencil_scalar_stats().synthesis_packed_max_message_elements ==
              MPI.Allreduce(local_synthesis_active, max, adapter.comm)
    end

    @testset "same-name complex packed degree limits" begin
        for T in (Float32, Float64)
            complex_cfg = create_gauss_config(
                5, 8; nlon=14, norm=:schmidt,
                real_norm=true, cs_phase=false,
            )
            complex_coefficients = zeros(
                Complex{T},
                nlm_cplx_calc(complex_cfg.lmax, complex_cfg.mmax, 1),
            )
            for l in 0:complex_cfg.lmax,
                m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax)
                complex_coefficients[
                    LM_cplx_index(complex_cfg.lmax, complex_cfg.mmax, l, m) + 1
                ] = Complex{T}(T(0.03 * (l + 1) - 0.01m), T(0.02m - 0.01l))
            end
            complex_field = synthesis_packed_cplx(complex_cfg, complex_coefficients)
            complex_spatial = place(
                adapter, complex_cfg, complex_field, :spatial,
            )
            ltr_complex = 3
            truncated = copy(complex_coefficients)
            noisy = copy(complex_coefficients)
            for l in (ltr_complex + 1):complex_cfg.lmax,
                m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax)
                index = LM_cplx_index(
                    complex_cfg.lmax, complex_cfg.mmax, l, m,
                ) + 1
                truncated[index] = 0
                noisy[index] = Complex{T}(T(90 + l), T(-70 + m))
            end

            extension._reset_pencil_scalar_stats!()
            analyzed = analysis_packed_cplx_l(
                complex_cfg, complex_spatial, ltr_complex,
            )
            @test analyzed isa PencilArray
            @test _collect_distributed_vector(analyzed) ≈ truncated atol=4e-4 rtol=4e-4
            starts = collect(Int, PencilArrays.range_local(pencil(analyzed))[1])
            local_active = count(starts) do packed_index
                any(packed_index == LM_cplx_index(
                        complex_cfg.lmax, complex_cfg.mmax, l, m,
                    ) + 1
                    for l in 0:ltr_complex
                    for m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax))
            end
            @test extension._pencil_scalar_stats().analysis_packed_max_message_elements ==
                  MPI.Allreduce(local_active, max, adapter.comm)

            noisy_pencil = _place_distributed_vector(noisy, adapter.comm)
            extension._reset_pencil_scalar_stats!()
            reconstructed = synthesis_packed_cplx_l(
                complex_cfg, noisy_pencil, ltr_complex;
                prototype_θφ=complex_spatial,
            )
            @test reconstructed isa PencilArray
            @test _collect_spatial(reconstructed, complex_cfg) ≈
                  synthesis_packed_cplx(complex_cfg, truncated) atol=4e-4 rtol=4e-4
            spectral_orders = PencilArrays.range_local(
                create_spectral_pencil(complex_cfg; comm=adapter.comm),
            )[2]
            local_unpack_active = sum((
                (m == 0 ? 1 : 2) * (ltr_complex - m + 1)
                for m_index in spectral_orders
                for m in (m_index - 1,) if m ≤ ltr_complex
            ); init=0)
            @test extension._pencil_scalar_stats().synthesis_packed_max_message_elements ==
                  MPI.Allreduce(local_unpack_active, max, adapter.comm)
            @test extension._pencil_scalar_stats().synthesis_packed_max_message_elements <
                  2nlm_cplx_calc(complex_cfg.lmax, complex_cfg.mmax, 1)

            # At ltr=0 the sole active entry is (l,m)=(0,0), so an exact
            # complex-packed unpack sends one value, not a padded ±m pair.
            for edge_ltr in (0, complex_cfg.lmax)
                extension._reset_pencil_scalar_stats!()
                edge_reconstructed = synthesis_packed_cplx_l(
                    complex_cfg, noisy_pencil, edge_ltr;
                    prototype_θφ=complex_spatial,
                )
                @test _collect_spatial(edge_reconstructed, complex_cfg) ≈
                      synthesis_packed_cplx_l(
                          complex_cfg, noisy, edge_ltr,
                      ) atol=4e-4 rtol=4e-4
                edge_local_payload = sum((
                    (m == 0 ? 1 : 2) * (edge_ltr - m + 1)
                    for m_index in spectral_orders
                    for m in (m_index - 1,) if m ≤ edge_ltr
                ); init=0)
                @test extension._pencil_scalar_stats().synthesis_packed_max_message_elements ==
                      MPI.Allreduce(edge_local_payload, max, adapter.comm)
            end

            rank = MPI.Comm_rank(adapter.comm)
            varying_ltr = rank == 0 ? ltr_complex : ltr_complex - 1
            @test _all_ranks_catch(adapter.comm) do
                analysis_packed_cplx_l(complex_cfg, complex_spatial, varying_ltr)
            end
        end
    end

    @testset "same-name axisymmetric and fixed-order Pencil APIs" begin
        axis_coefficients = ComplexF32[0.2, -0.1, 0.05, 0.03, -0.02, 0.01]
        axis_field = synthesis_axisym(cfg, axis_coefficients)
        axis_spatial = _place_distributed_vector(axis_field, adapter.comm)
        axis_analyzed = analysis_axisym(cfg, axis_spatial)
        @test axis_analyzed isa PencilArray
        @test _collect_distributed_vector(axis_analyzed) ≈ axis_coefficients atol=3f-5 rtol=3f-5
        axis_reconstructed = synthesis_axisym(cfg, axis_analyzed)
        @test axis_reconstructed isa PencilArray
        @test _collect_distributed_vector(axis_reconstructed) ≈ axis_field atol=3f-5 rtol=3f-5

        axis_l = analysis_axisym_l(cfg, axis_spatial, 3)
        @test size_global(axis_l) == (4, 1)
        @test _collect_distributed_vector(axis_l) ≈ axis_coefficients[1:4] atol=3f-5 rtol=3f-5
        axis_low = synthesis_axisym_l(cfg, axis_l, 3)
        @test _collect_distributed_vector(axis_low) ≈
              synthesis_axisym_l(cfg, axis_coefficients, 3) atol=3f-5 rtol=3f-5

        for im in 0:(cfg.mmax ÷ cfg.mres)
            m = im * cfg.mres
            mode_coefficients = ComplexF32[
                ComplexF32(0.03f0 * (l + 1), m == 0 ? 0 : -0.02f0 * (l + 1))
                for l in m:cfg.lmax
            ]
            mode_field = synthesis_packed_ml(cfg, im, mode_coefficients, cfg.lmax)
            mode_spatial = _place_distributed_vector(mode_field, adapter.comm)
            mode_analyzed = analysis_packed_ml(
                cfg, im, mode_spatial, cfg.lmax,
            )
            @test mode_analyzed isa PencilArray
            @test _collect_distributed_vector(mode_analyzed) ≈
                  mode_coefficients atol=3f-5 rtol=3f-5
            mode_reconstructed = synthesis_packed_ml(
                cfg, im, mode_analyzed, cfg.lmax,
            )
            @test mode_reconstructed isa PencilArray
            @test _collect_distributed_vector(mode_reconstructed) ≈
                  mode_field atol=3f-5 rtol=3f-5
        end
    end

    @testset "same-name batch Pencil APIs" begin
        for T in (Float32, Float64), nfields in (1, 2, 5)
            coefficients_t = Complex{T}.(coefficients)
            field_t = T.(field)
            fields = PencilArray{T}(undef, pencil(spatial), nfields)
            for k in 1:nfields
                @views parent(fields)[:, :, k] .= T(k) .* parent(spatial)
            end
            analyzed = analysis_batch(cfg, fields)
            @test analyzed isa PencilArray
            @test size_global(analyzed) ==
                  (cfg.lmax + 1, cfg.mmax + 1, nfields)
            analyzed_dense = zeros(Complex{T}, cfg.lmax + 1, cfg.mmax + 1, nfields)
            ranges = PencilArrays.range_local(pencil(analyzed))
            lglobals = collect(Int, ranges[1])
            mglobals = collect(Int, ranges[2])
            for k in 1:nfields, (j, mg) in pairs(mglobals), (i, lg) in pairs(lglobals)
                analyzed_dense[lg, mg, k] = parent(analyzed)[i, j, k]
            end
            MPI.Allreduce!(analyzed_dense, +, adapter.comm)
            for k in 1:nfields
                @test analyzed_dense[:, :, k] ≈ T(k) .* coefficients_t atol=3e-4 rtol=3e-4
            end
            analyzed_inplace = PencilArray{Complex{T}}(
                undef, pencil(analyzed), nfields,
            )
            @test analysis_batch!(cfg, analyzed_inplace, fields) === analyzed_inplace
            @test parent(analyzed_inplace) ≈ parent(analyzed) atol=3e-4 rtol=3e-4
            reconstructed = synthesis_batch(
                cfg, analyzed; prototype_θφ=fields,
            )
            @test reconstructed isa PencilArray
            @test size_global(reconstructed) == (cfg.nlat, cfg.nlon, nfields)
            for k in 1:nfields
                slice = PencilArray{T}(undef, pencil(spatial))
                @views parent(slice) .= parent(reconstructed)[:, :, k]
                @test _collect_spatial(slice, cfg) ≈ T(k) .* field_t atol=3e-4 rtol=3e-4
            end
            reconstructed_inplace = PencilArray{T}(
                undef, pencil(fields), nfields,
            )
            @test synthesis_batch!(
                cfg, reconstructed_inplace, analyzed;
                prototype_θφ=fields,
            ) === reconstructed_inplace
            @test parent(reconstructed_inplace) ≈
                  parent(reconstructed) atol=3e-4 rtol=3e-4

            complex_batch = synthesis_batch_cplx(
                cfg, analyzed; prototype_θφ=fields,
            )
            @test complex_batch isa PencilArray
            complex_reference = synthesis_batch_cplx(cfg, analyzed_dense)
            for k in 1:nfields
                slice = PencilArray{Complex{T}}(undef, pencil(spatial))
                @views parent(slice) .= parent(complex_batch)[:, :, k]
                @test _collect_spatial(slice, cfg) ≈
                      complex_reference[:, :, k] atol=3e-4 rtol=3e-4
            end
        end
    end

    @testset "distributed variant plans and collective validation" begin
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        plan = extension.DistAnalysisPlan(cfg, spatial)
        planned = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        dist_analysis!(plan, planned, spatial)
        @test planned ≈ coefficients atol=3e-11 rtol=3e-11
        fill!(planned, 0)
        @test analysis!(plan, planned, spatial) === planned
        @test planned ≈ coefficients atol=3e-11 rtol=3e-11

        spectral = analysis(cfg, spatial)
        synthesis_plan = extension.DistPlan(cfg, spatial)
        planned_spatial = similar(spatial)
        @test synthesis!(synthesis_plan, planned_spatial, spectral) === planned_spatial
        @test _collect_spatial(planned_spatial, cfg) ≈ field atol=3e-11 rtol=3e-11
        for m in 0:cfg.mmax
            m % cfg.mres == 0 && continue
            @test iszero(maximum(abs, @view planned[:, m + 1]))
        end

        complex_spatial = PencilArray{ComplexF64}(undef, pencil(spatial))
        parent(complex_spatial) .= complex.(parent(spatial), 0.1)
        @test _all_ranks_catch(adapter.comm) do
            analysis_packed(cfg, complex_spatial)
        end

        rank = MPI.Comm_rank(adapter.comm)
        overflowing_ltr = rank == 0 ? big(typemax(Int)) + 1 : big(3)
        @test _all_ranks_catch(adapter.comm) do
            analysis_packed_l(cfg, spatial, overflowing_ltr)
        end

        axis_field = synthesis_axisym(cfg, coefficients[:, 1])
        axis_spatial = _place_distributed_vector(axis_field, adapter.comm)
        varying_axis_ltr = rank == 0 ? 3 : 2
        @test _all_ranks_catch(adapter.comm) do
            analysis_axisym_l(cfg, axis_spatial, varying_axis_ltr)
        end

        mode_field = synthesis_packed_ml(cfg, 1, coefficients[3:end, 3], cfg.lmax)
        mode_spatial = _place_distributed_vector(mode_field, adapter.comm)
        varying_im = rank == 0 ? 1 : 2
        @test _all_ranks_catch(adapter.comm) do
            analysis_packed_ml(cfg, varying_im, mode_spatial, cfg.lmax)
        end

        varying_fields = PencilArray{Float64}(
            undef, pencil(spatial), rank == 0 ? 1 : 2,
        )
        fill!(parent(varying_fields), 0)
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch(cfg, varying_fields)
        end


        empty_fields = PencilArray{Float64}(undef, pencil(spatial), 0)
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch(cfg, empty_fields)
        end
        empty_coefficients = PencilArray{ComplexF64}(
            undef, create_spectral_pencil(cfg; comm=adapter.comm), 0,
        )
        empty_analysis_output = similar(empty_coefficients)
        empty_synthesis_output = PencilArray{Float64}(
            undef, pencil(spatial), 0,
        )
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch!(cfg, empty_analysis_output, empty_fields)
        end
        @test _all_ranks_catch(adapter.comm) do
            synthesis_batch(
                cfg, empty_coefficients; prototype_θφ=empty_fields,
            )
        end
        @test _all_ranks_catch(adapter.comm) do
            synthesis_batch_cplx(
                cfg, empty_coefficients; prototype_θφ=empty_fields,
            )
        end
        @test _all_ranks_catch(adapter.comm) do
            synthesis_batch!(
                cfg, empty_synthesis_output, empty_coefficients;
                prototype_θφ=empty_fields,
            )
        end

        malformed_pen = Pencil(
            (cfg.nlat + 1, cfg.nlon), (1,), adapter.comm,
        )
        malformed_real = PencilArray{Float64}(undef, malformed_pen)
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(adapter.comm) do
            synthesis_packed(
                cfg, _place_distributed_vector(packed, adapter.comm);
                prototype_θφ=malformed_real,
            )
        end
        MPI.Barrier(adapter.comm)
        malformed_stats = extension._pencil_scalar_stats()
        @test malformed_stats.synthesis_packed_max_message_elements == 0
        @test malformed_stats.synthesis_max_message_elements == 0

        complex_cfg = create_gauss_config(4, 7; nlon=12)
        complex_coefficients = zeros(
            ComplexF32,
            nlm_cplx_calc(complex_cfg.lmax, complex_cfg.mmax, 1),
        )
        complex_coefficients[
            LM_cplx_index(complex_cfg.lmax, complex_cfg.mmax, 2, -1) + 1
        ] = 0.2f0 - 0.1f0im
        distributed_complex = _place_distributed_vector(
            complex_coefficients, adapter.comm,
        )
        malformed_complex_pen = Pencil(
            (complex_cfg.nlat + 1, complex_cfg.nlon), (1,), adapter.comm,
        )
        malformed_complex = PencilArray{ComplexF32}(
            undef, malformed_complex_pen,
        )
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(adapter.comm) do
            synthesis_packed_cplx(
                complex_cfg, distributed_complex;
                prototype_θφ=malformed_complex,
            )
        end
        MPI.Barrier(adapter.comm)
        malformed_complex_stats = extension._pencil_scalar_stats()
        @test malformed_complex_stats.synthesis_packed_max_message_elements == 0
        @test malformed_complex_stats.synthesis_max_message_elements == 0

        wrong_complex_prototype = PencilArray{Float32}(
            undef,
            create_spatial_pencil(complex_cfg; comm=adapter.comm),
        )
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(adapter.comm) do
            synthesis_packed_cplx(
                complex_cfg, distributed_complex;
                prototype_θφ=wrong_complex_prototype,
            )
        end
        MPI.Barrier(adapter.comm)
        wrong_type_stats = extension._pencil_scalar_stats()
        @test wrong_type_stats.synthesis_packed_max_message_elements == 0
        @test wrong_type_stats.synthesis_max_message_elements == 0
    end

    @test dist_analysis_packed(cfg, spatial) ≈ packed atol=3e-11 rtol=3e-11
    local_full = dist_synthesis_packed(cfg, packed; prototype_θφ=spatial)
    full_pencil = PencilArray{eltype(local_full)}(undef, pencil(spatial))
    copyto!(parent(full_pencil), local_full)
    @test _collect_spatial(full_pencil, cfg) ≈ field atol=3e-11 rtol=3e-11

    ltr = 3
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    extension._reset_pencil_scalar_stats!()
    truncated = dist_analysis_packed(cfg, spatial; ltr)
    expected = analysis_packed_l(cfg, vec(field), ltr)
    @test truncated ≈ expected atol=3e-11 rtol=3e-11
    stats = extension._pencil_scalar_stats()
    spectral_pen = create_spectral_pencil(cfg; comm=adapter.comm)
    max_active_owned_orders = MPI.Allreduce(
        count(PencilArrays.range_local(spectral_pen)[2]) do m_index
            m = m_index - 1
            m <= ltr && m % cfg.mres == 0
        end,
        max, adapter.comm,
    )
    @test stats.full_matrix_helper_calls == 0
    @test stats.analysis_max_message_elements <=
          (ltr + 1) * max_active_owned_orders
    active_packed_count = sum(
        ltr - m + 1 for m in 0:cfg.mres:min(cfg.mmax, ltr)
    )
    @test stats.analysis_packed_max_message_elements == active_packed_count
    @test stats.analysis_packed_max_message_elements < cfg.nlm
    for m in 0:cfg.mmax
        m % cfg.mres == 0 || continue
        for l in max(m, ltr + 1):cfg.lmax
            @test iszero(truncated[LM_index(cfg.lmax, cfg.mres, l, m) + 1])
        end
    end

    high_noise = copy(packed)
    for m in 0:cfg.mmax
        m % cfg.mres == 0 || continue
        for l in max(m, ltr + 1):cfg.lmax
            high_noise[LM_index(cfg.lmax, cfg.mres, l, m) + 1] = 100 + 20im
        end
    end
    local_truncated = dist_synthesis_packed(
        cfg, high_noise; prototype_θφ=spatial, ltr,
    )
    truncated_pencil = PencilArray{eltype(local_truncated)}(undef, pencil(spatial))
    copyto!(parent(truncated_pencil), local_truncated)
    @test vec(_collect_spatial(truncated_pencil, cfg)) ≈
          synthesis_packed_l(cfg, high_noise, ltr) atol=3e-11 rtol=3e-11

    @test _all_ranks_catch(adapter.comm) do
        dist_analysis_packed(cfg, spatial; ltr=cfg.lmax + 1)
    end
    rank_ltr = MPI.Comm_rank(adapter.comm) == 0 ? ltr : ltr - 1
    @test _all_ranks_catch(adapter.comm) do
        dist_analysis_packed(cfg, spatial; ltr=rank_ltr)
    end
    invalid_synthesis_ltr = MPI.Comm_rank(adapter.comm) == 0 ? cfg.lmax + 1 : ltr
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis_packed(
            cfg, packed; prototype_θφ=spatial, ltr=invalid_synthesis_ltr,
        )
    end

    complex_cfg = create_gauss_config(3, 6; nlon=10)
    complex_coefficients = zeros(
        ComplexF64, nlm_cplx_calc(complex_cfg.lmax, complex_cfg.mmax, 1),
    )
    complex_coefficients[LM_cplx_index(3, 3, 2, -1) + 1] = 0.2 - 0.1im
    complex_coefficients[LM_cplx_index(3, 3, 3, 2) + 1] = -0.08 + 0.04im
    complex_field = synthesis_packed_cplx(complex_cfg, complex_coefficients)
    complex_spatial = place(adapter, complex_cfg, complex_field, :spatial)
    @test dist_analysis_packed_cplx(complex_cfg, complex_spatial) ≈
          complex_coefficients atol=4e-11 rtol=4e-11
    local_complex = dist_synthesis_packed_cplx(
        complex_cfg, complex_coefficients; prototype_θφ=complex_spatial,
    )
    complex_output = PencilArray{eltype(local_complex)}(undef, pencil(complex_spatial))
    copyto!(parent(complex_output), local_complex)
    @test _collect_spatial(complex_output, complex_cfg) ≈
          complex_field atol=4e-11 rtol=4e-11

    complex_coefficients32 = ComplexF32.(complex_coefficients)
    complex_field32 = synthesis_packed_cplx(complex_cfg, complex_coefficients32)
    complex_spatial32 = place(adapter, complex_cfg, complex_field32, :spatial)
    analyzed32 = dist_analysis_packed_cplx(complex_cfg, complex_spatial32)
    @test eltype(analyzed32) === ComplexF32
    @test analyzed32 ≈ complex_coefficients32 atol=3f-5 rtol=3f-5
    local_complex32 = dist_synthesis_packed_cplx(
        complex_cfg, complex_coefficients32; prototype_θφ=complex_spatial32,
    )
    @test eltype(local_complex32) === ComplexF32

    distributed_complex = analysis_packed_cplx(complex_cfg, complex_spatial32)
    @test distributed_complex isa PencilArray
    @test eltype(distributed_complex) === ComplexF32
    @test _collect_distributed_vector(distributed_complex) ≈
          complex_coefficients32 atol=3f-5 rtol=3f-5
    reconstructed_complex = synthesis_packed_cplx(
        complex_cfg, distributed_complex; prototype_θφ=complex_spatial32,
    )
    @test reconstructed_complex isa PencilArray
    @test _collect_spatial(reconstructed_complex, complex_cfg) ≈
          complex_field32 atol=3f-5 rtol=3f-5

    rank_coefficients = MPI.Comm_rank(adapter.comm) == 0 ?
        complex_coefficients[1:(end - 1)] : complex_coefficients
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis_packed_cplx(
            complex_cfg, rank_coefficients; prototype_θφ=complex_spatial,
        )
    end
    rank_complex_cfg = MPI.Comm_rank(adapter.comm) == 0 ?
        create_gauss_config(3, 6; nlon=10, mres=2) : complex_cfg
    @test _all_ranks_catch(adapter.comm) do
        dist_analysis_packed_cplx(rank_complex_cfg, complex_spatial)
    end
end

MPI.Barrier(MPI.COMM_WORLD)
