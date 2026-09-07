#!/usr/bin/env julia

using MPI
MPI.Init()

using Test
using PencilArrays
using PencilFFTs
using SHTnsKit

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

"""Copy a globally replicated matrix into the block owned by `pen`."""
function scatter_spectral(pen::Pencil, A::AbstractMatrix)
    ranges = PencilArrays.range_local(pen)
    block = Array{eltype(A)}(undef, PencilArrays.size_local(pen))
    for (jm, gm) in enumerate(ranges[2]), (il, gl) in enumerate(ranges[1])
        block[il, jm] = A[gl, gm]
    end
    return PencilArray(pen, block)
end

@testset "parallel local-evaluation contracts ($nprocs ranks)" begin
    lmax = mmax = 6
    cfg = create_gauss_config(lmax, lmax + 2; mmax, nlon=2mmax + 1)
    spectral_dims = (lmax + 1, mmax + 1)
    pen_m = Pencil(spectral_dims, comm)

    @testset "complex latitude evaluation is one-sided complex synthesis" begin
        A = zeros(ComplexF64, spectral_dims)
        A[5, 3] = 0.7 - 0.4im # (l,m) = (4,2), deliberately non-real
        A_p = scatter_spectral(pen_m, A)
        ilat = 3

        got = SHTnsKit.dist_SH_to_lat(
            cfg, A_p, cfg.x[ilat]; nphi=cfg.nlon, real_output=false)
        ref = vec(SHTnsKit.synthesis(cfg, A; real_output=false)[ilat, :])

        @test eltype(got) <: Complex
        @test isapprox(got, ref; rtol=1e-11, atol=1e-12)
        @test maximum(abs, imag.(got)) > 1e-4
    end

    @testset "configured global spectral dimensions are enforced" begin
        bad_dims = (lmax, mmax + 1)
        bad_pen = Pencil(bad_dims, comm)
        bad = scatter_spectral(bad_pen, zeros(ComplexF64, bad_dims))

        @test_throws DimensionMismatch SHTnsKit.dist_SH_to_point(cfg, bad, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SH_to_lat(cfg, bad, 0.2)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_point(
            cfg, bad, bad, bad, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_lat(
            cfg, bad, bad, bad, 0.2)
    end

    @testset "Q/S/T layouts and communicator groups must match" begin
        zeros_global = zeros(ComplexF64, spectral_dims)
        Q = scatter_spectral(pen_m, zeros_global)

        # Same global dimensions and communicator, but a different distributed
        # logical dimension, so the local ranges do not describe the same modes.
        pen_l = Pencil(spectral_dims, (1,), comm)
        S_l = scatter_spectral(pen_l, zeros_global)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_point(
            cfg, Q, S_l, Q, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_lat(
            cfg, Q, S_l, Q, 0.2)

        # Logical ownership matches, but memory order does not. Mixing these
        # arrays in one component-wise kernel is rejected explicitly.
        pen_perm = Pencil(spectral_dims, comm; permute=Permutation(2, 1))
        S_perm = PencilArray{ComplexF64}(undef, pen_perm)
        fill!(parent(S_perm), 0)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_point(
            cfg, Q, S_perm, Q, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_lat(
            cfg, Q, S_perm, Q, 0.2)

        # COMM_SELF has a different group even though the global dimensions
        # agree. The reduction communicator must be shared by every component.
        # Communicator preflight raises ArgumentError before layout validation;
        # DimensionMismatch is reserved here for the shape/layout cases above.
        pen_self = Pencil(spectral_dims, MPI.COMM_SELF)
        S_self = scatter_spectral(pen_self, zeros_global)
        @test_throws ArgumentError SHTnsKit.dist_SHqst_to_point(
            cfg, Q, S_self, Q, 0.2, 0.4)
        @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
            cfg, Q, S_self, Q, 0.2)
    end

    @testset "latitude truncation validation matches serial helpers" begin
        Z = scatter_spectral(pen_m, zeros(ComplexF64, spectral_dims))
        for bad_ltr in (-1, lmax + 1)
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(cfg, Z, 0.2; ltr=bad_ltr)
            @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
                cfg, Z, Z, Z, 0.2; ltr=bad_ltr)
        end
        for bad_mtr in (-1, mmax + 1)
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(cfg, Z, 0.2; mtr=bad_mtr)
            @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
                cfg, Z, Z, Z, 0.2; mtr=bad_mtr)
        end
    end

    @testset "collective evaluation arguments are validated" begin
        Z = scatter_spectral(pen_m, zeros(ComplexF64, spectral_dims))

        # A zero-length longitude vector is not a meaningful latitude sweep.
        @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(cfg, Z, 0.2; nphi=0)
        @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
            cfg, Z, Z, Z, 0.2; nphi=0)

        # These routines reduce partial modal sums across ranks, so every rank
        # must evaluate the same function with the same configuration.
        cfg_divergent = create_gauss_config(
            lmax, lmax + 2; mmax, nlon=2mmax + 1,
            mres=(rank == nprocs - 1 ? 2 : 1),
        )
        @test_throws ArgumentError SHTnsKit.dist_SH_to_point(
            cfg_divergent, Z, 0.2, 0.4)

        if nprocs > 1
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(
                cfg, Z, 0.2;
                nphi=(rank == nprocs - 1 ? cfg.nlon - 1 : cfg.nlon),
            )
            @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
                cfg, Z, Z, Z, 0.2;
                nphi=(rank == nprocs - 1 ? cfg.nlon - 1 : cfg.nlon),
            )
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(
                cfg, Z, 0.2;
                real_output=(rank != nprocs - 1),
            )
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(
                cfg, Z, 0.2;
                ltr=(rank == nprocs - 1 ? lmax - 1 : lmax),
            )
            @test_throws ArgumentError SHTnsKit.dist_SHqst_to_point(
                cfg, Z, Z, Z, rank == nprocs - 1 ? 0.3 : 0.2, 0.4)
        end
    end

    @testset "composite spatial operators keep the input communicator" begin
        spatial_dims = (cfg.nlat, cfg.nlon)
        Q = zeros(ComplexF64, spectral_dims)
        S = similar(Q); fill!(S, 0)
        T = similar(Q); fill!(T, 0)
        Q[3, 1], Q[4, 2] = 0.3, 0.4 - 0.2im
        S[2, 1], S[5, 3] = -0.2, 0.3 + 0.1im
        T[3, 1], T[4, 2] = 0.1, -0.2 + 0.4im
        scalar_values = synthesis(cfg, Q)
        theta_values, phi_values = synthesis_sphtor(cfg, S, T)
        degree_factors = [-l * (l + 1) for l in 0:lmax]
        input_pen = Pencil(spatial_dims, (1,), comm)
        input = scatter_spectral(input_pen, scalar_values)
        theta_input = scatter_spectral(input_pen, theta_values)

        duplicate_a = MPI.Comm_dup(comm)
        duplicate_b = MPI.Comm_dup(comm)
        try
            pen_a = Pencil(spatial_dims, (1,), duplicate_a)
            pen_b = Pencil(spatial_dims, (1,), duplicate_b)
            peer_a = scatter_spectral(pen_a, phi_values)
            peer_b = scatter_spectral(pen_b, phi_values)

            # Every candidate communicator is congruent to `comm`, but choosing
            # a different duplicate on each rank makes it unsafe as a collective
            # context. All composite stages must stay on `input`'s communicator.
            peer = iseven(rank) ? peer_a : peer_b
            for decomposition in ((1,), (2,))
                output_pen_a = Pencil(spatial_dims, decomposition, duplicate_a)
                output_pen_b = Pencil(spatial_dims, decomposition, duplicate_b)
                output_pen = iseven(rank) ? output_pen_b : output_pen_a
                ranges = PencilArrays.range_local(output_pen)
                for (use_rfft, real_output) in ((false, true), (true, true), (false, false))
                    @testset "output=$decomposition, rfft=$use_rfft, real=$real_output" begin
                        output = PencilArray{real_output ? Float64 : ComplexF64}(undef, output_pen)
                        fill!(parent(output), 0)
                        expected_divergence = synthesis(cfg, degree_factors .* S; real_output)[ranges...]
                        expected_vorticity = synthesis(cfg, degree_factors .* T; real_output)[ranges...]
                        expected_laplacian = synthesis(cfg, degree_factors .* Q; real_output)[ranges...]
                        @test SHTnsKit.dist_spatial_divergence(
                            cfg, theta_input, peer; prototype_θφ=output, use_rfft, real_output,
                        ) ≈ expected_divergence rtol=1e-11 atol=1e-12
                        @test SHTnsKit.dist_spatial_vorticity(
                            cfg, theta_input, peer; prototype_θφ=output, use_rfft, real_output,
                        ) ≈ expected_vorticity rtol=1e-11 atol=1e-12
                        @test SHTnsKit.dist_scalar_laplacian(
                            cfg, input; prototype_θφ=output, use_rfft, real_output,
                        ) ≈ expected_laplacian rtol=1e-11 atol=1e-12
                        @test SHTnsKit.dist_scalar_laplacian!(
                            cfg, output, input; use_rfft, real_output,
                        ) === output
                        @test parent(output) ≈ expected_laplacian rtol=1e-11 atol=1e-12
                    end
                end
            end

            self_pen = Pencil(spatial_dims, (1,), MPI.COMM_SELF)
            self_output = scatter_spectral(self_pen, zeros(Float64, spatial_dims))
            incongruent_output = rank == 0 ? self_output : peer_a
            output_before = copy(parent(incongruent_output))
            @test_throws ArgumentError SHTnsKit.dist_spatial_divergence(
                cfg, input, input; prototype_θφ=incongruent_output,
            )
            @test_throws ArgumentError SHTnsKit.dist_spatial_vorticity(
                cfg, input, input; prototype_θφ=incongruent_output,
            )
            @test_throws ArgumentError SHTnsKit.dist_scalar_laplacian(
                cfg, input; prototype_θφ=incongruent_output,
            )
            @test_throws ArgumentError SHTnsKit.dist_scalar_laplacian!(
                cfg, incongruent_output, input,
            )
            @test parent(incongruent_output) == output_before
        finally
            Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)._safe_comm_free(
                duplicate_a,
            )
            Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)._safe_comm_free(
                duplicate_b,
            )
        end
    end
end

rank == 0 && println("ParallelLocal correctness regression tests complete")
