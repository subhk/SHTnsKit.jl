#!/usr/bin/env julia
#
# MPI regression tests for the distributed defects found in the 2026-08 audit.
# Each testset pins one fix; every one of them either crashed, hung, or returned
# silently wrong numbers before.
#
#   1. dist_analysis_packed_cplx: LM_cplx negative-m rule (no (-1)^m) + complex input
#   2. packed storage: accumulation/normalization loops must stride by mres
#   3. sphtor table kernels: closed-form pole limits instead of the guarded 0
#   4. ranks owning zero φ columns (nranks > nlon) must not crash the φ gather
#   5. 2D (θ×φ) pencil: θ-slab reduction must not over-count φ-partners
#
# Run with: mpiexec -n 4 julia --project test/parallel/test_mpi_audit_fixes.jl

using MPI
MPI.Init()

using Test
using Random
using PencilArrays
using PencilFFTs
using SHTnsKit

const ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

root_println(args...) = (rank == 0 && (println(args...); flush(stdout)))

"""Scatter a globally-known matrix into the local block of `pen`."""
function scatter_field(pen::Pencil, F::AbstractMatrix)
    r = PencilArrays.range_local(pen)
    loc = Array{eltype(F)}(undef, length(r[1]), length(r[2]))
    for (jl, jg) in enumerate(r[2]), (il, ig) in enumerate(r[1])
        loc[il, jl] = F[ig, jg]
    end
    return PencilArray(pen, loc)
end

"""Max |local - F| over the rank's block of `pen`, reduced over all ranks."""
function max_local_error(local_block::AbstractMatrix, F::AbstractMatrix, pen::Pencil)
    r = PencilArrays.range_local(pen)
    err = 0.0
    for (jl, jg) in enumerate(r[2]), (il, ig) in enumerate(r[1])
        err = max(err, abs(local_block[il, jl] - F[ig, jg]))
    end
    return MPI.Allreduce(err, max, comm)
end

max_local_error(pa::PencilArray, F::AbstractMatrix) =
    max_local_error(parent(pa), F, pencil(pa))

@testset "MPI audit-fix regressions ($nprocs ranks)" begin

    @testset "dist_analysis_packed_cplx LM_cplx layout" begin
        lmax = 5
        nlat, nlon = lmax + 2, 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(20260807)
        pen = Pencil((nlat, nlon), (1,), comm)   # θ decomposition

        # Real field: the −m half is conj(a_{+m}) with NO (-1)^m. The old code's
        # (-1)^m flipped every odd-m coefficient.
        F = randn(rng, nlat, nlon)
        got = SHTnsKit.dist_analysis_packed_cplx(cfg, scatter_field(pen, F))
        ref = SHTnsKit.analysis_packed_cplx(cfg, complex.(F))
        @test isapprox(got, ref; rtol=1e-9, atol=1e-11)

        # Genuinely complex field (independent ±m): the old code fabricated the
        # −m half from the +m half and silently returned a symmetrized spectrum.
        Z = randn(rng, ComplexF64, nlat, nlon)
        gotc = SHTnsKit.dist_analysis_packed_cplx(cfg, scatter_field(pen, Z))
        refc = SHTnsKit.analysis_packed_cplx(cfg, Z)
        @test isapprox(gotc, refc; rtol=1e-9, atol=1e-11)
        @test !isapprox(gotc, got; rtol=1e-3)   # sanity: the two inputs differ

        # mres > 1 has no LM_cplx layout — fail loudly instead of mis-indexing.
        cfg2 = create_gauss_config(lmax, nlat; mmax=lmax, mres=2, nlon=nlon)
        @test_throws ArgumentError SHTnsKit.dist_analysis_packed_cplx(cfg2, scatter_field(pen, F))
        root_println("    [PASS] LM_cplx packed analysis")
    end

    @testset "packed storage strides by mres" begin
        lmax = mmax = 6
        mres = 2
        nlat, nlon = lmax + 2, max(2*mmax + 1, 4)
        # :fourpi also exercises the packed norm-conversion loop, which wrote
        # Alm_local[0] under @inbounds for every m that is not a multiple of mres.
        cfg = create_gauss_config(lmax, nlat; mmax=mmax, mres=mres, nlon=nlon, norm=:fourpi)
        rng = MersenneTwister(4242)
        F = randn(rng, nlat, nlon)
        pen = Pencil((nlat, nlon), (1,), comm)
        fpa = scatter_field(pen, F)

        packed = SHTnsKit.dist_analysis(cfg, fpa; use_packed_storage=true)
        dense = SHTnsKit.dist_analysis(cfg, fpa)
        info = ParExt.create_packed_storage_info(cfg)
        @test length(packed) == info.nlm_packed
        maxerr = 0.0
        for m in 0:mres:mmax, l in m:lmax
            maxerr = max(maxerr, abs(packed[info.lm_to_packed[l+1, m+1]] - dense[l+1, m+1]))
        end
        @test maxerr < 1e-11
        root_println("    [PASS] packed storage with mres=$mres")
    end

    @testset "sphtor pole limits on a pole-inclusive grid" begin
        lmax = 6
        nlat, nlon = lmax + 2, 2*lmax + 1
        cfg = create_regular_config(lmax, nlat; nlon=nlon, include_poles=true,
                                    precompute_plm=true)
        @test cfg.use_plm_tables                       # the table kernels are live
        @test minimum(abs.(1.0 .- abs.(cfg.x))) == 0.0 # grid really includes ±1

        rng = MersenneTwister(99)
        Slm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
        Tlm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
        for m in 0:cfg.mmax, l in max(1, m):lmax
            Slm[l+1, m+1] = randn(rng, ComplexF64)
            Tlm[l+1, m+1] = randn(rng, ComplexF64)
        end
        Slm[:, 1] .= real.(Slm[:, 1]); Tlm[:, 1] .= real.(Tlm[:, 1])

        # Serial reference uses the closed-form pole branch (src/kernels.jl).
        Vt, Vp = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        @test maximum(abs, Vt) > 1e-6                  # non-trivial reference

        pen = Pencil((nlat, nlon), (1,), comm)
        proto = scatter_field(pen, Vt)
        Vt_d, Vp_d = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm;
                                                    prototype_θφ=proto, real_output=true)
        # Before the fix the θ=0 and θ=π rows came back 0 from the table path.
        @test max_local_error(Vt_d, Vt, pen) < 1e-9
        @test max_local_error(Vp_d, Vp, pen) < 1e-9

        # Analysis direction reads the same tables.
        Sref, Tref = SHTnsKit.analysis_sphtor(cfg, Vt, Vp)
        S_d, T_d = SHTnsKit.dist_analysis_sphtor(cfg, scatter_field(pen, Vt),
                                                 scatter_field(pen, Vp))
        @test isapprox(S_d, Sref; rtol=1e-8, atol=1e-10)
        @test isapprox(T_d, Tref; rtol=1e-8, atol=1e-10)
        root_println("    [PASS] sphtor pole limits")
    end

    @testset "rank owning zero φ columns" begin
        lmax = mmax = 1
        nlat, nlon = 3, 3
        if nprocs <= nlon
            root_println("    [SKIP] zero-φ rank needs nprocs > nlon=$nlon")
        else
            cfg = create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon)
            rng = MersenneTwister(11)
            F = randn(rng, nlat, nlon)
            pen = Pencil((nlat, nlon), comm)   # decomposes φ → last ranks own nothing
            fpa = scatter_field(pen, F)
            # Used to BoundsError on the empty rank while its partners blocked in
            # the gather collective, hanging the job.
            alm = SHTnsKit.dist_analysis(cfg, fpa)
            @test isapprox(alm, SHTnsKit.analysis(cfg, F); rtol=1e-9, atol=1e-11)
            root_println("    [PASS] zero-φ rank")
        end
    end

    @testset "2D (θ×φ) pencil reduction" begin
        lmax = 6
        nlat, nlon = lmax + 2, 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        pen2 = try
            Pencil((nlat, nlon), (1, 2), comm)
        catch err
            root_println("    [SKIP] 2D pencil unsupported here: ", err)
            nothing
        end
        if pen2 === nothing
            @test true
        else
            r = PencilArrays.range_local(pen2)
            both_split = MPI.Allreduce(
                (length(r[1]) < nlat && length(r[2]) < nlon) ? 1 : 0, +, comm) == nprocs
            if !both_split
                root_println("    [SKIP] topology did not split both dims")
                @test true
            else
                rng = MersenneTwister(2026)
                F = randn(rng, nlat, nlon)
                fpa = scatter_field(pen2, F)
                # φ-partners of a θ-slab hold identical partials; over-counting
                # them scaled the whole spectrum by the φ-partition factor.
                @test isapprox(SHTnsKit.dist_analysis(cfg, fpa),
                               SHTnsKit.analysis(cfg, F); rtol=1e-9, atol=1e-11)

                Slm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
                Tlm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
                for m in 0:cfg.mmax, l in max(1, m):lmax
                    Slm[l+1, m+1] = randn(rng, ComplexF64)
                    Tlm[l+1, m+1] = randn(rng, ComplexF64)
                end
                Slm[:, 1] .= real.(Slm[:, 1]); Tlm[:, 1] .= real.(Tlm[:, 1])
                Vt, Vp = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
                Sref, Tref = SHTnsKit.analysis_sphtor(cfg, Vt, Vp)
                # This path used to Comm_split on first([]) with no guard at all.
                S_d, T_d = SHTnsKit.dist_analysis_sphtor(cfg, scatter_field(pen2, Vt),
                                                         scatter_field(pen2, Vp))
                @test isapprox(S_d, Sref; rtol=1e-8, atol=1e-10)
                @test isapprox(T_d, Tref; rtol=1e-8, atol=1e-10)
                root_println("    [PASS] 2D pencil reduction")
            end
        end
    end
end

MPI.Barrier(comm)
root_println("\nAll MPI audit-fix regression tests PASSED")
MPI.Finalize()
