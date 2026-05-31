#=
================================================================================
benchmark/transpose_scaling.jl  —  Task 9: Strong-scaling + load-balance
================================================================================

Measures for lmax ∈ {128, 256} and the number of MPI ranks it is launched with:

  (a) Transpose path (nlev=1):  dist_analysis!(plan,Alm,f) + dist_synthesis!(plan,f2,Alm)
  (b) Transpose batched (nlev=8, per-level amortised cost)
  (c) Dense-gather path:  SHTnsKit.dist_analysis(cfg,fθ) + dist_synthesis(cfg,A;…)
  (d) Legendre work imbalance across ranks (max/mean ratio of Σ_{m∈m_local}(lmax-m+1))

Timing protocol
---------------
* One warmup call (not timed) before each measurement block.
* Measurement: min over N_BENCH iterations of [MPI.Barrier + elapsed + Allreduce(MAX)].
  Using the MAX across ranks gives the slowest-rank wall time, which is what
  limits the application.

Architecture note (why the transpose path scales at larger rank counts)
-----------------------------------------------------------------------
The transpose path (DistTransposePlan):
  • Spectral data is m-distributed — each rank stores ONLY its own Alm slice.
    Memory per rank ∝ 1/np → no O(lmax²) replication.
  • One Alltoall collective (inside PencilFFTs ldiv!/mul!) per field/batch.
  • Legendre work is perfectly parallelised (each rank processes its own m-slice).

The dense-gather path (SHTnsKit.dist_analysis / dist_synthesis):
  • Every rank runs the FULL Legendre sum (no m-splitting).
  • Allreduce(Alm) after analysis copies O(lmax²) data — cost grows with lmax.
  • Adding ranks reduces the θ-local work but NOT the Legendre work or Allreduce.
  On ≤4 ranks on-node these paths can be comparable or the gather path faster
  (lower overhead); the architectural advantage of the transpose path only
  becomes decisive at large rank counts / cross-node scale.

Load-imbalance note
-------------------
Block m-distribution (PencilArrays default) assigns m=0…floor(mmax/np) to rank 0,
the next floor(mmax/np) values to rank 1, etc.  Because each m-bin has a different
Legendre work  W(m) = lmax − m + 1  (decreasing in m), low-m ranks do MORE work.
Imbalance ratio = max_rank_work / mean_rank_work.
A ratio ≤ ~1.4 is acceptable for most use-cases; the theoretical fix is an
interleaved (round-robin) m-distribution that equalises W(m) across ranks,
but this requires a custom PencilArrays distribution and is left as future work.

Results summary (single node, 8-core Apple M-series, Julia 1.11.1)
--------------------------------------------------------------------
lmax  np  transpose_ms  batched/lev_ms  gather_ms  m_imbalance(max/mean)
 128   1      2.83            2.75          20.95        1.000
 128   2      1.90            1.85          11.24        1.488
 128   4      1.19            1.13           8.04        1.733
 256   1     18.55           18.01         136.26        1.000
 256   2     12.52           12.55          70.65        1.494
 256   4      7.59            7.52          38.75        1.741

Speedup (transpose path, lmax=128): 1→2: 1.49×   1→4: 2.38×
Speedup (transpose path, lmax=256): 1→2: 1.48×   1→4: 2.44×

Ideal linear speedup would be 2×/4×; actual on-node speedup is sub-linear
(~1.5× at 2 ranks, ~2.4× at 4 ranks) because on-node Alltoall latency is
non-negligible relative to the Legendre work at these lmax values.

Batched/level vs per-field:
  lmax=128: batched=2.75ms ≤ single=2.83ms (3% saving at 1 rank)
  lmax=256: batched=18.01ms ≤ single=18.55ms (3% saving at 1 rank)
  Savings are modest for nlev=8 at these lmax values because the Alltoall
  dominates over repeated setup overhead.

Transpose vs gather at 4 ranks:
  lmax=128: 1.19ms vs  8.04ms → transpose is 6.8× faster
  lmax=256: 7.59ms vs 38.75ms → transpose is 5.1× faster
  The transpose path is dramatically faster even on-node because the gather
  path replicates the entire Legendre sum on every rank (no m-splitting) and
  adds an O(lmax²) Allreduce; the transpose path parallelises all of that.

Load-imbalance verdict:
  2 ranks: 1.49–1.49 (just above the ~1.4 guideline)
  4 ranks: 1.73–1.74 (clearly above; low-m ranks carry ~74% more Legendre work)
  CONCLUSION: Block m-distribution is UNACCEPTABLE at ≥ 4 ranks for production
  workloads.  Interleaved (round-robin) m-ownership — which equally distributes
  W(m) = lmax−m+1 across ranks — is needed and is the recommended follow-up task.
================================================================================
=#

using MPI
MPI.Init()

using SHTnsKit
using PencilArrays
using PencilFFTs
using LinearAlgebra
using Random

const comm  = MPI.COMM_WORLD
const rank  = MPI.Comm_rank(comm)
const np    = MPI.Comm_size(comm)

# ---------------------------------------------------------------------------
# Timing helper: min over N runs of slowest-rank wall time.
# ---------------------------------------------------------------------------
const N_BENCH = 20

function bench(g; n=N_BENCH)
    g()   # warmup
    best = Inf
    for _ in 1:n
        MPI.Barrier(comm)
        t = @elapsed g()
        t_max = MPI.Allreduce(t, MPI.MAX, comm)
        best = min(best, t_max)
    end
    return best
end

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

for lmax in (128, 256)
    nlat = lmax + 2
    nlon = 2 * lmax + 1
    cfg  = create_gauss_config(lmax, nlat; nlon=nlon)

    rank == 0 && println("\n--- lmax=$lmax  np=$np  nlat=$nlat  nlon=$nlon ---")
    flush(stdout)

    # -----------------------------------------------------------------------
    # (a) Transpose path — nlev=1
    # -----------------------------------------------------------------------
    plan  = DistTransposePlan(cfg; comm=comm, nlev=1, use_rfft=true)
    f     = allocate_spatial(plan)
    Alm   = allocate_spectral(plan)
    f2    = allocate_spatial(plan)

    # Correctness sanity-check: build a bandlimited field using serial SHTnsKit.synthesis,
    # fill the distributed spatial array, run dist_analysis!, then compare against the
    # serial SHTnsKit.analysis reference.  Only the analysis direction is checked; the
    # synthesis correctness is covered by the existing test suite.
    a0 = zeros(ComplexF64, lmax + 1, lmax + 1)
    for m in 0:lmax, l in m:lmax
        sc = 1.0 / (1 + l)^2
        a0[l+1, m+1] = m == 0 ? complex(sc) : complex(sc, 0.5sc)
    end
    f_full = SHTnsKit.synthesis(cfg, a0; real_output = true)  # (nlat, nlon) on every rank
    a_ref  = SHTnsKit.analysis(cfg, f_full)                   # (lmax+1, mmax+1) serial ref

    # Fill distributed spatial PencilArray.
    # parent(f) layout: (n_phi_local, n_theta_local, nlev=1)
    # range_local gives (phi_range_1based, theta_range_1based)
    r = PencilArrays.range_local(pencil(f))
    for (il, ig) in enumerate(r[2]), (jl, jg) in enumerate(r[1])
        parent(f)[jl, il, 1] = f_full[ig, jg]
    end

    dist_analysis!(plan, Alm, f)

    err_local = 0.0
    for (mi, m) in enumerate(plan.m_local), l in m:lmax
        err_local = max(err_local, abs(parent(Alm)[l+1, mi, 1] - a_ref[l+1, m+1]))
    end
    err_global = MPI.Allreduce(err_local, MPI.MAX, comm)
    if rank == 0
        if err_global > 1e-6
            @error "Transpose round-trip error too large: $err_global  (lmax=$lmax, np=$np)"
        else
            println("  round-trip err=$err_global  [OK]")
        end
    end
    # Prepare a random spatial field for timing (analysis→synthesis cycle)
    rand!(parent(f))

    tT = bench(() -> (dist_analysis!(plan, Alm, f); dist_synthesis!(plan, f2, Alm)))

    # Load-imbalance measurement
    wloc = sum(lmax - m + 1 for m in plan.m_local; init=0)
    nm   = length(plan.m_local)
    wmax = MPI.Allreduce(wloc, MPI.MAX, comm)
    wmin = MPI.Allreduce(wloc, MPI.MIN, comm)
    wsum = MPI.Allreduce(wloc,    +,    comm)
    nm_max = MPI.Allreduce(nm,   MPI.MAX, comm)
    nm_min = MPI.Allreduce(nm,   MPI.MIN, comm)

    # -----------------------------------------------------------------------
    # (b) Batched — nlev=8, report per-level amortised cost
    # -----------------------------------------------------------------------
    plan8 = DistTransposePlan(cfg; comm=comm, nlev=8, use_rfft=true)
    f8    = allocate_spatial(plan8);  rand!(parent(f8))
    A8    = allocate_spectral(plan8)
    g8    = allocate_spatial(plan8)

    dist_analysis!(plan8, A8, f8)
    dist_synthesis!(plan8, g8, A8)

    tB_total = bench(() -> (dist_analysis!(plan8, A8, f8); dist_synthesis!(plan8, g8, A8)))
    tB = tB_total / 8     # per-level cost

    # -----------------------------------------------------------------------
    # (c) Dense-gather path (existing SHTnsKit.dist_analysis / dist_synthesis)
    # -----------------------------------------------------------------------
    # Build a θ-decomposed spatial PencilArray: global (nlat, nlon), θ distributed.
    pen_g  = Pencil((nlat, nlon), (1,), comm)
    fθ     = PencilArray(pen_g, rand(Float64, PencilArrays.size_local(pen_g)...))

    # Warmup outside bench() so allocations don't pollute first timing call.
    A_g  = SHTnsKit.dist_analysis(cfg, fθ)
    _fr  = SHTnsKit.dist_synthesis(cfg, A_g; prototype_θφ=fθ, real_output=true)

    tG = bench(() -> begin
        A_g2 = SHTnsKit.dist_analysis(cfg, fθ)
        SHTnsKit.dist_synthesis(cfg, A_g2; prototype_θφ=fθ, real_output=true)
    end)

    # -----------------------------------------------------------------------
    # Print table row
    # -----------------------------------------------------------------------
    if rank == 0
        imb  = (wsum > 0) ? wmax / (wsum / np) : 1.0
        println(string(
            "RESULT  lmax=", lmax,
            "  np=",    np,
            "  transpose_ms=",       round(tT  * 1e3, digits=2),
            "  batched/lev_ms=",     round(tB  * 1e3, digits=2),
            "  gather_ms=",          round(tG  * 1e3, digits=2),
            "  m_local_range=[", nm_min, ",", nm_max, "]",
            "  legendre_work_range=[", wmin, ",", wmax, "]",
            "  m_imbalance(max/mean)=", round(imb, digits=3),
        ))
        flush(stdout)
    end

end   # lmax loop

rank == 0 && println("\nDone.")
MPI.Finalize()
