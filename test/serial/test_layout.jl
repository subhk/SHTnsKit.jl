# SHTnsKit.jl - Layout and Packed Index Tests
# Additional tests for packed layout functions beyond test_indexing.jl

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Layout Utilities" begin
    @testset "nlm_calc validation" begin
        @test_throws ArgumentError nlm_calc(-1, 0, 1)
        @test_throws ArgumentError nlm_calc(0, -1, 1)
        @test_throws ArgumentError nlm_calc(0, 0, 0)

        # mmax > lmax returns 0
        @test nlm_calc(5, 6, 1) == 0
    end

    @testset "nlm_cplx_calc validation" begin
        @test_throws ArgumentError nlm_cplx_calc(-1, 0, 1)
        @test_throws ArgumentError nlm_cplx_calc(0, 0, 2)  # mres != 1

        # mmax > lmax returns 0
        @test nlm_cplx_calc(5, 6, 1) == 0
    end

    @testset "nlm_calc with mres" begin
        # mres=2: only even m values
        lmax = 10
        nlm2 = nlm_calc(lmax, lmax, 2)
        # Count manually
        count = 0
        for m in 0:2:lmax
            count += lmax - m + 1
        end
        @test nlm2 == count

        # mres=3
        nlm3 = nlm_calc(lmax, lmax, 3)
        count3 = 0
        for m in 0:3:lmax
            count3 += lmax - m + 1
        end
        @test nlm3 == count3
    end

    @testset "LM_index validation" begin
        @test_throws ArgumentError LM_index(5, 1, 2, -1)   # negative m
        @test_throws ArgumentError LM_index(5, 2, 2, 1)    # m not multiple of mres
        @test_throws ArgumentError LM_index(5, 1, 1, 3)    # l < m
        @test_throws ArgumentError LM_index(5, 1, 6, 0)    # l > lmax
    end

    @testset "LiM_index validation" begin
        @test_throws ArgumentError LiM_index(5, 1, 2, -1)  # negative im
        @test_throws ArgumentError LiM_index(5, 1, 1, 3)   # l < im*mres
    end

    @testset "LM_index and LiM_index equivalence" begin
        lmax = 8
        mres = 1
        for m in 0:lmax
            im = m ÷ mres
            for l in m:lmax
                @test LM_index(lmax, mres, l, m) == LiM_index(lmax, mres, l, im)
            end
        end

        mres = 2
        for im in 0:(lmax ÷ mres)
            m = im * mres
            for l in m:lmax
                @test LM_index(lmax, mres, l, m) == LiM_index(lmax, mres, l, im)
            end
        end
    end

    @testset "build_li_mi consistency" begin
        lmax = 8
        mmax = 8
        mres = 1
        li, mi = build_li_mi(lmax, mmax, mres)
        nlm = nlm_calc(lmax, mmax, mres)

        @test length(li) == nlm
        @test length(mi) == nlm

        # Verify each entry corresponds to a valid (l,m) pair
        for k in 1:nlm
            @test li[k] >= mi[k]
            @test li[k] <= lmax
            @test mi[k] >= 0
            @test mi[k] <= mmax
        end

        # Verify it covers all valid pairs
        pairs = Set{Tuple{Int,Int}}()
        for k in 1:nlm
            push!(pairs, (li[k], mi[k]))
        end
        @test length(pairs) == nlm

        for m in 0:mmax, l in m:lmax
            @test (l, m) in pairs
        end
    end

    @testset "im_from_lm validation" begin
        @test_throws ArgumentError im_from_lm(-1, 5, 1)
        @test_throws ArgumentError im_from_lm(100, 5, 1)  # out of range
    end

    @testset "im_from_lm with mres > 1" begin
        lmax = 10
        mres = 2
        for m in 0:mres:lmax
            im = m ÷ mres
            for l in m:lmax
                lm = LM_index(lmax, mres, l, m)
                @test im_from_lm(lm, lmax, mres) == im
            end
        end
    end
end
