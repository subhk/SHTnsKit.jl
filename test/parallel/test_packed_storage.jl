# Non-MPI tests for packed storage utilities (LM_index, dense/packed conversions)
# These exercise serial functions from src/ without needing MPI.

using Test
using SHTnsKit

@testset "Packed Storage Utilities" begin

    @testset "Dense-packed roundtrip via LM_index" begin
        for lmax in [8, 16, 32]
            mmax = lmax
            mres = 1
            cfg = SHTnsKit.create_config(lmax; mmax=mmax)

            # Create random spectral coefficients in dense matrix form
            alm_dense = zeros(ComplexF64, lmax + 1, mmax + 1)
            for m in 0:mmax
                for l in m:lmax
                    alm_dense[l+1, m+1] = randn(ComplexF64)
                end
            end
            # m=0 should be real for a real field
            alm_dense[:, 1] .= real.(alm_dense[:, 1])

            # Convert dense → packed using LM_index
            nlm = cfg.nlm
            alm_packed = zeros(ComplexF64, nlm)
            for m in 0:mmax
                for l in m:lmax
                    idx = SHTnsKit.LM_index(lmax, mres, l, m) + 1
                    alm_packed[idx] = alm_dense[l+1, m+1]
                end
            end

            # Convert packed → dense
            alm_dense_rt = zeros(ComplexF64, lmax + 1, mmax + 1)
            for m in 0:mmax
                for l in m:lmax
                    idx = SHTnsKit.LM_index(lmax, mres, l, m) + 1
                    alm_dense_rt[l+1, m+1] = alm_packed[idx]
                end
            end

            # Verify roundtrip is exact
            @test alm_dense_rt == alm_dense

            # Verify packed array has the right length
            @test nlm == sum(lmax - m + 1 for m in 0:mmax)

            SHTnsKit.destroy_config(cfg)
        end
    end

    @testset "Dense-packed roundtrip with mmax < lmax" begin
        lmax = 32
        mmax = 16
        mres = 1
        cfg = SHTnsKit.create_config(lmax; mmax=mmax)

        alm_dense = zeros(ComplexF64, lmax + 1, mmax + 1)
        for m in 0:mmax, l in m:lmax
            alm_dense[l+1, m+1] = randn(ComplexF64)
        end
        alm_dense[:, 1] .= real.(alm_dense[:, 1])

        nlm = cfg.nlm
        alm_packed = zeros(ComplexF64, nlm)
        for m in 0:mmax, l in m:lmax
            idx = SHTnsKit.LM_index(lmax, mres, l, m) + 1
            alm_packed[idx] = alm_dense[l+1, m+1]
        end

        alm_dense_rt = zeros(ComplexF64, lmax + 1, mmax + 1)
        for m in 0:mmax, l in m:lmax
            idx = SHTnsKit.LM_index(lmax, mres, l, m) + 1
            alm_dense_rt[l+1, m+1] = alm_packed[idx]
        end

        @test alm_dense_rt == alm_dense
        @test nlm == sum(lmax - m + 1 for m in 0:mmax)

        SHTnsKit.destroy_config(cfg)
    end

    @testset "LM_index properties" begin
        lmax = 16
        mres = 1

        # All packed indices should be unique
        indices = Int[]
        for m in 0:lmax
            for l in m:lmax
                push!(indices, SHTnsKit.LM_index(lmax, mres, l, m))
            end
        end
        @test length(indices) == length(unique(indices))

        # Indices should span 0:(nlm-1) contiguously
        @test sort(indices) == collect(0:(length(indices) - 1))

        # Invalid arguments should throw
        @test_throws ArgumentError SHTnsKit.LM_index(lmax, mres, -1, 0)
        @test_throws ArgumentError SHTnsKit.LM_index(lmax, mres, 0, 1)   # l < m
        @test_throws ArgumentError SHTnsKit.LM_index(lmax, mres, lmax + 1, 0)  # l > lmax
    end

    @testset "Memory savings calculation" begin
        # Verify the expected memory formula inline
        # Dense storage: (lmax+1) * (mmax+1) complex numbers
        # Packed storage: only l >= m coefficients = sum_{m=0}^{mmax} (lmax - m + 1)
        for (lmax, mmax) in [(8, 8), (16, 16), (32, 32), (32, 16), (64, 64)]
            dense_elements = (lmax + 1) * (mmax + 1)
            packed_elements = sum(lmax - m + 1 for m in 0:mmax)

            # Packed should always be <= dense
            @test packed_elements <= dense_elements

            # When lmax == mmax, savings = mmax*(mmax+1)/2 wasted elements
            if lmax == mmax
                wasted = dense_elements - packed_elements
                expected_wasted = mmax * (mmax + 1) ÷ 2
                @test wasted == expected_wasted
            end

            # For lmax == mmax, savings percentage should be close to 50% for large lmax
            if lmax == mmax && lmax >= 32
                savings_pct = 100.0 * (dense_elements - packed_elements) / dense_elements
                @test savings_pct > 40.0  # Always > 40% for lmax >= 32
            end
        end

        # mmax == 0: no wasted elements
        @test sum(8 - m + 1 for m in 0:0) == (8 + 1) * (0 + 1)
    end
end
