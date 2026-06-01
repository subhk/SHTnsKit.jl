# SHTnsKit.jl - shtns.h compatibility flag constant tests
#
# Pins the exported SHT_* flag constants (src/api_compat.jl) to their documented
# bit values so accidental edits to the shtns.h-compatibility layer are caught.

using Test
using SHTnsKit

@testset "shtns.h flag constants" begin
    @testset "layout flags" begin
        @test SHTnsKit.SHT_NATIVE_LAYOUT   == 0
        @test SHTnsKit.SHT_THETA_CONTIGUOUS == 256
        @test SHTnsKit.SHT_PHI_CONTIGUOUS  == 512
    end

    @testset "option flags (bit positions)" begin
        @test SHTnsKit.SHT_SCALAR_ONLY   == 256 * 16
        @test SHTnsKit.SHT_LOAD_SAVE_CFG == 256 * 64
        @test SHTnsKit.SHT_ALLOW_GPU     == 256 * 128
        @test SHTnsKit.SHT_FP32          == 256 * 1024
    end

    @testset "option flags are single distinct bits" begin
        opt = [SHTnsKit.SHT_SCALAR_ONLY, SHTnsKit.SHT_LOAD_SAVE_CFG,
               SHTnsKit.SHT_ALLOW_GPU, SHTnsKit.SHT_FP32]
        # each a power of two
        for f in opt
            @test f > 0 && (f & (f - 1)) == 0
        end
        # mutually non-overlapping
        @test reduce(|, opt) == sum(opt)
        @test length(unique(opt)) == length(opt)
    end

    @testset "flags OR-compose with grid type" begin
        flags = SHTnsKit.SHT_REGULAR | SHTnsKit.SHT_SCALAR_ONLY | SHTnsKit.SHT_ALLOW_GPU
        @test (flags & SHTnsKit.SHT_SCALAR_ONLY) != 0
        @test (flags & SHTnsKit.SHT_ALLOW_GPU) != 0
        @test (flags % 256) == SHTnsKit.SHT_REGULAR   # grid type in low byte
    end
end
