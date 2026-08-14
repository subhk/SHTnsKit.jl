using Test
using ChainRulesCore
using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit

MPI.Initialized() || MPI.Init()

# A deliberately non-Array backing store. Any attempt to execute a transform
# before the reverse-rule storage guard trips will touch this array and fail
# with an error other than BackendUnavailableError.
struct FakeVendorArray{T,N} <: AbstractArray{T,N}
    storage::Array{T,N}
end

FakeVendorArray{T}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N} =
    FakeVendorArray{T,N}(Array{T,N}(undef, dims))
FakeVendorArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    FakeVendorArray{T}(undef, dims...)

Base.size(value::FakeVendorArray) = size(value.storage)
Base.IndexStyle(::Type{<:FakeVendorArray}) = IndexLinear()
Base.getindex(::FakeVendorArray, ::Int) = error("fake vendor storage was read")
Base.setindex!(::FakeVendorArray, _, ::Int) = error("fake vendor storage was written")
Base.similar(value::FakeVendorArray, ::Type{T}=eltype(value), dims::Dims=size(value)) where {T} =
    FakeVendorArray{T}(undef, dims)

@testset "Parallel AD storage boundary" begin
    cfg = create_gauss_config(3, 5; nlon=7)
    host_pencil = Pencil((cfg.nlat, cfg.nlon), MPI.COMM_SELF)
    field = PencilArray{Float64}(undef, host_pencil)
    parent(field) .= synthesis(cfg, reshape(
        ComplexF64[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        4, 4,
    ); real_output=true)

    scalar_coefficients, scalar_analysis_pb = rrule(dist_analysis, cfg, field)
    @test scalar_coefficients isa Matrix{ComplexF64}
    scalar_analysis_tangent = scalar_analysis_pb(one.(scalar_coefficients))[3]
    @test scalar_analysis_tangent isa PencilArray
    @test parent(scalar_analysis_tangent) isa Array

    scalar_field, scalar_synthesis_pb = rrule(
        dist_synthesis, cfg, scalar_coefficients; prototype_θφ=field,
    )
    scalar_synthesis_tangent = scalar_synthesis_pb(scalar_field)[3]
    @test scalar_synthesis_tangent isa Matrix{ComplexF64}

    vector_coefficients, vector_analysis_pb = rrule(
        dist_analysis_sphtor, cfg, field, field,
    )
    S, T = vector_coefficients
    vector_analysis_tangent = vector_analysis_pb((one.(S), one.(T)))
    @test vector_analysis_tangent[3] isa PencilArray
    @test vector_analysis_tangent[4] isa PencilArray
    @test parent(vector_analysis_tangent[3]) isa Array
    @test parent(vector_analysis_tangent[4]) isa Array

    vector_field, vector_synthesis_pb = rrule(
        dist_synthesis_sphtor, cfg, S, T; prototype_θφ=field,
    )
    vector_synthesis_tangent = vector_synthesis_pb(vector_field)
    @test vector_synthesis_tangent[3] isa Matrix{ComplexF64}
    @test vector_synthesis_tangent[4] isa Matrix{ComplexF64}

    fake_pencil = Pencil(FakeVendorArray, (cfg.nlat, cfg.nlon), MPI.COMM_SELF)
    fake = PencilArray{Float64}(undef, fake_pencil)

    @test_throws BackendUnavailableError rrule(dist_analysis, cfg, fake)
    @test_throws BackendUnavailableError rrule(
        dist_synthesis, cfg, scalar_coefficients; prototype_θφ=fake,
    )
    @test_throws BackendUnavailableError rrule(dist_analysis_sphtor, cfg, fake, fake)
    @test_throws BackendUnavailableError rrule(
        dist_synthesis_sphtor, cfg, S, T; prototype_θφ=fake,
    )
end
