"""
    SHTns37Capability

One required SHTns 3.7 mathematical capability on one execution backend.
`status` is one of `:required`, `:verified`, or `:unverified_hardware`.
"""
struct SHTns37Capability
    feature::Symbol
    backend::Symbol
    status::Symbol
    testfile::String
end

const SHTNS37_BACKENDS = (
    :cpu, :cuda, :amdgpu, :mpi_cpu, :mpi_cuda, :mpi_amdgpu,
)

const SHTNS37_CAPABILITIES = (
    :scalar_real_full, :scalar_complex_full, :scalar_l, :scalar_ml,
    :scalar_batch, :packed_storage, :sphtor_full, :sphtor_l,
    :sphtor_ml, :sphtor_batch, :qst_full, :qst_l, :qst_ml, :qst_batch,
    :point, :point_complex, :latitude, :latitude_complex,
    :qst_point, :qst_latitude, :gradient_point, :operators, :rotations,
)

const _SHTNS37_TESTFILES = Dict(
    :cpu => "test/parity/runtests_cpu.jl",
    :cuda => "test/gpu/cuda/runtests.jl",
    :amdgpu => "test/gpu/amdgpu/runtests.jl",
    :mpi_cpu => "test/parity/runtests_mpi.jl",
    :mpi_cuda => "test/gpu/cuda/mpi_runtests.jl",
    :mpi_amdgpu => "test/gpu/amdgpu/mpi_runtests.jl",
)

const _SHTNS37_CAPABILITIES = SHTns37Capability[
    SHTns37Capability(
        feature,
        backend,
        backend in (:cpu, :mpi_cpu) ? :required : :unverified_hardware,
        _SHTNS37_TESTFILES[backend],
    )
    for feature in SHTNS37_CAPABILITIES for backend in SHTNS37_BACKENDS
]

"""
    shtns37_capabilities() -> Vector{SHTns37Capability}

Return a copy of the executable SHTns 3.7 mathematical capability matrix.
Required cells are obligations, not claims of verification; hardware-backed cells
remain `:unverified_hardware` until their named runner has passed on that backend.
"""
shtns37_capabilities() = copy(_SHTNS37_CAPABILITIES)
