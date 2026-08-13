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
        backend in (:cpu, :mpi_cpu) ? :verified : :unverified_hardware,
        _SHTNS37_TESTFILES[backend],
    )
    for feature in SHTNS37_CAPABILITIES for backend in SHTNS37_BACKENDS
]

"""
    shtns37_capabilities() -> Vector{SHTns37Capability}

Return a copy of the executable SHTns 3.7 mathematical capability matrix.
CPU and MPI/CPU cells are verified by the final local parity gate. Accelerator
cells remain `:unverified_hardware` until their named runner has passed on
physical hardware for that backend.
"""
shtns37_capabilities() = copy(_SHTNS37_CAPABILITIES)

const _SHTNS37_CI_INVENTORY = (
    (backend=:cpu, workflow=".github/workflows/ci.yml",
     job="shtns37-cpu-parity"),
    (backend=:cuda, workflow=".github/workflows/gpu-parity.yml",
     job="cuda-parity"),
    (backend=:amdgpu, workflow=".github/workflows/gpu-parity.yml",
     job="amdgpu-parity"),
    (backend=:mpi_cpu, workflow=".github/workflows/mpi-examples.yml",
     job="shtns37-mpi-cpu-parity"),
    (backend=:mpi_cuda, workflow=".github/workflows/gpu-parity.yml",
     job="mpi-cuda-parity"),
    (backend=:mpi_amdgpu, workflow=".github/workflows/gpu-parity.yml",
     job="mpi-amdgpu-parity"),
)

"Return the checked-in CI job associated with each parity backend."
_shtns37_ci_inventory() = _SHTNS37_CI_INVENTORY

const _SHTNS37_STATUS_LABELS = (
    required="required",
    verified="verified",
    unverified_hardware="unverified hardware",
)

@inline function _shtns37_status_label(status::Symbol)
    status === :required && return _SHTNS37_STATUS_LABELS.required
    status === :verified && return _SHTNS37_STATUS_LABELS.verified
    status === :unverified_hardware && return _SHTNS37_STATUS_LABELS.unverified_hardware
    throw(ArgumentError("unsupported SHTns 3.7 capability status: $status"))
end

const _SHTNS37_BACKEND_LABELS = (
    cpu="CPU",
    cuda="CUDA",
    amdgpu="AMDGPU",
    mpi_cpu="MPI/CPU",
    mpi_cuda="MPI/CUDA",
    mpi_amdgpu="MPI/AMDGPU",
)

@inline _shtns37_backend_label(backend::Symbol) =
    getproperty(_SHTNS37_BACKEND_LABELS, backend)

"""
    _shtns37_parity_markdown() -> String

Render the documentation tables directly from [`shtns37_capabilities`](@ref)
and the checked-in CI inventory. This is deliberately called from a Documenter
`@eval` block so status cells cannot drift into a second hand-maintained matrix.
"""
function _shtns37_parity_markdown()
    rows = shtns37_capabilities()
    io = IOBuffer()

    println(io, "## Executable capability matrix")
    println(io)
    println(io, "| Capability | ",
            join((_shtns37_backend_label(backend) for backend in SHTNS37_BACKENDS), " | "),
            " |")
    println(io, "|:--|", join((":--:" for _ in SHTNS37_BACKENDS), "|"), "|")
    for feature in SHTNS37_CAPABILITIES
        statuses = String[]
        for backend in SHTNS37_BACKENDS
            row = only(filter(row -> row.feature === feature && row.backend === backend, rows))
            push!(statuses, _shtns37_status_label(row.status))
        end
        println(io, "| `", feature, "` | ", join(statuses, " | "), " |")
    end

    println(io)
    println(io, "## CI inventory")
    println(io)
    println(io, "| Backend | Parity runner | Workflow job |")
    println(io, "|:--|:--|:--|")
    for entry in _shtns37_ci_inventory()
        testfile = _SHTNS37_TESTFILES[entry.backend]
        runner_url = "https://github.com/subhk/SHTnsKit.jl/blob/main/$testfile"
        workflow_url = "https://github.com/subhk/SHTnsKit.jl/actions/workflows/$(basename(entry.workflow))"
        println(io, "| ", _shtns37_backend_label(entry.backend),
                " | [`", testfile, "`](", runner_url, ")",
                " | [`", entry.job, "`](", workflow_url, ") |")
    end

    return String(take!(io))
end
