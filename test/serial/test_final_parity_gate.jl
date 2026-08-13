using Test
using TOML
using SHTnsKit

const _FINAL_GATE_FIXTURE = joinpath(
    @__DIR__, "..", "fixtures", "compatibility", "task16_gate.toml"
)

function _source_match_count(root::String, pattern::Regex)
    files = String[joinpath(root, "src", "device_utils.jl")]
    append!(files, sort(filter(
        path -> endswith(path, ".jl"),
        readdir(joinpath(root, "ext"); join=true),
    )))
    return sum(files) do file
        count(line -> occursin(pattern, line), eachline(file))
    end
end

@testset "Task 16 final parity gate evidence" begin
    @test isfile(_FINAL_GATE_FIXTURE)
    isfile(_FINAL_GATE_FIXTURE) || return

    fixture = TOML.parsefile(_FINAL_GATE_FIXTURE)
    root = normpath(joinpath(@__DIR__, "..", ".."))

    local_gate = fixture["local_gate"]
    @test local_gate["serial_pass"] >= 67_171
    @test local_gate["parallel_grid_pass"] >= 1_673
    @test local_gate["jet_pass"] >= 30
    @test local_gate["aqua_pass"] >= 3
    @test local_gate["pkg_exit_code"] == 0
    @test local_gate["mpi_cpu_ranks"] == 4
    @test local_gate["mpi_cpu_exit_code"] == 0

    cells = fixture["cell"]
    @test Set(Symbol(cell["backend"]) for cell in cells) == Set(SHTNS37_BACKENDS)
    @test length(cells) == length(SHTNS37_BACKENDS)
    expected_status = Dict(
        :cpu => :verified,
        :cuda => :unverified_hardware,
        :amdgpu => :unverified_hardware,
        :mpi_cpu => :verified,
        :mpi_cuda => :unverified_hardware,
        :mpi_amdgpu => :unverified_hardware,
    )
    rows = shtns37_capabilities()
    for cell in cells
        backend = Symbol(cell["backend"])
        @test Symbol(cell["status"]) == expected_status[backend]
        @test all(row -> row.status == expected_status[backend],
                  filter(row -> row.backend == backend, rows))
        if backend in (:cuda, :amdgpu, :mpi_cuda, :mpi_amdgpu)
            @test cell["physical_hardware"] == false
            @test cell["result"] == "unavailable_hardware"
        else
            @test cell["physical_hardware"] == true
            @test cell["result"] == "passed"
        end
    end

    audit = fixture["host_transfer_audit"]
    exact_pattern = r"Array\(|collect\(|allowscalar|fallback|CPU\("
    hidden_pattern = r"\b(parent|copyto!|copy!)\b"
    @test audit["exact_pattern_match_count"] ==
          _source_match_count(root, exact_pattern)
    @test audit["parent_copy_match_count"] ==
          _source_match_count(root, hidden_pattern)
    @test audit["allowscalar_match_count"] == 0

    cuda_source = read(joinpath(root, "ext", "SHTnsKitGPUExt.jl"), String)
    amd_source = read(joinpath(root, "ext", "SHTnsKitAMDGPUExt.jl"), String)
    @test !occursin("allowscalar", cuda_source)
    @test !occursin("allowscalar", amd_source)
    @test count("return Array(_cuda", cuda_source) ==
          audit["legacy_cuda_host_result_bridges"]
    @test occursin("Historical host-buffer compatibility stays explicit and isolated", cuda_source)
    @test occursin("_staged_gpu_call", read(joinpath(root, "ext", "ParallelGPU.jl"), String))
end
