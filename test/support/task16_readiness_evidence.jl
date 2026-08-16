using SHA
using Test

function _task16_sha256(path::AbstractString)
    return bytes2hex(sha256(read(path)))
end

const _TASK16_AUDITED_SCOPE = [
    "Project.toml",
    ".github/workflows",
    "docs/src/shtns37-parity.md",
    "ext",
    "src",
    "test",
]

# The evidence record and the two files whose hashes it records cannot be part
# of their own content identity. All other files in the audited scope count.
const _TASK16_AUDITED_EXCLUSIONS = [
    "test/fixtures/compatibility/task16_gate.toml",
    "test/fixtures/compatibility/task16_local_commands.txt",
    "test/fixtures/compatibility/task16_local_summary.log",
]

function _task16_generated_audit_artifact(path::AbstractString)
    name = basename(path)
    return occursin(r"^Manifest(?:-v[^/]*)?\.toml$", name) ||
           occursin(r"\.jl(?:\.[^.]+)?\.cov$", name) ||
           endswith(name, ".jl.mem") ||
           name == "LocalPreferences.toml" ||
           name == "JuliaLocalPreferences.toml"
end

function _task16_audited_paths(root::AbstractString, scope, exclusions)
    excluded = Set(replace.(String.(exclusions), '\\' => '/'))
    candidates = String[]
    for relative in String.(scope)
        path = joinpath(root, relative)
        if isfile(path)
            push!(candidates, replace(relpath(path, root), '\\' => '/'))
        elseif isdir(path)
            for (directory, _, names) in walkdir(path), name in names
                file = joinpath(directory, name)
                isfile(file) && push!(
                    candidates, replace(relpath(file, root), '\\' => '/'),
                )
            end
        else
            throw(ArgumentError("missing Task 16 audited path: $relative"))
        end
    end
    unique!(sort!(candidates))
    all(exclusion -> exclusion in candidates, excluded) ||
        throw(ArgumentError("Task 16 evidence exclusion is outside its audited scope"))
    return filter(
        path -> path ∉ excluded && !_task16_generated_audit_artifact(path),
        candidates,
    )
end

"""
    task16_audited_tree_digest(root, scope, exclusions)

Compute a Git-independent content identity. The outer SHA-256 covers a sorted
manifest whose records are `relative-path NUL SHA256(file) newline`.
"""
function task16_audited_tree_digest(root::AbstractString, scope, exclusions)
    records = [path * '\0' * _task16_sha256(joinpath(root, path))
               for path in _task16_audited_paths(root, scope, exclusions)]
    return bytes2hex(sha256(codeunits(join(records, '\n') * '\n')))
end

function test_task16_readiness_evidence(root::AbstractString, fixture)
    @testset "Task 16 local readiness is non-certifying" begin
        local_gate = fixture["local_gate"]
        @test local_gate["certifying"] == false
        @test local_gate["evidence_kind"] == "local_readiness"
        @test !haskey(local_gate, "tested_commit")
        @test !haskey(local_gate, "tested_tree")
        @test haskey(local_gate, "audited_tree_digest")
        @test haskey(local_gate, "audited_scope")
        @test haskey(local_gate, "audited_exclusions")
        if all(key -> haskey(local_gate, key),
               ("audited_tree_digest", "audited_scope", "audited_exclusions"))
            @test local_gate["audited_scope"] == _TASK16_AUDITED_SCOPE
            @test local_gate["audited_exclusions"] == _TASK16_AUDITED_EXCLUSIONS
            @test occursin(r"^[0-9a-f]{64}$", local_gate["audited_tree_digest"])
            @test task16_audited_tree_digest(
                root, local_gate["audited_scope"], local_gate["audited_exclusions"],
            ) == local_gate["audited_tree_digest"]
        end

        commands_path = joinpath(root, local_gate["commands_file"])
        summary_path = joinpath(root, local_gate["summary_log"])
        @test isfile(commands_path)
        @test isfile(summary_path)
        @test _task16_sha256(commands_path) == local_gate["commands_sha256"]
        @test _task16_sha256(summary_path) == local_gate["summary_log_sha256"]

        commands = read(commands_path, String)
        @test occursin("JULIA_DEPOT_PATH=/private/tmp/julia_depot_shtnskit:/Users/subha/.julia", commands)
        @test occursin("SHTNSKIT_RUN_JET_TESTS=1", commands)
        @test occursin("SHTNSKIT_RUN_AQUA_TESTS=1", commands)
        @test occursin("--startup-file=no --project=. -e 'using Pkg; Pkg.test()'", commands)
        @test occursin("JULIA_BINDIR=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin", commands)
        @test occursin("/Users/subha/.julia/packages/MPI/pvbg6/bin/mpiexecjl -n 4", commands)
        @test occursin("--project=test/gpu/cuda test/parity/runtests_mpi.jl", commands)

        summary = read(summary_path, String)
        for (key, value) in (
            "audited_tree_digest" => local_gate["audited_tree_digest"],
            "serial_pass" => local_gate["serial_pass"],
            "parallel_grid_pass" => local_gate["parallel_grid_pass"],
            "jet_pass" => local_gate["jet_pass"],
            "aqua_pass" => local_gate["aqua_pass"],
            "pkg_exit_code" => local_gate["pkg_exit_code"],
            "mpi_cpu_ranks" => local_gate["mpi_cpu_ranks"],
            "mpi_cpu_exit_code" => local_gate["mpi_cpu_exit_code"],
        )
            @test occursin("$key=$value", summary)
        end
        @test occursin("certifying=false", summary)
        @test occursin("evidence_kind=local_readiness", summary)

        expected_status = Dict(
            :cpu => :required,
            :cuda => :unverified_hardware,
            :amdgpu => :unverified_hardware,
            :mpi_cpu => :required,
            :mpi_cuda => :unverified_hardware,
            :mpi_amdgpu => :unverified_hardware,
        )
        cells = fixture["cell"]
        @test Set(Symbol(cell["backend"]) for cell in cells) == Set(keys(expected_status))
        rows = SHTnsKit.shtns37_capabilities()
        for cell in cells
            backend = Symbol(cell["backend"])
            status = Symbol(cell["status"])
            @test status == expected_status[backend]
            @test all(row -> row.status == status,
                      filter(row -> row.backend == backend, rows))

            artifact = get(cell, "ci_artifact", "")
            artifact_sha = get(cell, "ci_artifact_sha256", "")
            if status == :verified
                @test startswith(artifact, "https://github.com/")
                @test occursin(r"^[0-9a-f]{64}$", artifact_sha)
                @test get(cell, "ci_run_id", 0) > 0
            else
                @test isempty(artifact)
                @test isempty(artifact_sha)
                @test get(cell, "ci_run_id", 0) == 0
            end
        end
        @test all(row -> row.status != :verified, rows)
    end
end
