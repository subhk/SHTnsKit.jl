using SHA
using Test

function _task16_sha256(path::AbstractString)
    return bytes2hex(sha256(read(path)))
end

function test_task16_readiness_evidence(root::AbstractString, fixture)
    @testset "Task 16 local readiness is non-certifying" begin
        local_gate = fixture["local_gate"]
        @test local_gate["certifying"] == false
        @test local_gate["evidence_kind"] == "local_readiness"
        @test local_gate["tested_commit"] == fixture["gate"]["baseline_head"]
        @test occursin(r"^[0-9a-f]{40}$", local_gate["tested_commit"])
        @test occursin(r"^[0-9a-f]{40}$", local_gate["tested_tree"])
        treeish = local_gate["tested_commit"] * "^{tree}"
        actual_tree = strip(read(`git -C $root rev-parse $treeish`, String))
        @test actual_tree == local_gate["tested_tree"]

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
            "tested_commit" => local_gate["tested_commit"],
            "tested_tree" => local_gate["tested_tree"],
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
