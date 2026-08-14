using Test
using TOML
using SHTnsKit

include(joinpath(@__DIR__, "..", "support", "task16_readiness_evidence.jl"))
include(joinpath(@__DIR__, "..", "support", "host_transfer_inventory.jl"))
using .HostTransferInventory

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

    test_task16_readiness_evidence(root, fixture)

    local_gate = fixture["local_gate"]
    @test local_gate["serial_pass"] >= 67_171
    @test local_gate["parallel_grid_pass"] >= 1_673
    @test local_gate["jet_pass"] >= 30
    @test local_gate["aqua_pass"] >= 3
    @test local_gate["pkg_exit_code"] == 0
    @test local_gate["mpi_cpu_ranks"] == 4
    @test local_gate["mpi_cpu_exit_code"] == 0
    @test get(local_gate, "certifying", true) == false
    @test haskey(local_gate, "audited_tree_digest")
    @test haskey(local_gate, "commands_sha256")
    @test haskey(local_gate, "summary_log_sha256")

    cells = fixture["cell"]
    @test Set(Symbol(cell["backend"]) for cell in cells) == Set(SHTNS37_BACKENDS)
    @test length(cells) == length(SHTNS37_BACKENDS)
    expected_status = Dict(
        :cpu => :required,
        :cuda => :unverified_hardware,
        :amdgpu => :unverified_hardware,
        :mpi_cpu => :required,
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

    audit_path = joinpath(root, "test", "fixtures", "compatibility",
                          "host_transfer_allowlist.toml")
    @test isfile(audit_path)
    transfer_fixture = TOML.parsefile(audit_path)
    scanned = scan_host_transfer_occurrences(root)
    allowed = transfer_fixture["entry"]
    @test transfer_fixture["audit"]["entry_count"] == length(scanned) == 662
    scanned_keys = Set(transfer_occurrence_key.(scanned))
    allowed_keys = Set(entry["key"] for entry in allowed)
    @test length(allowed_keys) == length(allowed)
    @test isempty(setdiff(scanned_keys, allowed_keys))
    @test isempty(setdiff(allowed_keys, scanned_keys))
    @test all(entry -> entry["path"] in (occurrence.path for occurrence in scanned), allowed)
    @test all(entry -> occursin(r"^[0-9a-f]{64}$", entry["snippet_sha256"]), allowed)
    @test all(entry -> !isempty(entry["classification"]), allowed)
    @test all(entry -> !isempty(entry["reason"]), allowed)
    @test count(entry -> entry["token"] == "allowscalar", allowed) == 0
    similar_array_entries = filter(entry -> entry["token"] == "similar_array", allowed)
    @test length(similar_array_entries) == 2
    @test all(entry -> entry["path"] == "ext/ParallelGPU.jl", similar_array_entries)
    @test all(entry -> entry["classification"] == "bounded_pinned_mpi_staging",
              similar_array_entries)
    @test Set(entry["classification"] for entry in allowed) == Set((
        "bounded_pinned_mpi_staging", "cpu_only", "legacy_host_result",
        "metadata_or_storage_preserving", "small_setup_table",
        "explicit_cpu_or_fallback", "unreachable_early_error_callback",
    ))
    @test all(required -> any(entry -> entry["classification"] == required, allowed), (
        "metadata_or_storage_preserving", "small_setup_table", "cpu_only",
        "bounded_pinned_mpi_staging", "legacy_host_result",
        "unreachable_early_error_callback",
    ))
    for occurrence in scanned
        entry = only(filter(entry -> entry["key"] == transfer_occurrence_key(occurrence), allowed))
        @test entry["path"] == occurrence.path
        @test entry["token"] == occurrence.token
        @test entry["snippet_sha256"] == occurrence.snippet_sha256
        @test entry["same_snippet_ordinal"] == occurrence.same_snippet_ordinal
        expected_review = classify_transfer_occurrence(occurrence)
        @test expected_review !== nothing
        @test entry["classification"] == expected_review.classification
        @test entry["reason"] == expected_review.reason
    end

    audit = fixture["host_transfer_audit"]
    @test audit["allowscalar_match_count"] == 0

    cuda_source = read(joinpath(root, "ext", "SHTnsKitGPUExt.jl"), String)
    amd_source = read(joinpath(root, "ext", "SHTnsKitAMDGPUExt.jl"), String)
    @test !occursin("allowscalar", cuda_source)
    @test !occursin("allowscalar", amd_source)
    @test count("return Array(_cuda", cuda_source) ==
          audit["legacy_cuda_host_result_bridges"]
    @test occursin("Historical host-buffer compatibility stays explicit and isolated", cuda_source)
    @test occursin("_staged_gpu_call", read(joinpath(root, "ext", "ParallelGPU.jl"), String))

    parallel_ad_source = read(joinpath(root, "ext", "SHTnsKitParallelADExt.jl"), String)
    @test occursin("on_device(parent(value))", parallel_ad_source)
    @test !occursin("HostPencilArray", parallel_ad_source)
    @test !occursin("Matrix{ComplexF64}(A)", parallel_ad_source)
    @test occursin("_require_host_pencil", parallel_ad_source)
    @test occursin("BackendUnavailableError", parallel_ad_source)
    scalar_synthesis_guard = findfirst(
        "_require_host_pencil(:dist_synthesis_pullback, prototype_θφ)",
        parallel_ad_source,
    )
    scalar_synthesis_forward = findfirst("y = SHTnsKit.dist_synthesis(", parallel_ad_source)
    vector_synthesis_guard = findfirst(
        "_require_host_pencil(:dist_synthesis_sphtor_pullback, prototype_θφ)",
        parallel_ad_source,
    )
    vector_synthesis_forward = findfirst(
        "y = SHTnsKit.dist_synthesis_sphtor(", parallel_ad_source,
    )
    @test first(scalar_synthesis_guard) < first(scalar_synthesis_forward)
    @test first(vector_synthesis_guard) < first(vector_synthesis_forward)

    parallel_runner = read(joinpath(root, "test", "parallel", "runtests.jl"), String)
    @test occursin("include(\"test_parallel_ad_storage.jl\")", parallel_runner)
end
