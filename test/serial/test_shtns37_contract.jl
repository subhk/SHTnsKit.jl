using Test
using TOML
using SHTnsKit

include(joinpath(@__DIR__, "..", "parity", "capabilities.jl"))
using .SHTns37TestCapabilities

@testset "SHTns 3.7 capability contract" begin
    required_family_members = (
        :synthesis_sph, :synthesis_sph_cplx,
        :synthesis_sph_l, :synthesis_sph_l_cplx, :synthesis_sph_ml,
        :synthesis_tor, :synthesis_tor_cplx,
        :synthesis_tor_l, :synthesis_tor_l_cplx, :synthesis_tor_ml,
        :synthesis_sphtor_l_cplx, :synthesis_qst_l_cplx,
        :synthesis_sphtor_batch_cplx, :synthesis_qst_batch_cplx,
    )
    @test all(in(SHTns37TestCapabilities.ENTRYPOINTS), required_family_members)
    @test all(group -> !isempty(group), SHTns37TestCapabilities.ENTRYPOINT_GROUPS)
    @test length(unique(SHTns37TestCapabilities.ENTRYPOINTS)) ==
        length(SHTns37TestCapabilities.ENTRYPOINTS)

    contract_names = (
        :SHTns37Capability, :SHTNS37_BACKENDS,
        :SHTNS37_CAPABILITIES, :shtns37_capabilities,
    )
    for name in contract_names
        @test isdefined(SHTnsKit, name)
        @test Base.isexported(SHTnsKit, name)
    end

    if all(name -> isdefined(SHTnsKit, name), contract_names)
        @test SHTnsKit.SHTNS37_BACKENDS == SHTns37TestCapabilities.BACKENDS
        @test SHTnsKit.SHTNS37_CAPABILITIES == SHTns37TestCapabilities.CAPABILITIES

        rows = SHTnsKit.shtns37_capabilities()
        expected_cells = Set(Iterators.product(
            SHTns37TestCapabilities.CAPABILITIES,
            SHTns37TestCapabilities.BACKENDS,
        ))
        actual_cells = Set((row.feature, row.backend) for row in rows)

        @test length(rows) == length(expected_cells)
        @test length(actual_cells) == length(rows)
        @test actual_cells == expected_cells

        for row in rows
            @test row isa SHTnsKit.SHTns37Capability
            @test row.testfile == SHTns37TestCapabilities.TESTFILES[row.backend]
            expected_status = row.backend in (:cpu, :mpi_cpu) ?
                :required : :unverified_hardware
            @test row.status == expected_status
        end

        empty!(rows)
        @test length(SHTnsKit.shtns37_capabilities()) == length(expected_cells)
    end

    for name in SHTns37TestCapabilities.ENTRYPOINTS
        @test isdefined(SHTnsKit, name)
        @test Base.isexported(SHTnsKit, name)
    end
end

@testset "SHTns 3.7 fixture, documentation, and CI inventory" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    manifest = TOML.parsefile(joinpath(root, "test", "fixtures", "shtns37", "manifest.toml"))
    fixtures = manifest["fixture"]

    fixture_capabilities = Set(Symbol(fixture["capability"]) for fixture in fixtures)
    @test fixture_capabilities == Set(SHTnsKit.SHTNS37_CAPABILITIES)

    rows = SHTnsKit.shtns37_capabilities()
    for fixture in fixtures, backend in SHTnsKit.SHTNS37_BACKENDS
        feature = Symbol(fixture["capability"])
        matching = filter(row -> row.feature == feature && row.backend == backend, rows)
        @test length(matching) == 1
        length(matching) == 1 || continue
        @test isfile(joinpath(root, only(matching).testfile))
    end

    @test isdefined(SHTnsKit, :_shtns37_ci_inventory)
    @test isdefined(SHTnsKit, :_shtns37_parity_markdown)

    docs_path = joinpath(root, "docs", "src", "shtns37-parity.md")
    workflow_path = joinpath(root, ".github", "workflows", "gpu-parity.yml")
    @test isfile(docs_path)
    @test isfile(workflow_path)

    if isdefined(SHTnsKit, :_shtns37_ci_inventory)
        inventory = SHTnsKit._shtns37_ci_inventory()
        @test Set(entry.backend for entry in inventory) == Set(SHTnsKit.SHTNS37_BACKENDS)
        @test length(unique((entry.workflow, entry.job) for entry in inventory)) ==
              length(inventory)

        for entry in inventory
            workflow = joinpath(root, entry.workflow)
            @test isfile(workflow)
            isfile(workflow) || continue
            source = read(workflow, String)
            @test occursin(Regex("(?m)^  $(entry.job):\\s*" * raw"$"), source)
        end

        hardware = Set((:cuda, :amdgpu, :mpi_cuda, :mpi_amdgpu))
        for row in rows
            row.backend in hardware && row.status == :verified || continue
            entry = only(filter(item -> item.backend == row.backend, inventory))
            source = read(joinpath(root, entry.workflow), String)
            @test occursin(Regex("(?m)^  $(entry.job):\\s*" * raw"$"), source)
        end
    end

    if isdefined(SHTnsKit, :_shtns37_parity_markdown)
        generated = SHTnsKit._shtns37_parity_markdown()
        labels = Dict(
            :verified => "verified",
            :required => "required",
            :unverified_hardware => "unverified hardware",
        )
        @test all(row -> haskey(labels, row.status), rows)
        for feature in SHTnsKit.SHTNS37_CAPABILITIES
            cells = String[]
            for backend in SHTnsKit.SHTNS37_BACKENDS
                row = only(filter(row -> row.feature == feature && row.backend == backend, rows))
                push!(cells, labels[row.status])
            end
            @test occursin("| `$(feature)` | $(join(cells, " | ")) |", generated)
        end

        if isfile(docs_path)
            docs_source = read(docs_path, String)
            @test occursin("SHTnsKit._shtns37_parity_markdown()", docs_source)
            @test !occursin(r"(?m)^\| `scalar_real_full` \|", docs_source)
        end
    end
end

@testset "SHTns 3.7 hardware workflow is strict" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    workflow_path = joinpath(root, ".github", "workflows", "gpu-parity.yml")
    isfile(workflow_path) || return
    source = read(workflow_path, String)

    expected_jobs = (
        "cuda-parity" => "runs-on: [self-hosted, linux, x64, cuda]",
        "amdgpu-parity" => "runs-on: [self-hosted, linux, x64, amdgpu]",
        "mpi-cuda-parity" => "runs-on: [self-hosted, linux, x64, cuda]",
        "mpi-amdgpu-parity" => "runs-on: [self-hosted, linux, x64, amdgpu]",
    )
    for (job, runner) in expected_jobs
        @test occursin(Regex("(?m)^  $job:\\s*" * raw"$"), source)
        @test occursin(runner, source)
    end

    @test occursin("CUDA.functional() || error", source)
    @test occursin("AMDGPU.functional() || error", source)
    @test occursin("length(devices) >= 2 || error", source)
    @test occursin("MPI.Comm_size(MPI.COMM_WORLD) == 2 || error", source)
    @test occursin("test/gpu/cuda/runtests.jl", source)
    @test occursin("test/gpu/amdgpu/runtests.jl", source)
    @test occursin("test/gpu/cuda/mpi_runtests.jl", source)
    @test occursin("test/gpu/amdgpu/mpi_runtests.jl", source)
    @test count("actions/upload-artifact@v4", source) == 4
end
