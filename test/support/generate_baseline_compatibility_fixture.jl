using SHTnsKit
using TOML

include(joinpath(@__DIR__, "compatibility_inventory.jl"))
using .CompatibilityInventory

root = normpath(joinpath(@__DIR__, "..", ".."))
fixture_path = joinpath(root, "test", "fixtures", "compatibility", "e2ce9027.toml")
fixture = TOML.parsefile(fixture_path)
revision = fixture["baseline"]["commit"]
inventory = inventory_sources(revision_source_pairs(root, revision), fixture["exports"])
isempty(inventory.parse_errors) || error("baseline parse errors: $(inventory.parse_errors)")

fixture["baseline"]["method_count"] = length(inventory.methods)
fixture["baseline"]["tuple_arity_method_count"] =
    count(method -> !isempty(method.tuple_arities), inventory.methods)
fixture["baseline"]["tuple_component_method_count"] =
    count(method -> !isempty(method.tuple_component_signatures), inventory.methods)
fixture["baseline"]["source_digest_sha256"] = inventory.source_digest
fixture["baseline"]["method_extraction"] =
    "all src/**/*.jl at the immutable baseline tree; normalized declaration AST"
fixture["method"] = method_to_fixture.(inventory.methods)

proof = Dict(
    "method_family" => "static_structural_and_hasmethod",
    "type" => "explicit_core_family_probe",
    "macro" => "explicit_core_family_probe",
    "generated_or_extension" => "extension_source_and_mpi_runner_probe",
)
fixture["runtime_probe"] = [
    Dict("name" => entry.name,
         "classification" => entry.kind,
         "proof" => proof[entry.kind])
    for entry in inventory.probes
]

open(fixture_path, "w") do io
    TOML.print(io, fixture; sorted=true)
end
