using TOML

include(joinpath(@__DIR__, "host_transfer_inventory.jl"))
using .HostTransferInventory

root = normpath(joinpath(@__DIR__, "..", ".."))
output = joinpath(root, "test", "fixtures", "compatibility", "host_transfer_allowlist.toml")
occurrences = scan_host_transfer_occurrences(root)
entries = Dict{String,Any}[]
for occurrence in occurrences
    review = classify_transfer_occurrence(occurrence)
    review === nothing && error("unclassified host-transfer occurrence: $occurrence")
    push!(entries, Dict(
        "key" => transfer_occurrence_key(occurrence),
        "path" => occurrence.path,
        "token" => occurrence.token,
        "snippet_sha256" => occurrence.snippet_sha256,
        "same_snippet_ordinal" => occurrence.same_snippet_ordinal,
        "classification" => review.classification,
        "reason" => review.reason,
    ))
end

fixture = Dict(
    "audit" => Dict(
        "scope" => "src/device_utils.jl and every ext/*.jl file",
        "scanner" => "deterministic non-overlapping spelling scanner; longer transfer spellings win",
        "entry_count" => length(entries),
    ),
    "entry" => entries,
)
open(output, "w") do io
    TOML.print(io, fixture; sorted=true)
end
