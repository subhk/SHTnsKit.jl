using Documenter
using SHTnsKit

# Set up documentation generation
println("Building SHTnsKit.jl documentation...")

# Check for optional documentation dependencies
has_literate = false
has_plots = false

try
    using Literate
    has_literate = true
    println(" Literate.jl available for example generation")
catch
    println(" Literate.jl not available - skipping example generation")
end

try
    using Plots
    has_plots = true
    println(" Plots.jl available for documentation plots")
catch
    println(" Plots.jl not available - plots will be skipped in examples")
end

#####
##### Generate literated examples (if available)
#####

if has_literate
    const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
    const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")
    
    # Create output directory if it doesn't exist
    if !isdir(OUTPUT_DIR)
        mkpath(OUTPUT_DIR)
    end
    
    # List of example files to process
    example_files = []
    
    if isdir(EXAMPLES_DIR)
        for file in readdir(EXAMPLES_DIR)
            if endswith(file, ".jl") && !startswith(file, ".")
                push!(example_files, file)
            end
        end
        
        println("Found $(length(example_files)) example files to process")
        
        # Process each example file
        for example in example_files
            try
                example_filepath = joinpath(EXAMPLES_DIR, example)
                println("Processing example: $example")
                
                # Generate markdown with Documenter flavor
                Literate.markdown(example_filepath, OUTPUT_DIR;
                                flavor = Literate.DocumenterFlavor(),
                                documenter = true,
                                execute = false)  # Set to true if examples should be executed
                                
                println("OK Generated: $(replace(example, ".jl" => ".md"))")
            catch e
                println("Warning: Failed to process $example: $e")
            end
        end
    else
        println("Examples directory not found: $EXAMPLES_DIR")
    end
end

#####
##### Documentation configuration
#####

# HTML format configuration
format = Documenter.HTML(
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://subhk.github.io/SHTnsKit.jl/stable",
    assets = [
        "assets/custom.css",
    ],
    analytics = "",
    collapselevel = 2,
    sidebar_sitename = true,
    edit_link = "main",
    repolink = "https://github.com/subhk/SHTnsKit.jl",
    size_threshold = 200 * 1024^2,  # 200 MiB
    size_threshold_warn = 10 * 1024^2   # 10 MiB warning
)

# Documentation pages structure
pages = Any[
    "Home" => "index.md",
    "Getting Started" => Any[
        "Installation" => "installation.md",
        "Quick Start" => "quickstart.md"
    ],
    "User Guide" => Any[
        "GPU Acceleration" => "gpu.md",
        "Distributed Computing" => "distributed.md",
        "Performance Guide" => "performance.md",
        "Performance Tips" => "performance_tips.md",
        "Advanced Usage" => "advanced.md"
    ],
    "Reference" => Any[
        "API Reference" => "api/index.md",
        "Examples Gallery" => "examples/index.md"
    ]
]

# Add literated examples if they exist
if has_literate && isdir(joinpath(@__DIR__, "src/literated"))
    literated_files = []
    literated_dir = joinpath(@__DIR__, "src/literated")
    
    for file in readdir(literated_dir)
        if endswith(file, ".md") && file != "index.md"
            # Create nice titles from filenames
            title = replace(replace(file, ".md" => ""), "_" => " ")
            title = titlecase(title)
            push!(literated_files, title => "literated/$file")
        end
    end
    
    if !isempty(literated_files)
        # Insert literated examples into the User Guide section
        user_guide_idx = findfirst(p -> p[1] == "User Guide", pages)
        if user_guide_idx !== nothing
            # Insert after Examples Gallery
            examples_section = Any["Generated Examples" => literated_files]
            splice!(pages[user_guide_idx][2], 3:2, examples_section)
        end
    end
end

#####
##### Build documentation
#####

println("Generating documentation with Documenter.jl...")

makedocs(;
    modules = [SHTnsKit],
    authors = "SHTnsKit.jl contributors",
    repo = "https://github.com/subhk/SHTnsKit.jl/blob/{commit}{path}#{line}",
    sitename = "SHTnsKit.jl",
    format = format,
    pages = pages,
    clean = true,
    doctest = true,
    linkcheck = false,  # Set to true for link checking (slower)
    checkdocs = :exports,
    warnonly = [:cross_references, :missing_docs],
    draft = false
)

#####
##### Deploy documentation
#####

# Only deploy on CI
if get(ENV, "CI", "false") == "true"
    println("Deploying documentation...")
    
    deploydocs(;
        repo = "github.com/subhk/SHTnsKit.jl",
        devbranch = "main",
        target = "build",
        deps = nothing,
        make = nothing,
        # Deploy both stable and dev docs since releases exist
        versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
        forcepush = false,
        deploy_config = Documenter.GitHubActions(),
        push_preview = true
    )
else
    println("Skipping deployment (not running in CI)")
    println("Documentation built successfully!")
    println("Open docs/build/index.html to view locally")
end

#####
##### Cleanup
#####

println("Cleaning up temporary files...")

# Clean up any temporary files created during documentation build
temp_patterns = [r"\.jld2$", r"\.h5$", r"\.tmp$"]
temp_files = String[]

function find_temp_files(dir, patterns)
    files = String[]
    if isdir(dir)
        for (root, dirs, filenames) in walkdir(dir)
            for filename in filenames
                for pattern in patterns
                    if occursin(pattern, filename)
                        push!(files, joinpath(root, filename))
                    end
                end
            end
        end
    end
    return files
end

# Look for temporary files in docs directory
temp_files = find_temp_files(@__DIR__, temp_patterns)

for file in temp_files
    try
        rm(file)
        println("Removed temporary file: $file")
    catch e
        println("Failed to remove $file: $e")
    end
end

println("Documentation build complete!")
