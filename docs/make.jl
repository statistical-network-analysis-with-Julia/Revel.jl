using Documenter
using Revel

DocMeta.setdocmeta!(Revel, :DocTestSetup, :(using Revel); recursive=true)

makedocs(
    sitename = "Revel.jl",
    modules = [Revel],
    authors = "Statistical Network Analysis with Julia",
    format = Documenter.HTML(
        prettyurls = get(ENV, "DOCS_PRETTY_URLS", get(ENV, "CI", "false")) == "true",
        canonical = "https://statistical-network-analysis-with-Julia.github.io/Revel.jl/dev/",
        assets = ["assets/favicon.ico"],
        footer = "[Ecosystem home](/) · [Packages](/packages/) · [Get started](/getting-started/) · [Capabilities](/capabilities/) — Built with [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).",
        edit_link = "main",
    ),
    repo = Documenter.Remotes.GitHub("Statistical-network-analysis-with-Julia", "Revel.jl"),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Memory and Layers" => "guide/layers.md",
            "Endogenous Effects" => "guide/effects.md",
            "Covariates" => "guide/covariates.md",
            "Interactions" => "guide/interactions.md",
            "Fitting" => "guide/fitting.md",
            "Goodness of Fit" => "guide/gof.md",
            "Relational Hyperevents" => "guide/hyperevents.md",
            "Concordance with Other Packages" => "guide/concordance.md",
            "Literature" => "guide/literature.md",
        ],
        "API Reference" => [
            "Memory and Layers" => "api/layers.md",
            "Endogenous Effects" => "api/statistics.md",
            "Covariates and Interactions" => "api/covariates.md",
            "Fitting" => "api/fitting.md",
            "Diagnostics" => "api/diagnostics.md",
            "Hyperevents" => "api/hyperevents.md",
        ],
    ],
    # STRICT. Undefined bindings, bad cross-references, duplicate docs and
    # malformed markdown are build ERRORS, so they cannot silently accumulate.
    #
    # `checkdocs = :exports` is the one deliberate exclusion: every *exported*
    # name must be documented, but internal machinery need not be -- filler
    # docstrings for names a user never types are worse than none.
    warnonly = false,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/statistical-network-analysis-with-Julia/Revel.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev", # Development alias; change to "v^" when adopting release-based stable docs.
        "dev" => "dev",
    ],
    push_preview = false, # Pull requests build docs; main/tags publish through Pages.
)
