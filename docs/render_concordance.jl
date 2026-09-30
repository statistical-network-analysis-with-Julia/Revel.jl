# Rewrite the tables of docs/src/guide/concordance.md from `effect_catalogue()`:
#
#     julia --project=docs docs/render_concordance.jl
#
# Everything between the two catalogue markers is replaced. A testset keeps the
# page and the catalogue in step, and this script is how to repair it.
using Revel

const PAGE = joinpath(@__DIR__, "src", "guide", "concordance.md")
const BEGIN_MARK = "<!-- catalogue:begin -->"
const END_MARK = "<!-- catalogue:end -->"
const HEADINGS = ["endogenous" => "Endogenous effects", "exogenous" => "Exogenous effects",
                  "interaction" => "Interactions", "hyperevent" => "Hyperevents",
                  "memory" => "Memory and event weights",
                  "scaling" => "Scaling"]

cell(s) = replace(s, "|" => "\\|")
code(s) = isempty(s) ? "" : "`" * cell(s) * "`"

function tables()
    cat = effect_catalogue()
    io = IOBuffer()
    for (family, heading) in HEADINGS
        println(io, "\n### ", heading, "\n")
        println(io, "| Revel | Measures | relevent | remstats | rem | goldfish | eventnet | Source |")
        println(io, "|---|---|---|---|---|---|---|---|")
        for r in eachrow(cat[cat.family .== family, :])
            println(io, "| ", code(r.revel), " | ", cell(r.configuration), " | ",
                    code(r.relevent), " | ", code(r.remstats), " | ", code(r.rem), " | ",
                    code(r.goldfish), " | ", cell(r.eventnet), " | ", cell(r.source), " |")
        end
    end
    return String(take!(io))
end

page = read(PAGE, String)
start = findfirst(BEGIN_MARK, page)
stop = findfirst(END_MARK, page)
(start === nothing || stop === nothing) && error("catalogue markers not found in $PAGE")
write(PAGE, page[1:last(start)] * "\n" * tables() * "\n" * page[first(stop):end])
println("rewrote the catalogue tables of ", PAGE)
