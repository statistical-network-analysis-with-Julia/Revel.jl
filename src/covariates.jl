# =============================================================================
# Exogenous effects
# =============================================================================
#
# The forms are inherited from Cox regression and ERGM/SAOM practice; what the
# REM literature added is an identification rule — a term that is constant across
# the risk set of an event cannot be estimated from an ordinal likelihood. That
# is why `GlobalEffect` enters a model only inside an interaction, and why a
# sender covariate drops out of a receiver-choice model.
#
# Relevent.jl already ports relevent's four covariate effects (`CovSnd`,
# `CovRec`, `CovInt`, `CovEvent`) and REM.jl eventnet's attribute statistics on
# `NodeAttribute`. The effects here add what neither has: the remstats/goldfish
# dyadic forms, covariates that change over time, and both compute interfaces.

"""
    Covariate(values; name="x", categorical=false)
    Covariate(times, values; name="x", categorical=false)

An actor-level covariate, indexed by actor ID.

The one-argument form is constant over time. The two-argument form is
**time-varying**, piecewise constant: `values[:, k]` is in force from `times[k]`
until `times[k+1]` (and `values[:, 1]` before `times[1]`), the structure of
remstats' and goldfish's changing attributes.

Non-numeric `values` (strings, symbols) are treated as categories: they work
with [`MatchEffect`](@ref) and the matching filters, and the numeric effects
refuse them. `categorical=true` treats numbers as category labels too (team
numbers, say). An actor without a value is an `ArgumentError`, never a silent
zero, and so is a `missing` value: an unobserved attribute is not a category.

# Example
```julia
using Revel
role = Covariate([:staff, :lead, :staff]; name="role")
load = Covariate([0.0, 10.0], [1.0 3.0; 2.0 2.0; 0.5 0.5]; name="load")
covariate_value(load, 1, 5.0)     # 1.0
covariate_value(load, 1, 12.0)    # 3.0
covariate_value(role, 2, 0.0)     # 2.0 — the code of :lead
Covariate([1, 1, 2]; name="team", categorical=true).categorical   # true
```
"""
struct Covariate
    times::Vector{Float64}
    values::Matrix{Float64}
    categorical::Bool
    levels::Vector{Any}
    label::String
end

function _encode(values::AbstractArray; categorical::Bool=false)
    # An unobserved attribute is not a category of its own (the ecosystem's
    # missing-data contract): two missing values would otherwise "match"
    if any(ismissing, values)
        where_ = findall(ismissing, values)
        throw(ArgumentError(
            "a Covariate cannot hold `missing` (at $(length(where_)) position" *
            "$(length(where_) == 1 ? "" : "s"), first $(first(where_))): an unobserved " *
            "value is not a category, and treating it as one makes two missing " *
            "values match. Impute the values, or leave the actors out of the actor " *
            "universe."))
    end
    # numbers are numeric whatever the container's element type says
    !categorical && all(v -> v isa Real, values) &&
        return Matrix{Float64}(reshape(values, size(values, 1), :)), false, Any[]
    levels = Any[]
    codes = Matrix{Float64}(undef, size(values, 1), size(values, 2))
    for (i, v) in pairs(IndexCartesian(), reshape(values, size(values, 1), :))
        k = findfirst(isequal(v), levels)
        k === nothing && (push!(levels, v); k = length(levels))
        codes[i] = k
    end
    return codes, true, levels
end

function Covariate(values::AbstractVector; name::AbstractString="x",
                   categorical::Bool=false)
    isempty(values) && throw(ArgumentError("a Covariate needs at least one actor"))
    codes, categorical, levels = _encode(values; categorical=categorical)
    all(isfinite, codes) || throw(ArgumentError("Covariate $name: values must be finite"))
    return Covariate([-Inf], codes, categorical, levels, String(name))
end

function Covariate(times::AbstractVector{<:Real}, values::AbstractMatrix;
                   name::AbstractString="x", categorical::Bool=false)
    size(values, 2) == length(times) || throw(ArgumentError(
        "Covariate $name: $(length(times)) change times for $(size(values, 2)) " *
        "columns of values (one column per time is required)"))
    isempty(times) && throw(ArgumentError("Covariate $name: needs at least one time"))
    issorted(times; lt=<=) || throw(ArgumentError(
        "Covariate $name: change times must be strictly increasing"))
    codes, categorical, levels = _encode(values; categorical=categorical)
    all(isfinite, codes) || throw(ArgumentError("Covariate $name: values must be finite"))
    return Covariate(collect(Float64, times), codes, categorical, levels, String(name))
end

_covariate(x::Covariate; kwargs...) = x
_covariate(x::AbstractVector; name::AbstractString="x") = Covariate(x; name=name)

_is_static(c::Covariate) = size(c.values, 2) == 1
_n_actors(c::Covariate) = size(c.values, 1)

Base.show(io::IO, c::Covariate) =
    print(io, "Covariate(:", c.label, ", ", _n_actors(c), " actors, ",
          c.categorical ? "categorical" : "numeric", ", ",
          _is_static(c) ? "static" : "$(size(c.values, 2)) time points", ")")

"""
    covariate_value(x::Covariate, actor, time) -> Float64

The value of `x` for `actor` at `time` (the category code for a categorical
covariate). Throws an `ArgumentError` for an actor the covariate does not cover.

# Example
```julia
using Revel
x = Covariate([0.0, 5.0], [1.0 2.0; 3.0 4.0])
covariate_value(x, 2, 4.9), covariate_value(x, 2, 5.0)    # (3.0, 4.0)
```
"""
@inline function covariate_value(c::Covariate, actor::Int, t::Real)
    1 <= actor <= size(c.values, 1) || throw(ArgumentError(
        "actor $actor has no value of covariate :$(c.label) " *
        "(it covers actors 1:$(size(c.values, 1)))"))
    size(c.values, 2) == 1 && return @inbounds c.values[actor, 1]
    k = max(1, searchsortedlast(c.times, t))
    return @inbounds c.values[actor, k]
end

function _require_numeric(c::Covariate, what::AbstractString)
    c.categorical && throw(ArgumentError(
        "$what needs a numeric covariate, but :$(c.label) is categorical " *
        "(levels $(c.levels)). Use MatchEffect for a category match, or code the " *
        "categories as numbers yourself."))
    return c
end

const _COVARIATE_FORMS = (:send, :receive, :same, :difference, :absdiff, :similarity,
                          :average, :minimum, :maximum, :sum, :product)

"""
    CovariateEffect(form, x, y=x; transform=identity, name=nothing)

An exogenous effect built from actor covariates. `form` selects how the sender's
value `xᵢ` and the receiver's value `xⱼ` (or `yⱼ`) enter:

| `form` | value | named constructor |
|---|---|---|
| `:send` | `xᵢ` | [`SendEffect`](@ref) |
| `:receive` | `xⱼ` | [`ReceiveEffect`](@ref) |
| `:same` | `1` if `xᵢ == xⱼ` | [`MatchEffect`](@ref) |
| `:absdiff` | `abs(xᵢ - xⱼ)` | [`DiffEffect`](@ref) |
| `:difference` | `xᵢ - xⱼ` | `DiffEffect(x; absolute=false)` |
| `:similarity` | `-abs(xᵢ - xⱼ)` | [`SimEffect`](@ref) |
| `:average` | `(xᵢ + xⱼ)/2` | [`AverageEffect`](@ref) |
| `:minimum` / `:maximum` | `min` / `max` | [`MinimumEffect`](@ref), [`MaximumEffect`](@ref) |
| `:sum` | `xᵢ + xⱼ` | [`SumEffect`](@ref) |
| `:product` | `xᵢ · yⱼ` | [`ProductEffect`](@ref) |

Every form accepts a time-varying [`Covariate`](@ref).

# Example
```julia
using Revel
age = [30.0, 45.0, 52.0]
h = build_history(Event{Float64}[])
compute(CovariateEffect(:absdiff, age), h, 1, 3, 0.0)    # 22.0
compute(CovariateEffect(:average, age), h, 1, 3, 0.0)    # 41.0
```
"""
struct CovariateEffect{F} <: AbstractRevelStatistic
    form::Symbol
    x::Covariate
    y::Covariate
    transform::F
    label::String
end

function CovariateEffect(form::Symbol, x, y=x; transform=identity, name=nothing)
    form in _COVARIATE_FORMS || throw(ArgumentError(
        "form must be one of $(_COVARIATE_FORMS), got :$form"))
    cx = _covariate(x)
    cy = y === x ? cx : _covariate(y)
    form === :same || (_require_numeric(cx, "CovariateEffect(:$form)");
                       _require_numeric(cy, "CovariateEffect(:$form)"))
    form === :product || cy === cx || throw(ArgumentError(
        "only the :product form takes a second covariate"))
    f = _transform_fn(transform)
    base = form === :product && cy !== cx ? "product.$(cx.label).$(cy.label)" :
           "$(form).$(cx.label)"
    return CovariateEffect{typeof(f)}(form, cx, cy, f, _label(name, _auto_name(base, "", f)))
end

function _value(stat::CovariateEffect, events, s::Int, r::Int, t::Float64)
    form = stat.form
    v = if form === :send
        covariate_value(stat.x, s, t)
    elseif form === :receive
        covariate_value(stat.x, r, t)
    else
        a = covariate_value(stat.x, s, t)
        b = covariate_value(stat.y, r, t)
        form === :same ? Float64(a == b) :
        form === :absdiff ? abs(a - b) :
        form === :difference ? a - b :
        form === :similarity ? -abs(a - b) :
        form === :average ? (a + b) / 2 :
        form === :minimum ? min(a, b) :
        form === :maximum ? max(a, b) :
        form === :sum ? a + b : a * b
    end
    return Float64(stat.transform(v))
end

_uses_history(::CovariateEffect) = false
_interval_constant(stat::CovariateEffect) = _is_static(stat.x) && _is_static(stat.y)

"""
    SendEffect(x; transform=identity, name=nothing)

The sender's covariate value: do actors with more of `x` send more? (remstats
`send()`; goldfish `ego()`; relevent `CovSnd`.) In an ordinal model it is
identified only while the risk set contains several senders — it drops out of a
receiver-choice model and of any model stratified by sender (Perry & Wolfe
2013).

# Example
```julia
using Revel
h = build_history(Event{Float64}[])
compute(SendEffect([0.0, 1.0, 1.0]; name="senior"), h, 2, 1, 0.0)    # 1.0
```
"""
SendEffect(x; kwargs...) = CovariateEffect(:send, x; kwargs...)

"""
    ReceiveEffect(x; transform=identity, name=nothing)

The receiver's covariate value: are actors with more of `x` addressed more?
(remstats `receive()`; goldfish `alter()`; relevent `CovRec`.)

# Example
```julia
using Revel
h = build_history(Event{Float64}[])
compute(ReceiveEffect([0.0, 1.0, 1.0]), h, 2, 1, 0.0)    # 0.0
```
"""
ReceiveEffect(x; kwargs...) = CovariateEffect(:receive, x; kwargs...)

"""
    MatchEffect(x; transform=identity, name=nothing)

Homophily on a category: `1` when sender and receiver have the same value of `x`
(remstats `same()`; goldfish `same()`). `x` may be categorical.

# Example
```julia
using Revel
dept = Covariate(["ops", "legal", "ops"]; name="dept")
h = build_history(Event{Float64}[])
compute(MatchEffect(dept), h, 1, 3, 0.0), compute(MatchEffect(dept), h, 1, 2, 0.0)   # (1.0, 0.0)
```
"""
MatchEffect(x; kwargs...) = CovariateEffect(:same, x; kwargs...)

"""
    DiffEffect(x; absolute=true, transform=identity, name=nothing)

The difference between the sender's and the receiver's values: absolute by
default (heterophily — remstats `difference()`; goldfish `diff()`), or signed
`xᵢ - xⱼ` with `absolute=false` (hierarchy: does the higher-status actor address
the lower?).

# Example
```julia
using Revel
rank = [3.0, 1.0]
h = build_history(Event{Float64}[])
compute(DiffEffect(rank), h, 2, 1, 0.0)                    # 2.0
compute(DiffEffect(rank; absolute=false), h, 2, 1, 0.0)    # -2.0
```
"""
DiffEffect(x; absolute::Bool=true, kwargs...) =
    CovariateEffect(absolute ? :absdiff : :difference, x; kwargs...)

"""
    SimEffect(x; transform=identity, name=nothing)

Homophily on a numeric covariate: `-abs(xᵢ - xⱼ)` (goldfish `sim()`), the mirror
image of [`DiffEffect`](@ref).

# Example
```julia
using Revel
h = build_history(Event{Float64}[])
compute(SimEffect([3.0, 1.0]), h, 1, 2, 0.0)    # -2.0
```
"""
SimEffect(x; kwargs...) = CovariateEffect(:similarity, x; kwargs...)

"""
    AverageEffect(x; transform=identity, name=nothing)

The mean of the two actors' values (remstats `average()`), a symmetric form that
suits undirected events; half of relevent's `CovInt`.

# Example
```julia
using Revel
h = build_history(Event{Float64}[])
compute(AverageEffect([3.0, 1.0]), h, 1, 2, 0.0)    # 2.0
```
"""
AverageEffect(x; kwargs...) = CovariateEffect(:average, x; kwargs...)

"""
    MinimumEffect(x; transform=identity, name=nothing)

The smaller of the two actors' values (remstats `minimum()`): the effect of a
trait both must have — the "weakest link" form used by Meijerink-Bosman et al.
(2023) for personality traits.

# Example
```julia
using Revel
h = build_history(Event{Float64}[])
compute(MinimumEffect([3.0, 1.0]), h, 1, 2, 0.0)    # 1.0
```
"""
MinimumEffect(x; kwargs...) = CovariateEffect(:minimum, x; kwargs...)

"""
    MaximumEffect(x; transform=identity, name=nothing)

The larger of the two actors' values (remstats `maximum()`): the effect of a
trait either may have.

# Example
```julia
using Revel
h = build_history(Event{Float64}[])
compute(MaximumEffect([3.0, 1.0]), h, 1, 2, 0.0)    # 3.0
```
"""
MaximumEffect(x; kwargs...) = CovariateEffect(:maximum, x; kwargs...)

"""
    SumEffect(x; transform=identity, name=nothing)

The sum of the two actors' values under one coefficient — relevent's `CovInt`,
where "Int" means "both roles", not a statistical interaction. It is collinear
with `SendEffect(x)` plus `ReceiveEffect(x)`.

# Example
```julia
using Revel
h = build_history(Event{Float64}[])
compute(SumEffect([3.0, 1.0]), h, 1, 2, 0.0)    # 4.0
```
"""
SumEffect(x; kwargs...) = CovariateEffect(:sum, x; kwargs...)

"""
    ProductEffect(x, y=x; transform=identity, name=nothing)

The sender's value of `x` times the receiver's value of `y` (Perry & Wolfe 2013;
goldfish `egoAlterInt`; remstats `send("x"):receive("y")`). Under a
sender-stratified or receiver-choice model this is the only way a sender trait
can enter.

# Example
```julia
using Revel
senior = [1.0, 0.0, 1.0]; legal = [0.0, 1.0, 1.0]
h = build_history(Event{Float64}[])
compute(ProductEffect(senior, legal), h, 1, 2, 0.0)    # 1.0 — senior → legal
```
"""
ProductEffect(x, y=x; kwargs...) = CovariateEffect(:product, x, y; kwargs...)

# -----------------------------------------------------------------------------
# Dyadic and global covariates
# -----------------------------------------------------------------------------

"""
    TieEffect(matrix; transform=identity, name="tie")
    TieEffect(times, matrices; transform=identity, name="tie")

An exogenous dyadic covariate `matrix[sender, receiver]`: a pre-existing tie
(friendship, a reporting line, co-membership), a distance, an alliance (remstats
`tie()`; goldfish `tie()`; relevent `CovEvent`). The two-argument form is
time-varying — `matrices[k]` is in force from `times[k]`. `transform=:indicator`
gives goldfish's presence form `I(x > 0)`.

# Example
```julia
using Revel
friends = [0 1 0; 1 0 0; 0 0 0]
h = build_history(Event{Float64}[])
compute(TieEffect(friends; name="friend"), h, 1, 2, 0.0)    # 1.0
later = TieEffect([0.0, 10.0], [friends, zeros(3, 3)])
compute(later, h, 1, 2, 11.0)                                # 0.0
```
"""
struct TieEffect{F} <: AbstractRevelStatistic
    times::Vector{Float64}
    values::Vector{Matrix{Float64}}
    transform::F
    label::String
end

function TieEffect(times::AbstractVector{<:Real}, matrices::AbstractVector;
                   transform=identity, name::AbstractString="tie")
    length(times) == length(matrices) && !isempty(times) || throw(ArgumentError(
        "TieEffect needs one matrix per change time"))
    issorted(times; lt=<=) || throw(ArgumentError(
        "TieEffect: change times must be strictly increasing"))
    ms = [Matrix{Float64}(m) for m in matrices]
    n = size(ms[1], 1)
    all(m -> size(m) == (n, n), ms) || throw(ArgumentError(
        "TieEffect requires square matrices of one size"))
    all(m -> all(isfinite, m), ms) || throw(ArgumentError("TieEffect values must be finite"))
    f = _transform_fn(transform)
    return TieEffect{typeof(f)}(collect(Float64, times), ms, f,
                                _auto_name(String(name), "", f))
end

TieEffect(matrix::AbstractMatrix; kwargs...) = TieEffect([-Inf], [matrix]; kwargs...)

function _value(stat::TieEffect, events, s::Int, r::Int, t::Float64)
    k = length(stat.values) == 1 ? 1 : max(1, searchsortedlast(stat.times, t))
    m = @inbounds stat.values[k]
    n = size(m, 1)
    (1 <= s <= n && 1 <= r <= n) || throw(ArgumentError(
        "dyad ($s, $r) has no value of $(stat.label) (the matrix covers actors 1:$n)"))
    return Float64(stat.transform(@inbounds m[s, r]))
end

_uses_history(::TieEffect) = false
_interval_constant(stat::TieEffect) = length(stat.values) == 1

"""
    GlobalEffect(f; name="global")
    GlobalEffect(times, values; name="global")

A covariate of time alone — time of day, a weekday, a period after a shock:
`f(t)`, or the piecewise-constant `values[k]` from `times[k]`.

A global covariate is the same for every dyad in a risk set, so its **main
effect is not identified** by the ordinary ordinal (partial) likelihood, which
does not depend on its coefficient: [`fit_revel`](@ref) refuses a model that
holds one outside an [`Interaction`](@ref). Its use there is as a moderator:
`Interaction(GlobalEffect(…), Reciprocation())` asks whether reciprocity is
stronger in some periods, and that product *is* identified ("effectively
dyadic", Lembo, Juozaitienė, Vinciotti & Wit 2026). This is the product-term
counterpart of fitting separate models per period with
[`fit_stratified`](@ref).

Lembo et al. (2026) also recover the main effects of global covariates, with a
partial likelihood whose controls are drawn at shifted times (a time-shifted
nested case-control design); that estimator is not implemented here. The
interval-timing model identifies them too, but a `GlobalEffect` changes between
events and is refused there.

# Example
```julia
using Revel
weekend = GlobalEffect(t -> mod(floor(t), 7) >= 5 ? 1.0 : 0.0; name="weekend")
h = build_history([Event(2, 1, 1.0)])
moderated = Interaction(weekend, Reciprocation())
compute(moderated, h, 1, 2, 5.5)    # 1.0 — a weekend, one past 2 → 1 event
compute(moderated, h, 1, 2, 2.5)    # 0.0
```
"""
struct GlobalEffect{F} <: AbstractRevelStatistic
    f::F
    label::String
end

GlobalEffect(f::Function; name::AbstractString="global") =
    GlobalEffect{typeof(f)}(f, String(name))

function GlobalEffect(times::AbstractVector{<:Real}, values::AbstractVector{<:Real};
                      name::AbstractString="global")
    length(times) == length(values) && !isempty(times) || throw(ArgumentError(
        "GlobalEffect needs one value per change time"))
    issorted(times; lt=<=) || throw(ArgumentError(
        "GlobalEffect: change times must be strictly increasing"))
    ts = collect(Float64, times); vs = collect(Float64, values)
    return GlobalEffect(t -> vs[max(1, searchsortedlast(ts, t))]; name=name)
end

_value(stat::GlobalEffect, events, ::Int, ::Int, t::Float64) = Float64(stat.f(t))
_uses_history(::GlobalEffect) = false

# -----------------------------------------------------------------------------
# Covariates read through the network: tertius and matched degrees
# -----------------------------------------------------------------------------

const _AGGREGATES = (:mean, :sum, :max, :min, :sd, :range, :entropy)

# `μ` and `m2` are the running (weighted) mean and sum of squared deviations of
# Welford's update, which has no cancellation for large values with a small spread
function _aggregate(kind::Symbol, n::Float64, sw::Float64, sx::Float64, μ::Float64,
                    m2::Float64, lo::Float64, hi::Float64, empty::Float64)
    n > 0 || return empty
    kind === :mean && return sw > 0 ? μ : empty
    kind === :sum && return sx
    kind === :max && return hi
    kind === :min && return lo
    kind === :range && return hi - lo
    # :sd — the (weighted) population standard deviation of the neighbours' values
    sw > 0 || return empty
    return sqrt(max(0.0, m2 / sw))
end

"""
    TertiusEffect(x; role=:receiver, direction=:in, aggregate=:mean,
                  tie_weighted=false, difference=false, exclude_other=true,
                  empty=0.0, transform=identity, name=nothing, layer=nothing,
                  memory=…, …)

An aggregate of covariate `x` over the network neighbours of one endpoint of the
candidate dyad — goldfish's `tertius`, generalised by Haunss & Hollway (2023):
"tertius effects summarize attributes of third nodes to any dyad". By default it
is the mean of `x` over the actors who have **sent to the receiver**: is an
actor addressed more when those already addressing it score high on `x`?

- `role` — whose neighbours: the candidate's `:receiver` (goldfish
  `type="alter"`) or `:sender` (`type="ego"`).
- `direction` — `:in` (actors who sent to that endpoint) or `:out` (actors it
  sent to).
- `aggregate` — `:mean`, `:sum`, `:max`, `:min`, `:sd` (the weighted population
  standard deviation) or `:range` for a numeric covariate; `:entropy` for a
  categorical one — the Shannon entropy `−Σ p_c log p_c` of the neighbours'
  categories, the diversity measure of the "tertius party diversity" effect of
  Haunss & Hollway (2023).
- `tie_weighted` — weight each neighbour by the layer weight of its tie.
- `difference=true` — return `abs(x[sender] - aggregate)`, goldfish's
  `tertiusDiff`: homophily at path distance two, the only way to express
  homophily in a two-mode network, where the sender and the receiver carry no
  common attribute.
- `exclude_other` — leave the dyad's other endpoint out of the neighbourhood.
- `empty` — the value when the endpoint has no neighbours (goldfish imputes the
  covariate's mean; pass it here to do the same).

This is an interaction built into the statistic: an actor covariate read through
a history-defined network. It is not equivalent to a product term.

# Example
```julia
using Revel
power = [1.0, 5.0, 3.0, 0.0]
h = build_history([Event(2, 4, 1.0), Event(3, 4, 2.0)])
compute(TertiusEffect(power), h, 1, 4, 3.0)                       # 4.0 — mean of 5 and 3
compute(TertiusEffect(power; aggregate=:max), h, 1, 4, 3.0)       # 5.0
compute(TertiusEffect(power; difference=true), h, 1, 4, 3.0)      # 3.0 — abs(1 − 4)
party = [:a, :green, :red, :a]
compute(TertiusEffect(party; aggregate=:entropy), h, 1, 4, 3.0)   # log(2) — two parties, one each
```
"""
struct TertiusEffect{L<:EventLayer, F} <: AbstractRevelStatistic
    x::Covariate
    layer::L
    role::Symbol
    direction::Symbol
    aggregate::Symbol
    tie_weighted::Bool
    difference::Bool
    exclude_other::Bool
    empty::Float64
    transform::F
    label::String
    counts::Vector{Float64}          # category weights (`:entropy` only)
end

function TertiusEffect(x; layer=nothing, role::Symbol=:receiver, direction::Symbol=:in,
                       aggregate::Symbol=:mean, tie_weighted::Bool=false,
                       difference::Bool=false, exclude_other::Bool=true, empty::Real=0.0, transform=identity,
                       name=nothing, kwargs...)
    role in (:sender, :receiver) || throw(ArgumentError(
        "role must be :sender or :receiver, got :$role"))
    direction in (:in, :out) || throw(ArgumentError(
        "direction must be :in or :out, got :$direction"))
    aggregate in _AGGREGATES || throw(ArgumentError(
        "aggregate must be one of $(_AGGREGATES), got :$aggregate"))
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "TertiusEffect")
    L = _resolve_layer(layer; layer_kw...)
    c = _covariate(x)
    if aggregate === :entropy
        c.categorical || throw(ArgumentError(
            "aggregate=:entropy measures the diversity of categories and needs a " *
            "categorical covariate; :$(c.label) is numeric (use :sd or :range)"))
        difference && throw(ArgumentError(
            "difference=true compares the sender's value with the aggregate, which " *
            "needs a numeric aggregate, not :entropy"))
    else
        _require_numeric(c, "TertiusEffect(aggregate=:$aggregate)")
    end
    f = _transform_fn(transform)
    base = (difference ? "tertiusDiff" : "tertius") * ".$(c.label)" *
           (aggregate === :mean ? "" : ".$(aggregate)") *
           (role === :receiver ? "" : ".sender") * (direction === :in ? "" : ".out")
    return TertiusEffect{typeof(L), typeof(f)}(
        c, L, role, direction, aggregate, tie_weighted, difference, exclude_other,
        Float64(empty), f, _label(name, _auto_name(base, _suffix(L), f)),
        zeros(length(c.levels)))
end

function _value(stat::TertiusEffect, events, s::Int, r::Int, t::Float64)
    L = stat.layer
    st = _sync!(L, events, t)
    v, other = stat.role === :receiver ? (r, s) : (s, r)
    incoming = stat.direction === :in
    entropy = stat.aggregate === :entropy
    entropy && fill!(stat.counts, 0.0)
    n = 0.0; sw = 0.0; sx = 0.0; μ = 0.0; m2 = 0.0; lo = Inf; hi = -Inf
    for k in (incoming ? _in_nb(st, v) : _out_nb(st, v))
        k == v && continue
        stat.exclude_other && k == other && continue
        w = incoming ? _w(L, st, k, v) : _w(L, st, v, k)
        w > 0 || continue
        xk = covariate_value(stat.x, k, t)
        wk = stat.tie_weighted ? w : 1.0
        n += 1.0; sw += wk; sx += wk * xk
        δ = xk - μ
        μ += wk * δ / sw
        m2 += wk * δ * (xk - μ)
        lo = min(lo, xk); hi = max(hi, xk)
        entropy && (@inbounds stat.counts[Int(xk)] += wk)
    end
    n > 0 || return Float64(stat.transform(stat.empty))
    agg = entropy ? _entropy(stat.counts, sw) :
          _aggregate(stat.aggregate, n, sw, sx, μ, m2, lo, hi, stat.empty)
    stat.difference && (agg = abs(covariate_value(stat.x, s, t) - agg))
    return Float64(stat.transform(agg))
end

function _entropy(counts::Vector{Float64}, total::Float64)
    h = 0.0
    @inbounds for c in counts
        c > 0 && (h -= (c / total) * log(c / total))
    end
    return h
end

_interval_constant(stat::TertiusEffect) =
    _is_static(stat.x) && _layer_interval_constant(stat.layer)

"""
    MatchedDegree(x; role=:receiver, similarity=nothing, transform=identity,
                  name=nothing, layer=nothing, memory=…, …)

A degree restricted to third actors who resemble the candidate's *other*
endpoint on covariate `x` — the attribute-filtered statistic of Brandenberger's
`rem` package.

- `role=:receiver` (default): the weight of past events **to the receiver** from
  actors `k` with `x[k] == x[sender]`. This is the partisan-influence statistic
  of Malang, Brandenberger & Leifeld (2019, H1a) — does an actor join a target
  once others of its own kind (party family, department) have? They note that
  it cannot tell influence from the other diffusion mechanisms that produce the
  same pattern.
- `role=:sender`: the weight of past events **from the sender** to actors `k`
  with `x[k] == x[receiver]` — does the sender already deal with the receiver's
  kind?

The candidate's own dyad is excluded, so the statistic is not confounded with
inertia. `similarity=(a, b) -> Real` replaces the equality match with a graded
weight; it receives the covariate's values (the categories themselves, for a
categorical covariate). A filter of this kind answers "do same-kind actors' past events raise
the rate?", which a product term `IndegreeReceiver × x` does not.

# Example
```julia
using Revel
party = [1, 1, 2, 9]        # actor 4 is the target
h = build_history([Event(2, 4, 1.0), Event(3, 4, 2.0), Event(2, 4, 3.0)])
compute(MatchedDegree(party), h, 1, 4, 4.0)    # 2.0 — the two events by party-mate 2
compute(IndegreeReceiver(), h, 1, 4, 4.0)      # 3.0 — every event to 4
```
"""
struct MatchedDegree{L<:EventLayer, G, F} <: AbstractRevelStatistic
    x::Covariate
    layer::L
    role::Symbol
    similarity::G
    transform::F
    label::String
end

function MatchedDegree(x; layer=nothing, role::Symbol=:receiver, similarity=nothing,
                       transform=identity, name=nothing, kwargs...)
    role in (:sender, :receiver) || throw(ArgumentError(
        "role must be :sender or :receiver, got :$role"))
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "MatchedDegree")
    L = _resolve_layer(layer; layer_kw...)
    c = _covariate(x)
    f = _transform_fn(transform)
    base = "matched" * (role === :receiver ? "Indegree" : "Outdegree") * ".$(c.label)"
    return MatchedDegree{typeof(L), typeof(similarity), typeof(f)}(
        c, L, role, similarity, f, _label(name, _auto_name(base, _suffix(L), f)))
end

function _value(stat::MatchedDegree, events, s::Int, r::Int, t::Float64)
    L = stat.layer
    st = _sync!(L, events, t)
    total = 0.0
    if stat.role === :receiver
        ref = covariate_value(stat.x, s, t)
        for k in _in_nb(st, r)
            (k == s || k == r) && continue
            w = _w(L, st, k, r)
            w > 0 || continue
            xk = covariate_value(stat.x, k, t)
            total += w * (stat.similarity === nothing ? Float64(xk == ref) :
                          Float64(stat.similarity(_level(stat.x, ref), _level(stat.x, xk))))
        end
    else
        ref = covariate_value(stat.x, r, t)
        for k in _out_nb(st, s)
            (k == s || k == r) && continue
            w = _w(L, st, s, k)
            w > 0 || continue
            xk = covariate_value(stat.x, k, t)
            total += w * (stat.similarity === nothing ? Float64(xk == ref) :
                          Float64(stat.similarity(_level(stat.x, ref), _level(stat.x, xk))))
        end
    end
    return Float64(stat.transform(total))
end

# The value a similarity function sees: the category itself, not its code
_level(c::Covariate, code::Float64) = c.categorical ? c.levels[Int(code)] : code

_interval_constant(stat::MatchedDegree) =
    _is_static(stat.x) && _layer_interval_constant(stat.layer)
