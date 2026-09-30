# Covariates

Exogenous effects are inherited almost unchanged from Cox regression and from
ERGM/SAOM practice. What the relational event literature added is an
**identification rule**: a term that is the same for every dyad in an event's
risk set cannot be estimated from an ordinal likelihood. It decides which of
the effects below can enter which model.

```julia
using Revel

history = build_history(Event{Float64}[])      # covariate effects need no history
age = [30.0, 45.0, 52.0, 38.0]
```

## Actor covariates

A covariate is a vector indexed by actor ID, or a [`Covariate`](@ref) when it
needs a name, categories or a time dimension.

| Effect | Value for `s → r` | remstats | goldfish | relevent |
|---|---|---|---|---|
| [`SendEffect`](@ref) | `x[s]` | `send()` | `ego()` | `CovSnd` |
| [`ReceiveEffect`](@ref) | `x[r]` | `receive()` | `alter()` | `CovRec` |
| [`SumEffect`](@ref) | `x[s] + x[r]` | | | `CovInt` |
| [`AverageEffect`](@ref) | `(x[s] + x[r])/2` | `average()` | | |
| [`MatchEffect`](@ref) | `1` if equal | `same()` | `same()` | |
| [`DiffEffect`](@ref) | `abs(x[s] − x[r])` | `difference()` | `diff()` | |
| [`SimEffect`](@ref) | `−abs(x[s] − x[r])` | | `sim()` | |
| [`MinimumEffect`](@ref), [`MaximumEffect`](@ref) | `min`, `max` | `minimum()`, `maximum()` | | |
| [`ProductEffect`](@ref) | `x[s] · y[r]` | `send():receive()` | `egoAlterInt()` | |

```julia
compute(SendEffect(age), history, 2, 1, 0.0)                          # 45.0
compute(DiffEffect(age), history, 2, 1, 0.0)                    # 15.0
compute(DiffEffect(age; absolute=false), history, 2, 1, 0.0)    # 15.0 — sender minus receiver
compute(AverageEffect(age), history, 2, 1, 0.0)                       # 37.5
```

Categories need no coding:

```julia
dept = Covariate(["ops", "legal", "ops", "legal"]; name="dept")
compute(MatchEffect(dept), history, 1, 3, 0.0)    # 1.0
name(MatchEffect(dept))                            # "same.dept"
```

An actor the covariate does not cover is an `ArgumentError`, never a silent
zero, and a numeric effect refuses a categorical covariate.

## Covariates that change over time

A time-varying covariate is piecewise constant: one column of values per change
time.

```julia
rank = Covariate([0.0, 100.0], [1.0 2.0; 1.0 1.0; 3.0 3.0; 2.0 2.0]; name="rank")
compute(SendEffect(rank), history, 1, 2, 50.0)     # 1.0
compute(SendEffect(rank), history, 1, 2, 150.0)    # 2.0 — promoted at t = 100
```

Every covariate effect, and every effect built on a covariate
([`TertiusEffect`](@ref), [`MatchedDegree`](@ref)), accepts one.

## Dyadic covariates

[`TieEffect`](@ref) reads a matrix `X[s, r]` — a pre-existing tie, a reporting
line, a distance, an alliance — static or changing over time:

```julia
reports_to = [0 1 0 0; 0 0 0 0; 0 1 0 0; 1 0 0 0]
compute(TieEffect(reports_to; name="reports_to"), history, 1, 2, 0.0)   # 1.0
```

## Global covariates

[`GlobalEffect`](@ref) is a function of time alone: a weekday, a shift, a period
after a shock. It is the same for every dyad, so **its main effect is not
identified** in an ordinal model — a specification that includes one on its own
is singular, and the fit says so. It exists to be interacted:

```julia
night = GlobalEffect(t -> mod(t, 24) >= 20 ? 1.0 : 0.0; name="night")
more_reciprocal_at_night = Interaction(night, Reciprocation())
```

## What is identified where

| Term | Full risk set | Receiver choice (`riskset=:sender`) | Timing model |
|---|---|---|---|
| `SendEffect(x)` | yes | **no** — constant within every choice set | yes |
| `ReceiveEffect(x)`, dyadic forms | yes | yes | yes |
| `GlobalEffect(f)` | **no** | **no** | not admissible (it changes between events) |
| `Interaction(GlobalEffect(f), stat)` | yes | yes | not admissible |
| `Interaction(SendEffect(x), stat)` | yes | yes — the only way a sender trait enters | yes |

[`statistic_collinearity`](@ref) reports a term with no variation inside the
risk sets (an infinite variance inflation factor) before any model is fitted.

## Covariates read through the network

Two effects combine a covariate with the history and belong to
[Interactions](interactions.md): [`TertiusEffect`](@ref), an aggregate of a
covariate over an actor's network neighbours, and [`MatchedDegree`](@ref), a
degree counted over third actors who resemble the other endpoint.

## Relevent's and REM's covariate effects

`CovSnd`, `CovRec`, `CovInt`, `CovEvent` and the fixed effects `FESnd`, `FERec`,
`FEInt` of Relevent.jl, and the `NodeAttribute` statistics of REM.jl, can be
mixed freely with Revel's in one model: they are methods of the same `compute`
generic.
