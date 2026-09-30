# Covariates

Exogenous effects are inherited almost unchanged from Cox regression and from
ERGM/SAOM practice. What the relational event literature added is an
**identification rule**: a term that is the same for every dyad in an event's
risk set cannot be estimated from an ordinal likelihood. It decides which of
the effects below can enter which model.

```@example covariates
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

```@example covariates
compute(SendEffect(age), history, 2, 1, 0.0)                          # 45.0
compute(DiffEffect(age), history, 2, 1, 0.0)                    # 15.0
compute(DiffEffect(age; absolute=false), history, 1, 2, 0.0)    # -15.0 — sender minus receiver
compute(AverageEffect(age), history, 2, 1, 0.0)                       # 37.5
```

Categories need no coding:

```@example covariates
dept = Covariate(["ops", "legal", "ops", "legal"]; name="dept")
compute(MatchEffect(dept), history, 1, 3, 0.0)    # 1.0
name(MatchEffect(dept))                            # "same.dept"
```

An actor the covariate does not cover is an `ArgumentError`, never a silent
zero, and a numeric effect refuses a categorical covariate.

## Covariates that change over time

A time-varying covariate is piecewise constant: one column of values per change
time. The first column also applies before the first change time — it is
carried backwards, not treated as missing.

```@example covariates
rank = Covariate([0.0, 100.0], [1.0 2.0; 1.0 1.0; 3.0 3.0; 2.0 2.0]; name="rank")
compute(SendEffect(rank), history, 1, 2, 50.0)     # 1.0
compute(SendEffect(rank), history, 1, 2, 150.0)    # 2.0 — promoted at t = 100
```

Every covariate effect, and every effect built on a covariate
([`TertiusEffect`](@ref), [`MatchedDegree`](@ref)), accepts one. A `missing`
value is refused: an unobserved attribute is not a category, and treating it as
one would make two missing values "match".

## Dyadic covariates

[`TieEffect`](@ref) reads a matrix `X[s, r]` — a pre-existing tie, a reporting
line, a distance, an alliance — static or changing over time:

```@example covariates
reports_to = [0 1 0 0; 0 0 0 0; 0 1 0 0; 1 0 0 0]
compute(TieEffect(reports_to; name="reports_to"), history, 1, 2, 0.0)   # 1.0
```

## Global covariates

[`GlobalEffect`](@ref) is a function of time alone: a weekday, a shift, a period
after a shock. It is the same for every dyad, so **its main effect is not
identified** by the ordinal likelihood, and [`fit_revel`](@ref) refuses a model
that holds one outside an interaction. Its use is as a moderator (here with
times in hours):

```@example covariates
night = GlobalEffect(t -> mod(t, 24) >= 20 ? 1.0 : 0.0; name="night")
more_reciprocal_at_night = Interaction(night, Reciprocation())
```

Lembo, Juozaitienė, Vinciotti & Wit (2026) recover the main effects of global
covariates with a partial likelihood whose controls are drawn at shifted times;
that estimator is not implemented here.

## What is identified where

| Term | Full risk set | Receiver choice (`riskset=:sender`) | Timing model |
|---|---|---|---|
| `SendEffect(x)` | yes | **no** — constant within every choice set | yes, if `x` is static (a time-varying covariate changes between events) |
| `ReceiveEffect(x)`, dyadic forms | yes | yes | yes |
| `GlobalEffect(f)` | **no** — refused | **no** — refused | not admissible (it changes between events) |
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
`FEInt` of Relevent.jl can be mixed freely with Revel's in one
[`fit_revel`](@ref) model: they are methods of the same `compute` generic, in the
same (history) signature. REM.jl's `NodeAttribute` statistics implement only
REM's signature, so they mix with Revel's inside `REM.fit_rem` — where every
Revel statistic works too — and `fit_revel` refuses them with an
`ArgumentError` that says so.
