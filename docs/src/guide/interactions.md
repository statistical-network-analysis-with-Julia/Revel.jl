# Interactions

Does reciprocity depend on who is asking? Is closure stronger within a
department? Does inertia fade after a reorganisation? Questions of this kind —
an endogenous effect moderated by a covariate — are the main vehicle for theory
in applied relational event models, and the least consolidated part of the
literature.

The review behind this package found eight ways in which published work
constructs such a moderation, and one point that is easy to miss: **they are
not the same model.** A product term asks whether the *generic* effect is
stronger for some actors. A filter asks whether the past events of *some
actors* raise the rate. Separate fits let every coefficient differ. Revel
implements each construction under its own name so that the choice is visible.

| Construction | In Revel | Asks |
|---|---|---|
| product term | [`Interaction`](@ref) | is the effect stronger when the covariate is high? |
| filtered history | `keep=`, [`MatchedDegree`](@ref), [`matching_third`](@ref) | do events *by or among* some actors matter? |
| type split | `types=`, [`split_by_type`](@ref), [`BalanceEffect`](@ref) | do past events of one kind matter? |
| attribute-weighted | [`TertiusEffect`](@ref) | do the traits of an actor's neighbours matter? |
| stratified fit | [`fit_stratified`](@ref) | does the whole model differ between contexts? |
| time-varying | [`fit_moving_window`](@ref), [`GlobalEffect`](@ref) | does the effect change over the sequence? |
| random slope, cross-level | not implemented | — |

```julia
using Revel, Random

senior = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
team = [1, 1, 1, 2, 2, 2]
events = simulate_events([Inertia(transform=:log1p), Reciprocation(transform=:log1p)],
                         [0.8, 0.8], 6, 400; rng=Xoshiro(1))
history = build_history(events[1:50])
t = events[51].time
```

## Product terms

```julia
by_sender = Interaction(Reciprocation(transform=:log1p), SendEffect(senior; name="senior"))
name(by_sender)                          # "log1p(reciprocity):senior"
compute(by_sender, history, 1, 2, t)
```

The parts may be any statistics of the ecosystem — Revel's, Relevent's `PShift`,
REM's — and any number of them. The literature's guidance is thin but
convergent:

- **Include the main effects.** `fit_revel(events, [a, b, Interaction(a, b)], n)`.
- **Scale cumulative statistics first** (`transform=:log1p`, or
  [`Standardized`](@ref)); the product of two unbounded counts is dominated by
  the late part of the sequence.
- **Centre a covariate** so the main effect keeps a meaning — no source in the
  review gives this advice for relational event models, and it is as true here
  as anywhere:

```julia
experience = [2.0, 11.0, 7.0, 1.0, 15.0, 4.0]
centred = Transformed(SendEffect(experience), x -> x - 6.7; name="experience_c")
moderated = Interaction(Inertia(transform=:log1p), centred)
```

- **Check what the product did to the design.** A product is correlated with
  its parts by construction:

```julia
stats = [Reciprocation(transform=:log1p), SendEffect(senior; name="senior"), by_sender]
statistic_collinearity(events, stats, 6).vif
```

## Filters

A filter changes *which history counts*. It can say things a product cannot.

**Events by actors like me.** [`MatchedDegree`](@ref) is a degree counted only
over third actors who share the candidate's attribute — the contagion statistic
of Malang, Brandenberger & Leifeld (2019):

```julia
compute(MatchedDegree(team), history, 1, 2, t)      # events to 2 from 1's team-mates
compute(IndegreeReceiver(), history, 1, 2, t)       # events to 2 from anyone
```

**Closure through actors like me.** The review found closure among
same-attribute actors only as a filtered statistic, never as a product term:

```julia
within_team = OTP(third=matching_third(team), name="otp_within_team")
compute(within_team, history, 1, 2, t)
```

**Any predicate on the past event.** `keep` receives the past event's sender,
receiver, time, weight and type:

```julia
from_seniors = Inertia(keep=(s, r, time, w, ty) -> senior[s] == 1, name="senior_inertia")
```

## Type splits

Conditioning on the type of the **past** events is the oldest construction in
the literature (signed events in Brandes, Lerner & Snijders 2009):

```julia
effects = split_by_type(Reciprocation, [:praise, :blame])
name.(effects)      # ["reciprocity[types=praise]", "reciprocity[types=blame]"]
```

Two legs of a triad can read different types, which gives the balance
statistics ([`BalanceEffect`](@ref)) and cross-network closure
([`TwoPathEffect`](@ref) with two layers).

There is a second, independent axis: whether the coefficient differs by the
type of the event *being explained*. That needs a risk set of dyad × type
(remstats `consider_type = "interact"`), which is not implemented; fit the
outcome types separately instead:

```jl
fit_stratified(events, stats, n; by = e -> e.eventtype)
```

## Attribute-weighted statistics

[`TertiusEffect`](@ref) aggregates a covariate over the actors tied to one end
of the dyad. With `difference=true` it is homophily at path distance two — the
only form homophily can take in a two-mode network:

```julia
compute(TertiusEffect(experience), history, 1, 2, t)                    # mean experience of 2's senders
compute(TertiusEffect(experience; difference=true), history, 1, 2, t)   # abs(x[1] − that mean)
```

## Stratified fits

```julia
base = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
fits = fit_stratified(events, base, 6; by = e -> e.time <= 200 ? "first half" : "second half")
compare_coefficients(fits)
```

Every event still builds the history; only the events explained differ. This
equals a product-term model only when *all* interactions with the stratifier
are included — then the two are the same model and the log-likelihoods add up:

```julia
late = GlobalEffect(t -> t > 200 ? 1.0 : 0.0; name="late")
full = fit_revel(events, [base; [Interaction(late, s) for s in base]], 6)
loglikelihood(full) ≈ sum(loglikelihood, values(fits))     # true
```

## Time as the moderator

```julia
path = fit_moving_window(events, base, 6; width=100.0)
compare_coefficients(path)
```

[`score_process_test`](@ref) tests the hypothesis that a coefficient is constant
over the sequence without choosing a window; see [Goodness of fit](gof.md).

## Actor-oriented models change the rules

In a receiver-choice model ([`fit_receiver_choice`](@ref)) a sender attribute
has no main effect and can appear *only* as a moderator. In a tie-oriented model
an inertia × sender-covariate term conflates "this kind of sender is more
active" with "this kind of sender is more repetitive"; Stadtfeld & Block (2017)
report estimates of opposite sign from the two models on the same data. Fitting
both is a useful check.
