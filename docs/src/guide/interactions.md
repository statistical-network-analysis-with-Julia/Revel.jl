# Interactions

Does reciprocity depend on who is asking? Is closure stronger within a
department? Does inertia fade after a reorganisation? Questions of this kind —
an endogenous effect moderated by a covariate — are the main vehicle for theory
in applied relational event models, and the least consolidated part of the
literature.

The review behind this package found seven constructions of such a moderation
in published work — the rows of the table below, six of them implemented — and
one point that is easy to miss: **they are not the same model.** A product term asks whether the *generic* effect is
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

```@example interactions
using Revel, Random

senior = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
team = [1, 1, 1, 2, 2, 2]
events = simulate_events([Inertia(transform=:log1p), Reciprocation(transform=:log1p)],
                         [0.8, 0.8], 6, 400; rng=Xoshiro(1))
history = build_history(events[1:50])
t = events[51].time
```

## Product terms

```@example interactions
by_sender = Interaction(Reciprocation(transform=:log1p), SendEffect(senior; name="senior"))
name(by_sender)                          # "log1p(reciprocity):senior"
compute(by_sender, history, 1, 2, t)
```

The parts may be Revel's or Relevent's statistics (`PShift`, `CovSnd`, …), any
number of them. (A REM.jl statistic has only REM's interface, so an
interaction holding one works in `REM.fit_rem`, not in [`fit_revel`](@ref).) The
advice that recurs in the applied literature, though few papers state it:

- **Include the main effects.** `fit_revel(events, [a, b, Interaction(a, b)], n)`.
- **Scale cumulative statistics first** (`transform=:log1p`, or
  [`Standardized`](@ref)); the product of two unbounded counts is dominated by
  the late part of the sequence.
- **Centre a covariate** so the main effect keeps a meaning. None of the
  relational event papers the review read gives this advice, and it is as true
  here as anywhere:

```@example interactions
experience = [2.0, 11.0, 7.0, 1.0, 15.0, 4.0]
centred = Transformed(SendEffect(experience), x -> x - 6.7; name="experience_c")
moderated = Interaction(Inertia(transform=:log1p), centred)
```

- **Check what the product did to the design.** A product is correlated with
  its parts by construction:

```@example interactions
stats = [Reciprocation(transform=:log1p), SendEffect(senior; name="senior"), by_sender]
statistic_collinearity(events, stats, 6).vif
```

## Filters

A filter changes *which history counts*. It can say things a product cannot.

**Events by actors like me.** [`MatchedDegree`](@ref) is a degree counted only
over third actors who share the candidate's attribute — the partisan-influence
statistic of Malang, Brandenberger & Leifeld (2019), which, as they note, does
not tell influence from other diffusion mechanisms:

```@example interactions
compute(MatchedDegree(team), history, 1, 2, t)      # events to 2 from 1's team-mates
compute(IndegreeReceiver(), history, 1, 2, t)       # events to 2 from anyone
```

**Closure through actors like me.** The review found closure among
same-attribute actors only as a filtered statistic, never as a product term:

```@example interactions
within_team = OTP(third=matching_third(team), name="otp_within_team")
compute(within_team, history, 1, 2, t)
```

**Any predicate on the past event.** `keep` receives the past event's sender,
receiver, time, weight and type:

```@example interactions
from_seniors = Inertia(keep=(s, r, time, w, ty) -> senior[s] == 1, name="senior_inertia")
```

## Type splits

Conditioning on the type of the **past** events is the oldest construction in
the literature (signed events in Brandes, Lerner & Snijders 2009):

```@example interactions
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
only form homophily can take in a two-mode network — and with
`aggregate=:entropy` on a categorical covariate it is the diversity of the
neighbours' categories ("tertius party diversity", Haunss & Hollway 2023):

```@example interactions
compute(TertiusEffect(experience), history, 1, 2, t)                    # mean experience of 2's senders
compute(TertiusEffect(experience; difference=true), history, 1, 2, t)   # abs(x[1] − that mean)
teams = Covariate(team; name="team", categorical=true)
compute(TertiusEffect(teams; aggregate=:entropy), history, 1, 2, t)     # how mixed 2's senders are
```

## Stratified fits

```@example interactions
base = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
fits = fit_stratified(events, base, 6; by = e -> e.time <= 200 ? "first half" : "second half")
compare_coefficients(fits)
```

Every event still builds the history; only the events explained differ. The
strata must be defined by something other than who acts — a period, a context,
the type of the event — or [`fit_stratified`](@ref) gives each stratum the risk
set of the dyads that could have produced it (see its docstring). This equals a
product-term model only when *all* interactions with the stratifier are
included — then the two are the same model and the log-likelihoods add up:

```@example interactions
late = GlobalEffect(t -> t > 200 ? 1.0 : 0.0; name="late")
full = fit_revel(events, [base; [Interaction(late, s) for s in base]], 6)
loglikelihood(full) ≈ sum(loglikelihood, values(fits))     # true
```

The stratum likelihoods are disjoint factors of one partial likelihood, so the
stratum estimates are independent and their differences can be tested — a
difference in significance is not a significant difference:

```@example interactions
compare_coefficients(fits; reference="first half")    # adds difference, se_difference, p_difference
```

## Time as the moderator

```@example interactions
path = fit_moving_window(events, base, 6; width=100.0)
compare_coefficients(path)
```

[`score_process_test`](@ref) checks the model over the whole sequence without
choosing a window; a rejection says the specification drifts somewhere, not
which effect does. See [Goodness of fit](gof.md).

## Actor-oriented models change the rules

In a receiver-choice model ([`fit_receiver_choice`](@ref)) a sender attribute
has no main effect and can appear *only* as a moderator. Stadtfeld & Block
(2017, p. 340) report an estimate of opposite sign for the same effect
("outdegree × friendship") in a tie-oriented and an actor-oriented model of the
same data, and trace it to what the other effects of each model absorb.
Fitting both is a useful check.
