"""
    Revel.jl — relational event effects, interactions and diagnostics

Revel.jl specifies, fits and checks relational event models. It holds the
effects, the effect × covariate interactions and the goodness-of-fit diagnostics
mapped by a review of the relational event model literature (2008–2026),
organised the way that review found the literature to be organised:

- **four structural configurations** — the dyad (in either direction), node
  degree, the two-path and the three-path — as five parametric statistics
  (`DyadEffect`, `DegreeEffect` and `DyadDegreeEffect` for one degree or two
  combined, `TwoPathEffect`, `FourCycleEffect`), with the literature's names as
  constructors (`Inertia`, `Reciprocation`, `OTP`, `ITP`, `OSP`, `ISP`, …);
- **measurement choices that are orthogonal to the configuration** — the memory
  kernel (`FullMemory`, `HalfLife`, `Window`, `IntervalMemory`, `PowerLaw`,
  `LinearDecay`), event-type and attribute filters, event weights, scaling and
  the zero-history value — carried by an `EventLayer`;
- **exogenous effects** on static or time-varying actor covariates, dyadic
  covariates and global covariates;
- **interactions** in each construction the literature uses — product terms,
  filtered statistics, type splits, attribute-weighted statistics, stratified and
  moving-window fits — kept distinct because they are different models;
- **goodness of fit**: prediction (ranks, recall, deviance residuals), score
  processes and score tests, simulation-based auxiliary statistics, closing-time
  distributions and a collinearity diagnostic;
- **the exact full-risk-set likelihoods** of `relevent::rem.dyad` (src/engine/):
  the ordinal and the exponential interval-timing likelihood over every dyad,
  streamed interval by interval, with the shared tie and separation policies.

Every statistic implements both compute interfaces of the ecosystem, so it works
in Revel's `fit_revel` (full, restricted or sampled risk sets) and in
`REM.fit_rem`.
"""
module Revel

using DataFrames: DataFrame, metadata!
using Distributions: Chisq, Kolmogorov, Normal, ccdf
using LinearAlgebra
using Logging: with_logger, NullLogger
using PrecompileTools
using Random
using Statistics
using StatsAPI

using NetworkCore
using REM
using REM: Event, AbstractStatistic

# The shared statistic protocol (NetworkCore.jl `src/statistics.jl`): `compute` and
# `name` are the ecosystem's generics, extended here by name — never redefined.
import REM: compute, name

# Shared presentation and GOF infrastructure, and the tied-event vocabulary
using NetworkCore: GOFResult, GOFStatistic, n_simulations, mc_pvalue, check_tie_policy
# The full-risk-set likelihoods run on the shared optimizer and coefficient table,
# and follow the ecosystem's one separation verdict and policy (NetworkCore.jl
# `src/separation.jl`): warn, report `converged == false`, flag the separated
# terms and withhold inference.
using NetworkCore: CoefficientTable, newton_fit, z_pvalues, SeparationVerdict,
                   separation_from_margins, warn_separation, separation_caveat
import NetworkCore: gof
# The result-metadata protocol, forwarded by RevelFit to the underlying fit
import NetworkCore: estimand, objective, is_exact, se_method, missing_method,
                 tie_method, approximations

import StatsAPI: coef, coefnames, stderror, vcov, confint, loglikelihood, nobs, dof,
                 aic, aicc, bic, coeftable

# --- shared generics and the types needed to use the package on its own -------
export compute, name
export Event, InteractionHistory, update_history!
export PShift, pshift_types
# the full-risk-set estimators behind `fit_revel` and their results: public,
# reached as `Revel.fit_obpm`, …; `fit_revel` is the entry point
public fit_obpm, fit_timing, OrdinalBPMResult, TimingModelResult, is_interval_constant
export coef, coefnames, stderror, vcov, confint, loglikelihood, nobs, dof, aic, bic,
       coeftable
export gof, n_simulations

# --- memory kernels and layers ------------------------------------------------
export AbstractMemory, FullMemory, HalfLife, Window, IntervalMemory, PowerLaw, LinearDecay,
       KernelMemory
export kernel_weight, interval_partition
export EventLayer

# --- statistics ---------------------------------------------------------------
export AbstractRevelStatistic, build_history
# the five configurations
export DyadEffect, DegreeEffect, DyadDegreeEffect, TwoPathEffect, FourCycleEffect
# their named forms
export Inertia, Reciprocation, DyadActivity
export OutdegreeSender, IndegreeSender, TotaldegreeSender
export OutdegreeReceiver, IndegreeReceiver, TotaldegreeReceiver
export TotaldegreeDyad, DegreeMin, DegreeMax, DegreeDiff, DegreeAssortativity
export OTP, ITP, OSP, ISP, SharedPartners, BalanceEffect, matching_third
# order-based devices and neighbourhood statistics
export RecencyRank, TimeSince, inverse_gap, PShiftABAB, UndirectedPShift
export NodeTransitivity, StructuralSimilarity

# --- covariates ---------------------------------------------------------------
export Covariate, covariate_value, CovariateEffect
export SendEffect, ReceiveEffect, MatchEffect, DiffEffect, SimEffect
export AverageEffect, MinimumEffect, MaximumEffect, SumEffect, ProductEffect
export TieEffect, GlobalEffect
export TertiusEffect, MatchedDegree

# --- interactions and wrappers --------------------------------------------------
export Interaction, Transformed, Standardized, split_by_type

# --- designs and fitting --------------------------------------------------------
export RiskSetView, each_risk_set, event_design, two_mode_dyads
export RevelFit, fit_revel, revel, fit_receiver_choice
export fit_stratified, fit_moving_window, compare_coefficients, profile_memory
export simulate_events

# --- goodness of fit and diagnostics --------------------------------------------
export event_diagnostics, prediction_summary
export score_process_test, score_test
export mechanism_shares, closing_times, statistic_collinearity

# --- relational hyperevents (src/hyper.jl) ---------------------------------------
# HYPER-EXPORTS (the hyperevent exports go directly below this line)
export HyperEvent, HyperHistory, update_hyper_history!, build_hyper_history
export is_directed, participants
export AbstractHyperStatistic, HyperedgeSize, HyperCovariate
export SubsetRepetition, ExactRepetition, SharedPriorEvents, PriorSuccess
export DirectedSubsetRepetition, UnorderedRepetition, ReceiverSetRepetition,
       SenderReceiverSetRepetition, HyperSenderActivity, HyperReceiverPopularity,
       HyperReciprocation, OutInPopularity, InteractionAmongReceivers
export HyperClosure
export hyper_design, HyperFit, fit_rhem, rhem, simulate_hyperevents

# --- the concordance ------------------------------------------------------------
export effect_catalogue

include("engine/history.jl")
include("engine/pshift.jl")
include("memory.jl")
include("layers.jl")
include("statistics.jl")
include("endogenous.jl")
include("covariates.jl")
include("interactions.jl")
include("design.jl")
# The full-risk-set engine: risk sets, the two likelihoods, separation, results
include("engine/risksets.jl")
include("engine/likelihoods.jl")
include("engine/separation.jl")
include("engine/ordinal.jl")
include("engine/timing.jl")
include("engine/results.jl")
include("fit.jl")
include("simulate.jl")
include("gof.jl")
# HYPER-INCLUDE (the hyperevent include goes directly below this line)
include("hyper.jl")
include("catalogue.jl")

# ---------------------------------------------------------------------------
# Precompile workload
# ---------------------------------------------------------------------------
# The README's calls, replayed as the README writes them, so that their
# methods are cached in the package image: a statistic vector built by `[a, b, c]`
# and one built by `[stats; d; e]` are different specialisations, and
# `load_dataset` is recompiled after any package that loads Distributions
# invalidates NetworkCore's copy. A prefix of the World Trade Center calls keeps
# it cheap; the types are those of the full data. Then the rest of the README
# path on a toy sequence: simulation, an exact fit (the streamed full-risk-set
# engine), a sampled fit (the design engine and REM.fit_rem), `show` and a
# two-replicate `gof`. It must stay silent: every
# fit below converges, and all randomness is a local `Xoshiro`.
@setup_workload begin
    _pc_stats = [SumEffect([1.0, 0.0, 0.0, 1.0, 0.0]; name="x"),
                 Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    @compile_workload begin
        wtc = NetworkCore.load_dataset(:wtc_police_calls)
        calls = [Event(row[2], row[3], Float64(row[1])) for row in eachrow(wtc.events)]
        calls = calls[1:80]
        coordinator = SumEffect(Float64.(wtc.is_icr); name="coordinator")
        stats = [coordinator, Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
        fit = fit_revel(calls, stats, wtc.n_actors)
        sprint(show, MIME"text/plain"(), coeftable(fit))
        sprint(show, MIME"text/plain"(),
               score_test(fit, [PShift(:AB_BA), RecencyRank(:send), OTP(transform=:log1p)]))
        better = fit_revel(calls, [stats; PShift(:AB_BA); RecencyRank(:send)], wtc.n_actors)
        prediction_summary(better).recall
        sprint(show, MIME"text/plain"(), better)

        events = simulate_events(_pc_stats, [0.5, 0.8, 0.6], 5, 60;
                                 rng=Random.Xoshiro(20261002))
        toy = fit_revel(events, _pc_stats, 5)
        coef(toy); stderror(toy)
        sampled = fit_revel(events, _pc_stats, 5; n_controls=6, rng=Random.Xoshiro(1))
        sprint(show, MIME"text/plain"(), sampled)
        gof(toy; n_sim=2, rng=Random.Xoshiro(2))
    end
end

end # module
