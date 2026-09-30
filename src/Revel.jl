"""
    Revel.jl — relational event effects, interactions and diagnostics

Revel.jl builds on Relevent.jl and REM.jl. It adds the effects, the
effect × covariate interactions and the goodness-of-fit diagnostics mapped by a
review of the relational event model literature (2008–2026), organised the way
that review found the literature to be organised:

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
  distributions and a collinearity diagnostic.

Every statistic implements both compute interfaces of the ecosystem, so it works
in `Relevent.fit_obpm`/`fit_timing` (full risk set), in `REM.fit_rem` and in
Revel's own `fit_revel` (full, restricted or sampled risk sets).
"""
module Revel

using DataFrames: DataFrame, metadata!
using Distributions: Chisq, Kolmogorov, ccdf
using LinearAlgebra
using Random
using Statistics
using StatsAPI

using Networks
using REM
using REM: Event, AbstractStatistic
using Relevent
using Relevent: InteractionHistory, update_history!, PShift, pshift_types

# The shared statistic protocol (Networks.jl `src/statistics.jl`): `compute` and
# `name` are the ecosystem's generics, extended here by name — never redefined.
import REM: compute, name

# Shared presentation and GOF infrastructure, and the tied-event vocabulary
using Networks: GOFResult, GOFStatistic, n_simulations, mc_pvalue, check_tie_policy
import Networks: gof
# The result-metadata protocol, forwarded by RevelFit to the underlying fit
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 tie_method, approximations

import StatsAPI: coef, coefnames, stderror, vcov, confint, loglikelihood, nobs, dof,
                 aic, bic, coeftable

# --- shared generics and the types needed to use the package on its own -------
export compute, name
export Event, InteractionHistory, update_history!
export PShift, pshift_types
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

include("memory.jl")
include("layers.jl")
include("statistics.jl")
include("endogenous.jl")
include("covariates.jl")
include("interactions.jl")
include("design.jl")
include("fit.jl")
include("simulate.jl")
include("gof.jl")
# HYPER-INCLUDE (the hyperevent include goes directly below this line)
include("hyper.jl")
include("catalogue.jl")

end # module
