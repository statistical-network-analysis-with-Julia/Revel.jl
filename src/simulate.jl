# =============================================================================
# Simulation
# =============================================================================

# Draw an index with probability proportional to exp(η); returns (index, log Σ exp(η))
function _sample_softmax(rng::AbstractRNG, η::Vector{Float64}, n::Int)
    ηmax = -Inf
    @inbounds for d in 1:n
        ηmax = max(ηmax, η[d])
    end
    isfinite(ηmax) || throw(ArgumentError(
        "the linear predictor is not finite for some candidate; the process has " *
        "exploded (scale cumulative statistics with transform=:log1p or " *
        "Standardized, or use a decaying memory)"))
    total = 0.0
    @inbounds for d in 1:n
        total += exp(η[d] - ηmax)
    end
    u = rand(rng) * total
    acc = 0.0
    @inbounds for d in 1:n
        acc += exp(η[d] - ηmax)
        u <= acc && return d, ηmax + log(total)
    end
    return n, ηmax + log(total)
end

"""
    simulate_events(statistics, coefficients, n_actors, n_events;
                    rng=Random.default_rng(), times=nothing, baseline=nothing,
                    directed=true, riskset=:full, senders=nothing,
                    history=Event{Float64}[], eventtype=:event, weights=nothing)
        -> Vector{Event{Float64}}

Simulate a relational event sequence from a model: at each step the next dyad is
drawn from the risk set with probability proportional to `exp(θ'x)`, the
statistics `x` being read off the events simulated so far.

The clock is one of three:

- default — events occur at times `1, 2, …, n_events` (an ordinal model says
  nothing about time);
- `times=` — the given event times are used, in order. This is how a fitted
  ordinal model is simulated for goodness of fit: conditional on the observed
  timestamps, so that time-based memory (a half-life in clock units) keeps its
  meaning;
- `baseline=λ₀` — waiting times are exponential with rate `λ₀ Σ exp(θ'x)`, the
  interval-timing model. The hazard is held constant between events, so every
  statistic must be constant between events ([`FullMemory`](@ref) layers).

`riskset` is `:full`, a vector of dyads (e.g. [`two_mode_dyads`](@ref)), or
`:sender` together with `senders=` (a vector giving the sender of each event) to
simulate receiver choices only. `history` seeds the process with earlier events.
`eventtype` (one `Symbol`, or one per event) and `weights` (one per event) are
exogenous marks stamped on the simulated events — the ordinal model conditions
on them. All randomness comes from `rng`.

Cumulative statistics with positive coefficients feed back on themselves and can
make the process explode; scale them (`transform=:log1p`, [`Standardized`](@ref))
or use a decaying memory.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [1.0, 0.5], 5, 100; rng=Xoshiro(1))
length(events), events[end].time         # (100, 100.0)
timed = simulate_events([Inertia(transform=:log1p)], [1.0], 5, 50;
                        baseline=0.1, rng=Xoshiro(1))
issorted(e.time for e in timed)          # true
```
"""
function simulate_events(statistics, coefficients::AbstractVector{<:Real},
                         n_actors::Int, n_events::Int;
                         rng::AbstractRNG=Random.default_rng(), times=nothing,
                         baseline::Union{Nothing,Real}=nothing, directed::Bool=true,
                         riskset=:full, senders=nothing,
                         history::AbstractVector{<:Event}=Event{Float64}[],
                         eventtype=:event, weights=nothing)
    stats = Tuple(statistics)
    p = length(stats)
    p >= 1 || throw(ArgumentError("need at least one statistic"))
    length(coefficients) == p || throw(ArgumentError(
        "$(length(coefficients)) coefficients for $p statistics"))
    n_events >= 0 || throw(ArgumentError("n_events must be non-negative"))
    θ = collect(Float64, coefficients)

    times === nothing || baseline === nothing || throw(ArgumentError(
        "pass either `times` (given event times) or `baseline` (simulated waiting " *
        "times), not both"))
    if times !== nothing
        length(times) == n_events || throw(ArgumentError(
            "$(length(times)) event times for n_events = $n_events"))
        issorted(times) || throw(ArgumentError("`times` must be non-decreasing"))
    end
    if baseline !== nothing
        baseline > 0 || throw(ArgumentError("baseline rate must be positive"))
        for stat in stats
            Relevent.is_interval_constant(stat) || throw(ArgumentError(
                "simulating waiting times holds the hazard constant between events, " *
                "which requires statistics that do not change between events; " *
                "$(name(stat)) does (use FullMemory layers, or simulate the event " *
                "order with `times=`)"))
        end
    end

    eventtype isa Symbol || length(eventtype) == n_events || throw(ArgumentError(
        "`eventtype` must be one Symbol or one per event"))
    weights === nothing || length(weights) == n_events || throw(ArgumentError(
        "`weights` must hold one weight per event"))

    by_sender = riskset === :sender
    if by_sender
        senders === nothing && throw(ArgumentError(
            "riskset=:sender simulates receiver choices and needs `senders=`, the " *
            "sender of each event"))
        length(senders) == n_events || throw(ArgumentError(
            "$(length(senders)) senders for n_events = $n_events"))
        directed || throw(ArgumentError("riskset=:sender needs directed events"))
    elseif !(riskset === :full || riskset isa AbstractVector)
        throw(ArgumentError(
            "simulate_events takes riskset=:full, a vector of dyads, or :sender " *
            "with `senders=`; a risk set defined from the observed sequence " *
            "(:active, a function) cannot be simulated from"))
    end
    dyads = by_sender ? Vector{Tuple{Int,Int}}(undef, n_actors - 1) :
            riskset === :full ? _full_dyads(n_actors, directed) :
            [(Int(s), Int(r)) for (s, r) in riskset]

    seed = sort(collect(Event{Float64},
                        (Event(e.sender, e.receiver, _tfloat(e.time);
                               eventtype=e.eventtype, weight=e.weight) for e in history));
                by=e -> e.time)
    h = build_history(seed)
    t = isempty(seed) ? 0.0 : seed[end].time
    out = Event{Float64}[]
    sizehint!(out, n_events)
    D = length(dyads)
    η = Vector{Float64}(undef, D)

    for m in 1:n_events
        if by_sender
            k = 0
            for a in 1:n_actors
                a == senders[m] && continue
                k += 1
                dyads[k] = (Int(senders[m]), a)
            end
        end
        # The statistics are read at the time of the event being drawn, except
        # under a simulated clock, where they are held at the last event's time
        t_read = times !== nothing ? Float64(times[m]) :
                 baseline === nothing ? t + 1.0 : t
        _linear_predictor!(η, stats, θ, h, dyads, t_read)
        d, logtotal = _sample_softmax(rng, η, D)
        t = baseline === nothing ? t_read : t + randexp(rng) / (baseline * exp(logtotal))
        s, r = dyads[d]
        ev = Event(s, r, t; eventtype=eventtype isa Symbol ? eventtype : Symbol(eventtype[m]),
                   weight=weights === nothing ? 1.0 : Float64(weights[m]))
        update_history!(h, ev)
        push!(out, ev)
    end
    return out
end

function _linear_predictor!(η::Vector{Float64}, stats::S, θ::Vector{Float64}, history,
                            dyads, t::Float64) where S<:Tuple
    @inbounds for (d, (s, r)) in enumerate(dyads)
        vals = map(stat -> compute(stat, history, s, r, t), stats)
        acc = 0.0
        for k in eachindex(vals)
            acc += θ[k] * vals[k]
        end
        η[d] = acc
    end
    return η
end
