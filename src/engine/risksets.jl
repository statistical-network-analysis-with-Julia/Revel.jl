# =============================================================================
# The full risk set, interval by interval
# =============================================================================
#
# The two exact likelihoods (`fit_obpm`, `fit_timing`) need, for every interval
# m (in time order): the case's index into the full risk set, the waiting time,
# and the `n(n−1) × p` matrix of statistics for EVERY dyad in the risk set, read
# off the strictly pre-event interaction history (no look-ahead).
#
# Materializing all of those matrices costs `O(E · n(n−1) · p)` doubles — 906 MiB
# at (n, E, p) = (100, 2000, 6) — which puts a ceiling on exact full-risk-set
# estimation long before the arithmetic becomes infeasible. Everything else
# about an interval is O(1) or O(n²) *once*, so the work is split:
#
#   * `_RiskSetPlan` — the O(E + n²) skeleton, always materialized: which dyad
#     is the case, the waiting time, the time at which the statistics are read,
#     and which events the history absorbs after the interval is emitted (the
#     tie-freeze policies absorb a whole block at once). Also the sparse Efron
#     denominator weights: within a tie block every tied case takes the SAME
#     weight 1 − (j−1)/d, so a list of dyad indices plus one scalar per interval
#     replaces a dense `n(n−1)`-vector per event.
#
#   * `_RiskSets` — the design matrices, under one of three policies:
#       `:all`     every matrix materialized once: fastest, `O(E · n² · p)`.
#       `:chunked` a BOUNDED cache of `chunk` matrices, refilled by replaying the
#                  history from the start on each pass: `O(chunk · n² · p)`.
#       `:none`    `:chunked` with `chunk = 1` — one matrix alive at a time.
#     The passes visit the intervals in the SAME order under every policy and the
#     matrices hold the same values (a design matrix is a deterministic function
#     of the pre-interval history and the read time), so the likelihood, gradient
#     and Hessian are accumulated in an identical order and the fits are
#     bit-identical — asserted in the tests, not hoped for.
#
# `cache=:auto` (the default) is `:all` while the projected footprint fits in
# `cache_bytes` (256 MiB for the fitters called directly; `fit_revel` passes a
# quarter of the free memory) and `:chunked` above it.
#
# Statistics are converted to a tuple so the inner loop over dyads compiles to
# statically dispatched compute calls instead of dynamic dispatch through an
# abstractly-typed vector.

# -----------------------------------------------------------------------------
# Tied event times
# -----------------------------------------------------------------------------
#
# Both likelihoods claim more than an event list gives them when two events
# share a timestamp, and they claim DIFFERENT things, so they take different
# subsets of the shared `NetworkCore.TIE_POLICIES` vocabulary:
#
#   fit_obpm  — a likelihood over the ORDER of events. A tie means the order is
#               genuinely unknown; sorting it invents information (and, because
#               the statistics are read off the pre-event history, lets the event
#               placed first enter the statistics of the one placed second). The
#               likelihood is a multinomial partial likelihood, so the classical
#               Cox tie corrections apply verbatim:
#                 :error (default) | :ordered | :breslow | :efron
#               `:batch` is refused — with the history frozen across the tied
#               events, a "simultaneous batch" IS the Breslow correction.
#
#   fit_timing — an exact-time (exponential) likelihood. Under a continuous-time
#               model P(tie) = 0: a tie is not a broken ordering but a violated
#               assumption — a coarsened clock, or a genuinely simultaneous
#               batch. There is no partial likelihood here to correct, so
#               :breslow and :efron are refused (they are undefined here):
#                 :error (default) | :ordered | :batch
#               `:batch` reads the tie as one simultaneous batch: the events
#               cannot have influenced one another (history frozen across the
#               block) and the block consumes ONE exposure interval. `:ordered`
#               instead lets each tied event after the first enter with a
#               ZERO-LENGTH waiting interval — exposure the model then never
#               sees — while still updating the history, i.e. it claims that one
#               event caused the next in no time at all.

const _OBPM_TIES_SUPPORTED = (:error, :ordered, :breslow, :efron)
const _OBPM_TIES_MODEL = "`fit_obpm` (the ordinal likelihood: a likelihood over the ORDER of events)"
const _OBPM_TIES_REASONS = Dict(
    :batch => "an ordinal likelihood has no exposure interval for a batch to " *
              "consume; holding the risk set fixed across the tied events and " *
              "giving each its own multinomial term IS the Breslow correction, " *
              "so pass `ties=:breslow` (or `:efron`) instead")

const _TIMING_TIES_SUPPORTED = (:error, :ordered, :batch)
const _TIMING_TIES_MODEL =
    "`fit_timing` (exponential-baseline interval likelihood: an EXACT-TIME model)"
const _PARTIAL_ONLY =
    "Breslow and Efron are corrections to a PARTIAL likelihood, in which the " *
    "baseline hazard is profiled out and only the order of events is used. " *
    "`fit_timing` maximizes the exact exponential likelihood of the waiting " *
    "TIMES; there is no partial-likelihood denominator here for them to " *
    "re-weight. Ties under a continuous-time model are a violated assumption, " *
    "not a broken ordering: read them as a coarsened, simultaneous batch " *
    "(`ties=:batch`), or fit the ordinal model with `fit_obpm(...; "
const _TIMING_TIES_REASONS = Dict(
    :breslow => _PARTIAL_ONLY * "ties=:breslow)` if the order is what you care about",
    :efron   => _PARTIAL_ONLY * "ties=:efron)` if the order is what you care about")

# `ties=:error`: name the tie, do not fit. `claim` says what the model is
# claiming that the tied data cannot support. (`_tie_blocks` is in design.jl.)
function _reject_ties(sorted::Vector{Event{T}}, blocks::Vector{UnitRange{Int}},
                      claim::AbstractString, advice::AbstractString) where T
    tied = filter(b -> length(b) > 1, blocks)
    isempty(tied) && return nothing
    b = first(tied)
    n_tied_events = sum(length, tied)
    throw(ArgumentError(
        "Event sequence contains tied timestamps: events $(first(b))–$(last(b)) " *
        "($(length(b)) of them, in time order) all occur at t = " *
        "$(sorted[first(b)].time)" *
        (length(tied) > 1 ?
         "; $(length(tied)) timestamps carry ties in all ($n_tied_events events)" :
         "") * ". $claim $advice"))
end

# -----------------------------------------------------------------------------
# The plan
# -----------------------------------------------------------------------------

const _CACHE_MODES = (:auto, :all, :chunked, :none)
const _DEFAULT_CACHE_BYTES = 1 << 28      # 256 MiB

struct _RiskSetPlan{T, S<:Tuple}
    sorted::Vector{Event{T}}
    statistics::S
    dyads::Vector{Tuple{Int,Int}}
    p::Int
    n_int::Int
    case_idx::Vector{Int}          # 0 on the right-censored tail
    waiting::Vector{Float64}
    read_time::Vector{Float64}     # time at which interval m's statistics are read
    absorb::Vector{UnitRange{Int}} # events absorbed AFTER interval m is emitted
    # Efron denominator weights, sparsely: the tied dyads of interval m's block
    # (shared, hence a per-interval reference) and the weight they take. `nothing`
    # unless `ties=:efron` actually bit, which keeps the unweighted inner loop
    # bit-for-bit the same.
    tw_dyads::Union{Nothing, Vector{Vector{Int}}}
    tw_val::Union{Nothing, Vector{Float64}}
end

_n_risk_dyads(plan::_RiskSetPlan) = length(plan.dyads)

# Bytes one design matrix costs, and the projected cost of caching all of them.
_design_bytes(plan::_RiskSetPlan) = _n_risk_dyads(plan) * plan.p * sizeof(Float64)
_full_cache_bytes(plan::_RiskSetPlan) = _design_bytes(plan) * plan.n_int

_risk_set_plan(events::Vector{Event{T}}, statistics::AbstractVector, n_actors::Int;
               kwargs...) where T =
    _risk_set_plan(events, Tuple(statistics), n_actors; kwargs...)

function _risk_set_plan(events::Vector{Event{T}}, statistics::Tuple, n_actors::Int;
                        t0::T=zero(T), t_end::Union{Nothing,T}=nothing,
                        ties::Symbol=:ordered) where T
    n_actors >= 2 || throw(ArgumentError("need at least two actors"))
    isfinite(t0) || throw(ArgumentError("t0 must be finite"))
    t_end === nothing || isfinite(t_end) || throw(ArgumentError("t_end must be finite"))
    sorted = sort(events, by=e -> e.time)
    blocks = _tie_blocks(sorted)
    freeze = ties in (:breslow, :efron, :batch)

    p = length(statistics)
    dyads = [(s, r) for s in 1:n_actors for r in 1:n_actors if s != r]
    dyad_index = Dict(dy => d for (d, dy) in enumerate(dyads))

    if !isempty(sorted) && t0 > sorted[1].time
        throw(ArgumentError("t0 = $t0 is after the first event time " *
                            "$(sorted[1].time); the observation onset must " *
                            "precede all events"))
    end

    if t_end !== nothing && !isempty(sorted) && t_end < sorted[end].time
        throw(ArgumentError("t_end = $t_end is before the last event time " *
                            "$(sorted[end].time); the observation window must " *
                            "contain every event"))
    end
    n_int = length(sorted) + (t_end === nothing ? 0 : 1)

    case_idx = Vector{Int}(undef, n_int)
    waiting = Vector{Float64}(undef, n_int)
    read_time = Vector{Float64}(undef, n_int)
    absorb = fill(1:0, n_int)
    weighted = ties === :efron && any(b -> length(b) > 1, blocks)
    tw_dyads = weighted ? Vector{Vector{Int}}(undef, n_int) : nothing
    tw_val = weighted ? Vector{Float64}(undef, n_int) : nothing
    t_prev = float(t0)

    for block in blocks
        d = length(block)
        # Efron re-weights each tied CASE's contribution to ONE risk-set
        # denominator; a dyad that is its own competitor has no such weight
        # (and the naive one can go negative). Refuse rather than invent.
        tied_dyads = [(sorted[k].sender, sorted[k].receiver) for k in block]
        if ties === :efron && d > 1 && !allunique(tied_dyads)
            throw(ArgumentError(
                "ties=:efron requires the events tied at one timestamp to be " *
                "distinct dyads, but a dyad acts twice at t = " *
                "$(sorted[first(block)].time). Efron's correction re-weights each " *
                "tied case's contribution to ONE risk-set denominator, and a " *
                "risk-set member that is its own competitor has no such weight. " *
                "Use `ties=:breslow` (whose denominator is the plain risk-set " *
                "sum) or `ties=:ordered`."))
        end

        # Under a correction the whole block is read off the history as it stands
        # BEFORE any of the tied events, at the block's (single) timestamp; the
        # block is then absorbed as a whole, so no simultaneous event can enter
        # another's statistics.
        tb = float(sorted[first(block)].time)
        shared = freeze && d > 1
        tied_idx = weighted && d > 1 ?
            [dyad_index[dy] for dy in tied_dyads] : Int[]

        for (j, m) in enumerate(block)
            ev = sorted[m]
            ci = get(dyad_index, (ev.sender, ev.receiver), 0)
            ci > 0 || throw(ArgumentError("event $(m) references actors outside 1:$n_actors"))
            case_idx[m] = ci
            waiting[m] = float(ev.time - t_prev)
            t_prev = float(ev.time)
            read_time[m] = shared ? tb : float(ev.time)
            # Absorb the event now unless a tie correction is in force, in which
            # case the whole block is absorbed after its last interval.
            absorb[m] = freeze ? (m == last(block) ? block : (1:0)) : (m:m)
            if weighted
                tw_dyads[m] = tied_idx
                tw_val[m] = d > 1 ? 1.0 - (j - 1) / d : 1.0
            end
        end
    end

    # The right-censored tail: exposure from the last event to the end of
    # observation, with no event term (case_idx == 0).
    if t_end !== nothing
        m = n_int
        case_idx[m] = 0
        waiting[m] = float(t_end - t_prev)
        read_time[m] = float(t_end)
        if weighted
            tw_dyads[m] = Int[]
            tw_val[m] = 1.0
        end
    end

    return _RiskSetPlan(sorted, statistics, dyads, p, n_int, case_idx, waiting,
                        read_time, absorb, tw_dyads, tw_val)
end

# Interval m's design matrix, into `dest`, against `history` as it stands before
# the interval. A deterministic function of (history, read time): this is why the
# cached and streamed policies agree bit-for-bit.
function _fill_design!(dest::AbstractMatrix{Float64}, plan::_RiskSetPlan, history,
                       m::Int)
    t = plan.read_time[m]
    stats = plan.statistics
    @inbounds for (dd, (s, r)) in enumerate(plan.dyads)
        vals = map(stat -> compute(stat, history, s, r, t), stats)
        for k in 1:plan.p
            dest[dd, k] = vals[k]
        end
    end
    return dest
end

_absorb_interval!(history, plan::_RiskSetPlan, m::Int) =
    (for k in plan.absorb[m]; update_history!(history, plan.sorted[k]); end; history)

# The Efron denominator weights of interval m, materialized into the reusable
# buffer `buf` — or `nothing` when no weight bites, which is every policy but a
# biting `:efron` and keeps the unweighted inner loop unchanged.
function _tie_weights!(buf::Vector{Float64}, plan::_RiskSetPlan, m::Int)
    plan.tw_dyads === nothing && return nothing
    fill!(buf, 1.0)
    w = plan.tw_val[m]
    @inbounds for d in plan.tw_dyads[m]
        buf[d] = w
    end
    return buf
end

# -----------------------------------------------------------------------------
# The risk sets, under the three cache policies
# -----------------------------------------------------------------------------

abstract type _RiskSets end

# `cache=:all` — every design matrix materialized once. Fastest; O(E · n² · p).
struct _CachedRiskSets{T,S} <: _RiskSets
    plan::_RiskSetPlan{T,S}
    X::Vector{Matrix{Float64}}
    tw_buf::Vector{Float64}
end

# `cache=:chunked` / `:none` — a bounded cache of `length(buffers)` design
# matrices, refilled by replaying the interaction history from the start on every
# pass. `:none` is `chunk = 1`. Memory O(chunk · n² · p); the price is that the
# statistics are recomputed once per derivative evaluation instead of once.
struct _StreamedRiskSets{T,S} <: _RiskSets
    plan::_RiskSetPlan{T,S}
    buffers::Vector{Matrix{Float64}}
    history::InteractionHistory{T}
    tw_buf::Vector{Float64}
end

# Visit the intervals in time order, calling `f(m, Xm, twm)` on each: `Xm` is the
# risk-set design matrix and `twm` the Efron denominator weights (or `nothing`).
# The order is the same under every policy, so any accumulation over `f` is
# summed in the same order and lands on the same floating-point number.
function _each_interval(f::F, rs::_CachedRiskSets) where F
    plan = rs.plan
    for m in 1:plan.n_int
        f(m, rs.X[m], _tie_weights!(rs.tw_buf, plan, m))
    end
    return nothing
end

function _each_interval(f::F, rs::_StreamedRiskSets) where F
    plan = rs.plan
    history = _reset_history!(rs.history)
    k = length(rs.buffers)
    m = 1
    while m <= plan.n_int
        hi = min(m + k - 1, plan.n_int)
        # Fill the chunk, walking the history forward as we go...
        for j in m:hi
            _fill_design!(rs.buffers[j - m + 1], plan, history, j)
            _absorb_interval!(history, plan, j)
        end
        # ...then consume it. (Tied intervals under a freeze policy share one
        # matrix in `:all`; here they are recomputed, which costs a little and
        # gives the identical values — the history is frozen and the read time is
        # the block's.)
        for j in m:hi
            f(j, rs.buffers[j - m + 1], _tie_weights!(rs.tw_buf, plan, j))
        end
        m = hi + 1
    end
    return nothing
end

# Resolve `cache=:auto` to a concrete policy and a concrete cache size, in design
# matrices. `chunk` defaults to as many as fit in `cache_bytes`; a chunk that
# covers every interval IS `:all` (same memory, but the streamed path would
# recompute every statistic on every pass for nothing), so it collapses to it.
# Returns `(mode, k)` with `k` the number of matrices alive at once, so the
# footprint a fit will pay is `k * _design_bytes(plan)` — computable without
# allocating any of it.
function _resolve_cache(plan::_RiskSetPlan; cache::Symbol=:auto,
                        chunk::Union{Nothing,Int}=nothing,
                        cache_bytes::Int=_DEFAULT_CACHE_BYTES)
    cache in _CACHE_MODES || throw(ArgumentError(
        "cache must be one of $(_CACHE_MODES), got :$cache. `:all` materializes " *
        "every risk-set design matrix (fastest, O(E·n²·p) memory), `:chunked` " *
        "keeps a bounded cache of `chunk` of them and recomputes the rest on " *
        "each pass, `:none` keeps exactly one, and `:auto` picks `:all` when " *
        "the projected footprint fits in `cache_bytes`."))
    chunk === nothing || chunk >= 1 ||
        throw(ArgumentError("chunk must be at least 1, got $chunk"))

    mode = cache
    mode === :auto &&
        (mode = _full_cache_bytes(plan) <= cache_bytes ? :all : :chunked)
    mode === :none && return (:none, 1)
    if mode === :chunked
        k = something(chunk, cache_bytes ÷ max(1, _design_bytes(plan)))
        k >= plan.n_int && return (:all, plan.n_int)
        return (:chunked, max(1, k))
    end
    return (:all, plan.n_int)
end

function _risk_sets(plan::_RiskSetPlan{T,S}; cache::Symbol=:auto,
                    chunk::Union{Nothing,Int}=nothing,
                    cache_bytes::Int=_DEFAULT_CACHE_BYTES) where {T,S}
    mode, k = _resolve_cache(plan; cache=cache, chunk=chunk,
                             cache_bytes=cache_bytes)
    D = _n_risk_dyads(plan)
    tw_buf = plan.tw_dyads === nothing ? Float64[] : Vector{Float64}(undef, D)

    if mode === :all
        history = InteractionHistory{T}()
        X = Vector{Matrix{Float64}}(undef, plan.n_int)
        m = 1
        while m <= plan.n_int
            Xm = Matrix{Float64}(undef, D, plan.p)
            _fill_design!(Xm, plan, history, m)
            X[m] = Xm
            # A frozen tie block reads ONE matrix off ONE history at ONE time:
            # share it rather than recomputing (and re-storing) it per event.
            hi = m
            while hi < plan.n_int && isempty(plan.absorb[hi])
                hi += 1
                X[hi] = Xm
            end
            for j in m:hi
                _absorb_interval!(history, plan, j)
            end
            m = hi + 1
        end
        return _CachedRiskSets(plan, X, tw_buf)
    end

    buffers = [Matrix{Float64}(undef, D, plan.p) for _ in 1:k]
    return _StreamedRiskSets(plan, buffers, InteractionHistory{T}(), tw_buf)
end

# The eager 5-tuple `(dyads, case_idx, X, waiting, W)` with every design matrix
# and every dense Efron weight vector materialized: `cache=:all` in one call, the
# clearest statement of what the risk sets ARE (the tests read it directly).
_risk_set_stats(events::Vector{Event{T}}, statistics::AbstractVector, n_actors::Int;
                kwargs...) where T =
    _risk_set_stats(events, Tuple(statistics), n_actors; kwargs...)

function _risk_set_stats(events::Vector{Event{T}}, statistics::Tuple, n_actors::Int;
                         kwargs...) where T
    plan = _risk_set_plan(events, statistics, n_actors; kwargs...)
    rs = _risk_sets(plan; cache=:all)
    W = plan.tw_dyads === nothing ? nothing :
        [copy(_tie_weights!(Vector{Float64}(undef, _n_risk_dyads(plan)), plan, m))
         for m in 1:plan.n_int]
    return plan.dyads, plan.case_idx, rs.X, plan.waiting, W
end
