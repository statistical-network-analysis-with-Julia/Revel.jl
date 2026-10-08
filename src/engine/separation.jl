# =============================================================================
# Separation on the full risk set: the ecosystem's shared verdict
# =============================================================================
#
# Whether a finite maximum exists is a property of the risk sets, not of where
# Newton stopped, so it is decided from the design alone: the margin rows of a
# direction of recession are built here and handed to the ONE criterion,
# `NetworkCore.separation_from_margins`, which decides it by a linear programme
# and certifies the direction in exact arithmetic.
#
# - Ordinal: a conditional logit with one stratum per interval. Each case must
#   score at least as high as every alternative along d: rows x_case − x_j.
#   (The Efron denominator weights are positive, so every dyad stays in the
#   choice set.)
# - Timing: in joint coordinates z = (1, x) for β = (log λ₀, θ), an interval of
#   positive length is Poisson exposure, so every dyad needs −z_j'd ≥ 0 (a rate
#   may only fall); every event needs z_case'd ≥ 0 (its linear term may only
#   rise). Together these make an event with exposure an equality. An event in
#   a zero-length interval (a tie) is required to rise on its own, rather than
#   only in the sum over such events: that can miss a direction along which
#   those events trade off, never invent one.
#
# Memory. The full risk set repeats the same rows many times (every dyad with
# no history has the same statistics), so the rows are deduplicated exactly as
# they stream past, and only the distinct ones are kept. When even those exceed
# the budget (`cache_bytes`, the same bound the design matrices respect), the
# verdict is reached by row generation instead: decide on the rows kept, check
# the direction found against every row as the risk sets stream past again, add
# the rows it violates (or cannot sign for certain), and repeat. A direction
# that no row contradicts is a direction of recession of the whole design; a
# design whose kept rows admit none and span every coefficient admits none
# either. Both routes give the same verdict (tested).

# Distinct non-zero margin rows, in order of first appearance, deduplicated
# through a hash table of row hashes (collisions chained and compared exactly).
mutable struct _MarginRows
    q::Int
    heads::Dict{UInt64,Int}
    next::Vector{Int}
    data::Vector{Float64}           # row-major, q entries per row
end
_MarginRows(q::Int) = _MarginRows(q, Dict{UInt64,Int}(), Int[], Float64[])

Base.length(rows::_MarginRows) = length(rows.next)

function _row_equals(rows::_MarginRows, k::Int, buf::Vector{Float64})
    o = (k - 1) * rows.q
    @inbounds for j in 1:rows.q
        rows.data[o + j] === buf[j] || return false
    end
    return true
end

# Add `buf` unless it is already there; return whether it was added.
function _push_margin!(rows::_MarginRows, buf::Vector{Float64})
    h = hash(buf)
    k = get(rows.heads, h, 0)
    first = k
    while k != 0
        _row_equals(rows, k, buf) && return false
        k = rows.next[k]
    end
    append!(rows.data, buf)
    push!(rows.next, first)
    rows.heads[h] = length(rows.next)
    return true
end

# Whether `buf` is one of the rows kept
function _row_index(rows::_MarginRows, buf::Vector{Float64})
    k = get(rows.heads, hash(buf), 0)
    while k != 0
        _row_equals(rows, k, buf) && return true
        k = rows.next[k]
    end
    return false
end

_margin_matrix(rows::_MarginRows) =
    Matrix{Float64}(transpose(reshape(rows.data, rows.q, :)))

# Call `f(m, buf)` on every non-zero margin row of every interval; `buf` is
# reused. Non-finite statistics make the likelihood undefined.
function _each_margin(f::F, rs::_RiskSets; timing::Bool) where F
    plan = rs.plan
    p = plan.p
    buf = zeros(timing ? p + 1 : p)
    function emit(m)
        allzero = true
        @inbounds for k in eachindex(buf)
            isfinite(buf[k]) || throw(ArgumentError(
                "the risk-set design has a non-finite statistic value; the " *
                "likelihood is undefined"))
            buf[k] += 0.0                   # −0.0 and 0.0 are one value here
            allzero &= iszero(buf[k])
        end
        allzero || f(m, buf)
        return nothing
    end
    _each_interval(rs) do m, X, _
        ci = plan.case_idx[m]
        if timing
            if ci > 0
                buf[1] = 1.0
                @inbounds for k in 1:p; buf[k + 1] = X[ci, k]; end
                emit(m)
            end
            plan.waiting[m] > 0 || return
            @inbounds for j in axes(X, 1)
                buf[1] = -1.0
                for k in 1:p; buf[k + 1] = -X[j, k]; end
                emit(m)
            end
        else
            ci > 0 || return
            @inbounds for j in axes(X, 1)
                j == ci && continue
                for k in 1:p; buf[k] = X[ci, k] - X[j, k]; end
                emit(m)
            end
        end
    end
    return nothing
end

# A row's value along `d`, and the scale of its terms
function _along(buf::Vector{Float64}, d::Vector{Float64})
    s = sc = 0.0
    @inbounds for k in eachindex(buf)
        t = buf[k] * d[k]
        s += t
        sc += abs(t)
    end
    return s, sc
end

# Rows whose sign along `d` is not certainly non-negative: negative, or too
# close to zero for floating point to sign (a row whose every term is zero is
# exactly zero, and fine). Floating-point error in `s` is far below 1e-9·sc.
_unsettled(s, sc) = sc > 0 && s <= 1e-9 * sc

# The intervals a separating direction predicts perfectly: those with a margin
# strictly positive along `d` (the case beats an alternative; an exposure rate
# runs to zero). This feeds the warning only.
function _separated_intervals(rs::_RiskSets, d::Vector{Float64}; timing::Bool)
    units = Int[]
    _each_margin(rs; timing) do m, buf
        (isempty(units) || units[end] != m) || return
        s, sc = _along(buf, d)
        s > 1e-9 * sc && push!(units, m)
    end
    return units
end

# Row generation, for a design whose distinct rows exceed the budget: `rows`
# holds the first ones; at most `max_rows` more are added per round.
function _separation_by_rows(rs::_RiskSets, rows::_MarginRows, family::Symbol;
                             timing::Bool, max_rows::Int)
    while true
        R = _margin_matrix(rows)
        v = separation_from_margins(R; family=family)
        added = 0
        if v.separated
            d = v.direction
            _each_margin(rs; timing) do _, buf
                added < max_rows || return
                s, sc = _along(buf, d)
                _unsettled(s, sc) && _push_margin!(rows, buf) && (added += 1)
            end
            added == 0 && return v
        else
            # Not separated on these rows. If they span every coefficient, no
            # other row can make a direction strictly positive; otherwise add
            # the rows with a component in their null space.
            N = nullspace(R)
            size(N, 2) == 0 && return v
            _each_margin(rs; timing) do _, buf
                added < max_rows || return
                proj = transpose(N) * buf
                maximum(abs, proj) > 1e-9 * norm(buf) && _push_margin!(rows, buf) &&
                    (added += 1)
            end
            added == 0 && return v
        end
    end
end

# The shared verdict for the risk sets of an ordinal fit (`family == :clogit`)
# or an interval-timing fit (`family == :poisson`: the exponential-baseline
# likelihood is a Poisson log-linear likelihood with offset log Δt). `terms`
# index the coefficients in `coef` order, the log baseline first for timing;
# `units` are the intervals predicted perfectly. `budget` bounds the bytes of
# distinct margin rows kept at once.
function _separation_verdict(rs::_RiskSets; timing::Bool=false,
                             budget::Int=_DEFAULT_CACHE_BYTES)
    q = timing ? rs.plan.p + 1 : rs.plan.p
    family = timing ? :poisson : :clogit
    # a kept row costs its q values plus about 24 bytes of index
    max_rows = max(4q, budget ÷ (8q + 24))
    rows = _MarginRows(q)
    complete = Ref(true)
    _each_margin(rs; timing) do _, buf
        complete[] || return
        length(rows) < max_rows ? _push_margin!(rows, buf) :
            (_row_index(rows, buf) || (complete[] = false))
    end
    v = complete[] ? separation_from_margins(_margin_matrix(rows); family=family) :
                     _separation_by_rows(rs, rows, family; timing, max_rows)
    # SeparationVerdict(separated, terms, direction, units, certified, family)
    v.separated || return SeparationVerdict(false, Int[], Float64[], Int[], false, family)
    units = _separated_intervals(rs, v.direction; timing)
    return SeparationVerdict(true, v.terms, v.direction, units, v.certified, family)
end

# Run `f()` with its log messages discarded: on a separated design the
# optimizer's warnings about the singular information or the iteration limit
# it stops at are symptoms, and the separation warning replaces them.
_quietly(f, quiet::Bool) = quiet ? with_logger(f, NullLogger()) : f()
