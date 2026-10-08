using Test, Revel, REM, NetworkCore, Random, LinearAlgebra, Statistics, DataFrames
import Distributions
using Aqua

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# A random directed sequence on strictly increasing times, with types and weights
function random_events(rng, n, m; types=(:event,), weights=false)
    events = Event{Float64}[]
    t = 0.0
    for _ in 1:m
        s = rand(rng, 1:n)
        r = rand(rng, 1:(n - 1)); r >= s && (r += 1)
        t += 0.25 + rand(rng)
        push!(events, Event(s, r, t; eventtype=rand(rng, types),
                            weight=weights ? 0.5 + 2rand(rng) : 1.0))
    end
    return events
end

# Brute-force reference for a layer's dyad weight: the definition, written out
function ref_weight(past, i, j, t, memory; types=nothing, weighted=false, clock=:time,
                    keep=nothing)
    total = 0.0
    for (k, e) in enumerate(past)
        (e.sender == i && e.receiver == j) || continue
        types === nothing || e.eventtype in types || continue
        keep === nothing || keep(e.sender, e.receiver, e.time, e.weight, e.eventtype) || continue
        age = clock === :order ? Float64(length(past) + 1 - k) : t - e.time
        total += kernel_weight(memory, age) * (weighted ? e.weight : 1.0)
    end
    return total
end

dyads_of(n) = [(s, r) for s in 1:n for r in 1:n if s != r]

# Run a documentation block statement by statement in `mod`, and compare every
# statement whose line ends in a `# value` comment with that value. A comment is
# a claim when its text — after an optional leading `≈`, cut at ` — `, `; ` or
# `  ` and at the last ` = ` — parses as a literal: numbers, strings, symbols,
# `true`/`false`, `nothing`, `NaN`/`Inf`, vectors and tuples of those, and
# `log`/`sqrt`/`exp` or arithmetic of numbers. Anything else is prose. A number
# written with d decimals matches to half a unit of its last digit; `≈` allows
# 1 %. Returns (claims checked, mismatch descriptions); throws on an error.
const _LITERAL_CALLS = (:+, :-, :*, :/, :^, :log, :sqrt, :exp, :log1p, :log2)
_is_literal(x) = x isa Union{Number, String, QuoteNode, Bool, Nothing} ||
    x in (:NaN, :Inf, :nothing, :true, :false) ||
    (x isa Expr && (x.head in (:vect, :tuple) && all(_is_literal, x.args) ||
                    x.head === :call && x.args[1] in _LITERAL_CALLS &&
                        all(a -> _is_literal(a) && !(a isa String), x.args[2:end])))
function _claim(comment::AbstractString)
    text = strip(comment)
    approx = startswith(text, "≈")
    approx && (text = strip(text[nextind(text, 1):end]))
    for sep in (" — ", " – ", "; ", "  ", " (")
        text = first(split(text, sep))
    end
    occursin(" = ", text) && (text = strip(last(split(text, " = "))))
    isempty(text) && return nothing
    ex = try Meta.parse(text; raise=true) catch; return nothing end
    (ex isa Expr && ex.head === :incomplete) && return nothing
    _is_literal(ex) || return nothing
    digits = maximum((length(m.captures[1]) for m in eachmatch(r"\d\.(\d+)", text)); init=0)
    return (value=Core.eval(Main, ex), approx=approx, digits=digits, text=text)
end
_matches(v, c) = false
_matches(v::Number, c::Number, approx, digits) =
    (isnan(c) && isnan(v)) || v == c ||
    (approx ? isapprox(v, c; rtol=0.01, atol=1e-12) :
     digits > 0 && abs(v - c) <= 0.5 * 10.0^(-digits) * (1 + 1e-9))
_matches(v::AbstractString, c::AbstractString, _, _) = v == c
_matches(v::Symbol, c::Symbol, _, _) = v == c
_matches(v::Bool, c::Bool, _, _) = v == c
_matches(v::Nothing, c::Nothing, _, _) = true
_matches(v::Union{AbstractVector,Tuple}, c::Union{AbstractVector,Tuple}, a, d) =
    length(v) == length(c) && all(_matches(x, y, a, d) for (x, y) in zip(v, c))
_matches(v, c, a, d) = false
function check_block(mod::Module, block::AbstractString; where="")
    checked = 0; bad = String[]
    pos = 1
    while pos <= lastindex(block)
        ex, next = Meta.parse(block, pos; raise=true)
        src = block[pos:prevind(block, next)]
        pos = next
        ex === nothing && continue
        val = Core.eval(mod, ex)
        line = rstrip(last(split(rstrip(src), "\n")))
        # the comment of the statement's last line (not a `#` inside a string)
        m = match(r"^(?:[^\"#]|\"(?:[^\"\\]|\\.)*\")*#(.*)$", line)
        m === nothing && continue
        startswith(strip(line), "#") && continue
        c = _claim(m.captures[1])
        c === nothing && continue
        checked += 1
        _matches(val, c.value, c.approx, c.digits) ||
            push!(bad, "$where: `$(strip(split(line, "#")[1]))` is $(repr(val)), the comment says $(c.text)")
    end
    return checked, bad
end

const MEMORIES = (FullMemory(), HalfLife(3.0), HalfLife(3.0; normalized=true),
                  Window(4.0), IntervalMemory(1.0, 6.0), PowerLaw(0.7; offset=0.5),
                  LinearDecay(8.0), KernelMemory(a -> 1 / (1 + a)^2))

# ----------------------------------------------------------------------------
# Memory kernels
# ----------------------------------------------------------------------------

@testset "Memory kernels" begin
    @test kernel_weight(FullMemory(), 1e9) == 1.0
    @test kernel_weight(HalfLife(5.0), 5.0) ≈ 0.5
    @test kernel_weight(HalfLife(5.0), 10.0) ≈ 0.25
    @test kernel_weight(HalfLife(5.0; normalized=true), 0.0) ≈ log(2) / 5
    @test kernel_weight(HalfLife(Inf), 1e6) == 1.0
    # A window is closed at its width; an interval is open below and closed above,
    # so adjacent intervals partition the past without double counting
    @test kernel_weight(Window(3.0), 3.0) == 1.0
    @test kernel_weight(Window(3.0), 3.0 + 1e-9) == 0.0
    @test kernel_weight(IntervalMemory(1.0, 7.0), 1.0) == 0.0
    @test kernel_weight(IntervalMemory(1.0, 7.0), 7.0) == 1.0
    for age in (0.0, 0.5, 1.0, 3.0, 7.0, 7.5)
        parts = interval_partition([0.0, 1.0, 7.0])
        @test sum(kernel_weight(m, age) for m in parts) ==
              (0 < age <= 7 ? 1.0 : 0.0)
    end
    @test kernel_weight(PowerLaw(2.0), 4.0) ≈ 1 / 16
    @test kernel_weight(PowerLaw(1.0; offset=1.0), 0.0) == 1.0
    @test_throws ArgumentError kernel_weight(PowerLaw(1.0), 0.0)
    @test kernel_weight(LinearDecay(10.0), 2.5) ≈ 0.75
    @test kernel_weight(LinearDecay(10.0), 11.0) == 0.0
    @test kernel_weight(KernelMemory(a -> 2a), 3.0) == 6.0

    @test_throws ArgumentError HalfLife(0.0)
    @test_throws ArgumentError HalfLife(Inf; normalized=true)
    @test_throws ArgumentError Window(-1.0)
    @test_throws ArgumentError IntervalMemory(2.0, 2.0)
    @test_throws ArgumentError IntervalMemory(-1.0, 2.0)
    @test_throws ArgumentError PowerLaw(0.0)
    @test_throws ArgumentError PowerLaw(1.0; offset=-1.0)
    @test_throws ArgumentError LinearDecay(0.0)
    @test_throws ArgumentError KernelMemory(identity; support=0.0)
    @test_throws ArgumentError interval_partition([1.0])
    @test_throws ArgumentError interval_partition([0.0, 2.0, 2.0])
    @test length(interval_partition([0, 1, 24, 168])) == 3
end

# ----------------------------------------------------------------------------
# Hand-computed effects
# ----------------------------------------------------------------------------

@testset "Dyad effects (hand-computed)" begin
    h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 3.0),
                       Event(2, 1, 3.5)])
    @test compute(Inertia(), h, 1, 2, 4.0) == 2.0
    @test compute(Inertia(), h, 2, 1, 4.0) == 1.0
    @test compute(Inertia(), h, 3, 1, 4.0) == 0.0
    @test compute(Reciprocation(), h, 1, 2, 4.0) == 1.0
    @test compute(Reciprocation(), h, 2, 1, 4.0) == 2.0
    @test compute(DyadActivity(), h, 1, 2, 4.0) == 3.0
    @test compute(DyadActivity(), h, 2, 1, 4.0) == 3.0

    # Butts's persistence: 2 of actor 1's 3 sends went to 2
    @test compute(Inertia(scaling=:prop), h, 1, 2, 4.0) ≈ 2 / 3
    # a sender with no history takes `empty`
    @test compute(Inertia(scaling=:prop), h, 3, 1, 4.0) == 0.0
    @test compute(Inertia(scaling=:prop, empty=0.25), h, 3, 1, 4.0) == 0.25
    # reciprocity as a share of the SENDER's receipts (relevent FrRecSnd):
    # actor 1 received once, from 2
    @test compute(Reciprocation(scaling=:prop), h, 1, 2, 4.0) == 1.0
    @test compute(Reciprocation(scaling=:prop), h, 1, 3, 4.0) == 0.0
    # ... or of the RECEIVER's sends (Kitts et al.'s embedding reciprocation):
    # actor 2 sent once, to 1
    @test compute(Reciprocation(scaling=:prop, denominator=:receiver_out), h, 1, 2, 4.0) == 1.0
    # actor 1 sent 3 times, twice to 2: candidate 2 → 1 reads w(1,2)/outdeg(1)
    @test compute(Reciprocation(scaling=:prop, denominator=:receiver_out), h, 2, 1, 4.0) ≈ 2 / 3

    @test compute(Inertia(transform=:indicator), h, 1, 2, 4.0) == 1.0
    @test compute(Inertia(transform=:log1p), h, 1, 2, 4.0) ≈ log(3)
    @test compute(Inertia(transform=sqrt), h, 1, 2, 4.0) ≈ sqrt(2)

    # half-life 1: the two 1 → 2 events are 3 and 1 time units old at t = 4
    @test compute(Inertia(memory=HalfLife(1.0)), h, 1, 2, 4.0) ≈ 0.5^3 + 0.5
    @test compute(Inertia(memory=Window(1.0)), h, 1, 2, 4.0) == 1.0
    @test compute(Inertia(memory=IntervalMemory(1.0, 3.0)), h, 1, 2, 4.0) == 1.0   # only t = 1
    # event clock: the 1 → 2 events are the 1st and 3rd of 4, i.e. 4 and 2 events old
    @test compute(Inertia(memory=HalfLife(2.0), clock=:order), h, 1, 2, 4.0) ≈ 0.25 + 0.5

    # the evaluation time need not have the history's own number type
    @test compute(Inertia(), h, 1, 2, 4) == 2.0
    @test compute(Interaction(Inertia(), PShift(:AB_BA)), h, 1, 2, 4) == 2.0
    @test compute(Transformed(Inertia(), log1p), h, 1, 2, 4) ≈ log(3)

    @test name(Inertia()) == "inertia"
    @test name(Inertia(scaling=:prop)) == "inertia.prop"
    @test name(Inertia(memory=HalfLife(2.0))) == "inertia[halflife=2.0]"
    @test name(Inertia(transform=:log1p)) == "log1p(inertia)"
    @test name(Reciprocation(types=:a, weighted=true)) == "reciprocity[types=a,weighted]"
    @test name(Inertia(name="mine")) == "mine"

    @test_throws ArgumentError Inertia(scaling=:std)
    @test_throws ArgumentError Inertia(transform=:cube)
    @test_throws ArgumentError Inertia(bogus=1)
    @test_throws ArgumentError Inertia(layer=EventLayer(), memory=Window(1.0))
    @test_throws ArgumentError Inertia(layer=:not_a_layer)
    @test_throws ArgumentError DyadEffect(EventLayer(); direction=:up)
    @test_throws ArgumentError DyadEffect(EventLayer(); denominator=:nobody)
    @test_throws ArgumentError EventLayer(clock=:wall)
end

@testset "Event layers: types, weights, filters, symmetry" begin
    events = [Event(1, 2, 1.0; eventtype=:praise, weight=2.0),
              Event(1, 2, 2.0; eventtype=:blame, weight=5.0),
              Event(2, 1, 3.0; eventtype=:praise, weight=1.5)]
    h = build_history(events)
    @test compute(Inertia(types=:praise), h, 1, 2, 4.0) == 1.0
    @test compute(Inertia(types=[:praise, :blame]), h, 1, 2, 4.0) == 2.0
    @test compute(Inertia(weighted=true), h, 1, 2, 4.0) == 7.0
    @test compute(Inertia(types=:blame, weighted=true), h, 1, 2, 4.0) == 5.0
    # the attribute filter: only events with a heavy weight count
    heavy = Inertia(keep=(s, r, t, w, ty) -> w > 3, name="heavy")
    @test compute(heavy, h, 1, 2, 4.0) == 1.0
    # symmetric layers record an event for both of its participants
    @test compute(Inertia(symmetric=true), h, 1, 2, 4.0) == 3.0
    @test compute(Inertia(symmetric=true), h, 2, 1, 4.0) == 3.0
    @test compute(OutdegreeSender(symmetric=true), h, 2, 1, 4.0) == 3.0
    # ... so an actor's out-, in- and total degree are all the events it took
    # part in (once, not twice), and a share is taken of the events
    @test compute(IndegreeSender(symmetric=true), h, 2, 1, 4.0) == 3.0
    @test compute(TotaldegreeSender(symmetric=true), h, 2, 1, 4.0) == 3.0
    @test compute(TotaldegreeSender(symmetric=true, scaling=:prop), h, 2, 1, 4.0) == 1.0
    @test compute(DegreeMin(symmetric=true), h, 1, 3, 4.0) == 0.0
    @test compute(TotaldegreeDyad(symmetric=true), h, 1, 2, 4.0) == 6.0

    # one shared layer, one index of the history
    L = EventLayer(memory=HalfLife(2.0))
    a, b = Inertia(layer=L), Reciprocation(layer=L)
    @test a.layer === b.layer
    @test compute(a, h, 1, 2, 4.0) ≈ 0.5^1.5 + 0.5
    @test compute(b, h, 1, 2, 4.0) ≈ 0.5^0.5
    @test length(L.states) == 1

    # split_by_type is the type-split interaction
    effects = split_by_type(Reciprocation, [:praise, :blame])
    @test name.(effects) == ["reciprocity[types=praise]", "reciprocity[types=blame]"]
    @test [compute(e, h, 1, 2, 4.0) for e in effects] == [1.0, 0.0]

    # The null actor of relevent (receiver 0, "to the group") has no dyad
    g = build_history([Event(1, 0, 1.0), Event(1, 2, 2.0)])
    @test compute(OutdegreeSender(), g, 1, 2, 3.0) == 1.0
    # an actor beyond everything seen so far simply has no history
    @test compute(Inertia(), g, 9, 10, 3.0) == 0.0
    @test compute(OTP(), g, 9, 10, 3.0) == 0.0

    @test occursin("halflife=2.0", sprint(show, L))
    @test sprint(show, EventLayer()) == "EventLayer(full memory, all events)"
end

@testset "Layer sync: appended, replayed and separate histories" begin
    rng = Xoshiro(3)
    events = random_events(rng, 5, 40)
    stat = Inertia(memory=HalfLife(2.0))
    win = OutdegreeSender(memory=Window(3.0))
    h = InteractionHistory()
    for (m, e) in enumerate(events)
        for (s, r) in dyads_of(5)
            @test compute(stat, h, s, r, e.time) ≈
                  ref_weight(events[1:(m - 1)], s, r, e.time, HalfLife(2.0)) atol = 1e-12
        end
        update_history!(h, e)
    end
    # A second, shorter history through the SAME statistic does not see the first
    h2 = build_history(events[1:5])
    t = events[6].time
    @test compute(stat, h2, events[1].sender, events[1].receiver, t) ≈
          ref_weight(events[1:5], events[1].sender, events[1].receiver, t, HalfLife(2.0))
    # ... and the first is still intact afterwards
    t_end = events[end].time + 1
    @test compute(stat, h, 1, 2, t_end) ≈ ref_weight(events, 1, 2, t_end, HalfLife(2.0))
    # Replaying a history in place (what the streamed risk sets of the exact fitters do)
    # rebuilds the layer rather than double counting
    # (only the documented `events` field: that is all a layer reads)
    empty!(h.events)
    append!(h.events, events[1:10])
    t10 = events[11].time
    @test compute(stat, h, 1, 2, t10) ≈ ref_weight(events[1:10], 1, 2, t10, HalfLife(2.0))
    @test compute(win, h, 1, 2, t10) ==
          sum(ref_weight(events[1:10], 1, j, t10, Window(3.0)) for j in 2:5)
    # a snapshot kernel can be read at an earlier clock again
    @test compute(win, h, 1, 2, events[5].time) ==
          sum(ref_weight(events[1:10], 1, j, events[5].time, Window(3.0)) for j in 2:5)
end

@testset "Every memory kernel against the definition" begin
    rng = Xoshiro(20260930)
    n = 5
    events = random_events(rng, n, 60; types=(:a, :b), weights=true)
    for memory in MEMORIES, (types, weighted, clock) in
            ((nothing, false, :time), (:a, true, :time), (nothing, false, :order))
        clock === :order && memory isa PowerLaw && continue
        kw = (memory=memory, types=types, weighted=weighted, clock=clock)
        tt = types === nothing ? nothing : (types,)
        ref(past, i, j, t) = ref_weight(past, i, j, t, memory; types=tt,
                                        weighted=weighted, clock=clock)
        stats = (Inertia(; kw...), Reciprocation(; kw...), OutdegreeSender(; kw...),
                 IndegreeReceiver(; kw...), TotaldegreeSender(; kw...), OTP(; kw...),
                 ISP(; kw..., combine=:product))
        h = InteractionHistory()
        worst = 0.0
        for (m, e) in enumerate(events)
            past = events[1:(m - 1)]
            if m % 7 == 0       # check a few time points, all dyads
                for (s, r) in dyads_of(n)
                    out_s = sum(ref(past, s, j, e.time) for j in 1:n if j != s)
                    in_s = sum(ref(past, j, s, e.time) for j in 1:n if j != s)
                    in_r = sum(ref(past, j, r, e.time) for j in 1:n if j != r)
                    otp = sum(min(ref(past, s, k, e.time), ref(past, k, r, e.time))
                              for k in 1:n if k != s && k != r)
                    isp = sum(ref(past, k, s, e.time) * ref(past, k, r, e.time)
                              for k in 1:n if k != s && k != r)
                    expected = (ref(past, s, r, e.time), ref(past, r, s, e.time), out_s,
                                in_r, out_s + in_s, otp, isp)
                    for (stat, want) in zip(stats, expected)
                        worst = max(worst, abs(compute(stat, h, s, r, e.time) - want))
                    end
                end
            end
            update_history!(h, e)
        end
        @test worst < 1e-10
    end
end

@testset "Degree effects (hand-computed)" begin
    h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 3.0),
                       Event(2, 3, 4.0)])
    # actor 1: out 3, in 0; actor 2: out 1, in 2; actor 3: out 0, in 2
    @test compute(OutdegreeSender(), h, 1, 2, 5.0) == 3.0
    @test compute(IndegreeSender(), h, 2, 1, 5.0) == 2.0
    @test compute(TotaldegreeSender(), h, 2, 1, 5.0) == 3.0
    @test compute(OutdegreeReceiver(), h, 3, 1, 5.0) == 3.0
    @test compute(IndegreeReceiver(), h, 1, 3, 5.0) == 2.0
    @test compute(TotaldegreeReceiver(), h, 1, 2, 5.0) == 3.0
    # distinct partners: actor 1 sent to 2 and 3
    @test compute(OutdegreeSender(measure=:partners), h, 1, 2, 5.0) == 2.0
    @test compute(IndegreeReceiver(measure=:partners), h, 1, 3, 5.0) == 2.0
    @test compute(TotaldegreeSender(measure=:partners), h, 2, 1, 5.0) == 2.0
    # shares of the 4 past events (relevent's normalised degrees)
    @test compute(OutdegreeSender(scaling=:prop), h, 1, 2, 5.0) == 0.75
    @test compute(IndegreeReceiver(scaling=:prop), h, 1, 3, 5.0) == 0.5
    @test compute(TotaldegreeReceiver(scaling=:prop), h, 1, 2, 5.0) == 3 / 8
    e = build_history(Event{Float64}[])
    @test compute(OutdegreeSender(scaling=:prop, empty=1 / 3), e, 1, 2, 0.0) == 1 / 3
    # windowed partners drop out with the window
    @test compute(OutdegreeSender(measure=:partners, memory=Window(2.5)), h, 1, 2, 5.0) == 1.0

    @test compute(TotaldegreeDyad(), h, 1, 3, 5.0) == 5.0
    @test compute(DegreeMin(), h, 1, 3, 5.0) == 2.0
    @test compute(DegreeMax(), h, 1, 3, 5.0) == 3.0
    @test compute(DegreeDiff(), h, 1, 3, 5.0) == 1.0
    # out-degree(1) × in-degree(3) = 3 × 2
    @test compute(DegreeAssortativity(), h, 1, 3, 5.0) == 6.0
    @test compute(TotaldegreeDyad(scaling=:prop), h, 1, 3, 5.0) == 5 / 8

    @test name(OutdegreeSender()) == "outdegreeSender"
    @test name(TotaldegreeReceiver(scaling=:prop)) == "totaldegreeReceiver.prop"
    @test name(DegreeMin()) == "degreeMin"
    @test_throws ArgumentError OutdegreeSender(measure=:partners, scaling=:prop)
    @test_throws ArgumentError DegreeEffect(EventLayer(); role=:bystander)
    @test_throws ArgumentError DegreeEffect(EventLayer(); kind=:sideways)
    @test_throws ArgumentError DyadDegreeEffect(EventLayer(); combine=:mean)
end

@testset "Two-paths (hand-computed)" begin
    # 1 → 3 twice, 3 → 2 once; 1 → 4 once, 4 → 2 three times
    h = build_history([Event(1, 3, 1.0), Event(1, 3, 2.0), Event(3, 2, 3.0),
                       Event(1, 4, 4.0), Event(4, 2, 5.0), Event(4, 2, 6.0),
                       Event(4, 2, 7.0)])
    @test compute(OTP(), h, 1, 2, 8.0) == min(2, 1) + min(1, 3)               # 2
    @test compute(OTP(combine=:product), h, 1, 2, 8.0) == 2 * 1 + 1 * 3       # 5
    @test compute(OTP(combine=:count), h, 1, 2, 8.0) == 2.0
    @test compute(OTP(combine=:sum), h, 1, 2, 8.0) == (2 + 1) + (1 + 3)       # 7
    @test compute(OTP(combine=:max), h, 1, 2, 8.0) == 2 + 3                   # 5
    @test compute(OTP(combine=:product, root=true), h, 1, 2, 8.0) == sqrt(5)
    @test compute(OTP(), h, 2, 1, 8.0) == 0.0
    # the same two-paths read from the other end are incoming two-paths of 2 → 1
    @test compute(ITP(), h, 2, 1, 8.0) == 2.0
    @test compute(ITP(), h, 1, 2, 8.0) == 0.0
    # 3 and 4 both sent to 2 (shared target) and were both sent to by 1
    @test compute(OSP(), h, 3, 4, 8.0) == min(1, 3)
    @test compute(ISP(), h, 3, 4, 8.0) == min(2, 1)
    @test compute(OSP(), h, 1, 2, 8.0) == 0.0
    @test compute(SharedPartners(), h, 1, 2, 8.0) == 2.0
    @test compute(SharedPartners(), h, 2, 1, 8.0) == 2.0

    # `order`: through 3 the first leg (t = 1, 2) precedes the second (t = 3);
    # reversing the requirement removes both intermediaries
    @test compute(OTP(ordered=true), h, 1, 2, 8.0) == 2.0
    @test compute(OTP(order=:leg2_first), h, 1, 2, 8.0) == 0.0
    late = build_history([Event(3, 2, 1.0), Event(1, 3, 2.0)])
    @test compute(OTP(), late, 1, 2, 3.0) == 1.0
    @test compute(OTP(ordered=true), late, 1, 2, 3.0) == 0.0     # 3 → 2 came first
    # for a cycle r → k → s the path's first step is the receiver's leg
    cyc = build_history([Event(2, 3, 1.0), Event(3, 1, 2.0)])
    @test compute(ITP(ordered=true), cyc, 1, 2, 3.0) == 1.0
    @test compute(ITP(order=:leg1_first), cyc, 1, 2, 3.0) == 0.0

    # closure among same-attribute actors: only 3 is on actor 1's team
    team = [1, 2, 1, 2]
    @test compute(OTP(third=matching_third(team)), h, 1, 2, 8.0) == 1.0

    # cross-network closure: leg 1 on :a events, leg 2 on :b events
    mixed = build_history([Event(1, 3, 1.0; eventtype=:a), Event(3, 2, 2.0; eventtype=:b),
                           Event(1, 4, 3.0; eventtype=:b), Event(4, 2, 4.0; eventtype=:b)])
    A, B = EventLayer(types=:a), EventLayer(types=:b)
    @test compute(TwoPathEffect(A, B), mixed, 1, 2, 5.0) == 1.0
    @test compute(TwoPathEffect(B, B), mixed, 1, 2, 5.0) == 1.0
    @test compute(TwoPathEffect(B, A), mixed, 1, 2, 5.0) == 0.0

    # a symmetric leg reads both directions: through 3, w(1,3) + w(3,1) = 1 + 2
    # on the sender's side and w(3,2) = 1 on the receiver's
    sym = build_history([Event(1, 3, 1.0), Event(3, 1, 2.0), Event(3, 1, 2.5),
                         Event(3, 2, 3.0), Event(2, 4, 3.5)])
    L = EventLayer()
    @test compute(TwoPathEffect(L; dir1=:sym), sym, 1, 2, 4.0) == 1.0         # min(3, 1)
    @test compute(TwoPathEffect(L; dir1=:sym, combine=:sum), sym, 1, 2, 4.0) == 4.0
    @test compute(TwoPathEffect(L; dir1=:sym, dir2=:sym, combine=:product), sym, 1, 2,
                  4.0) == 3.0 * 1.0
    # … and a third actor reached only by the receiver's leg adds nothing
    @test compute(TwoPathEffect(L; dir1=:sym, dir2=:sym, combine=:product), sym, 1, 4,
                  4.0) == 0.0

    @test name(OTP()) == "otp"
    @test name(OTP(combine=:count)) == "otp.count"
    @test name(ISP(memory=Window(2.0))) == "isp[window=2.0]"
    @test_throws ArgumentError OTP(combine=:median)
    @test_throws ArgumentError OSP(ordered=true)
    @test_throws ArgumentError OTP(ordered=true, order=:leg1_first)
    @test_throws ArgumentError TwoPathEffect(EventLayer(); dir1=:sym, order=:leg1_first)
    @test_throws ArgumentError TwoPathEffect(EventLayer(); dir2=:around)
end

@testset "Structural balance (Brandes, Lerner & Snijders 2009)" begin
    h = build_history([Event(1, 3, 1.0; eventtype=:positive),
                       Event(2, 3, 2.0; eventtype=:negative),
                       Event(4, 1, 3.0; eventtype=:negative),
                       Event(4, 2, 4.0; eventtype=:negative),
                       Event(4, 2, 5.0; eventtype=:negative)])
    # 3 is 1's friend and 2's enemy, so 2 is an enemy of 1's friend and 1 is a
    # friend of 2's enemy (Brandes et al. 2009: friendOfEnemy(a,b) = √Σ ω⁻(a,i)ω⁺(i,b));
    # 4 is an enemy of both (weights 1 and 2)
    @test compute(BalanceEffect(:enemy_of_friend), h, 1, 2, 6.0) == 1.0
    @test compute(BalanceEffect(:friend_of_enemy), h, 2, 1, 6.0) == 1.0
    @test compute(BalanceEffect(:friend_of_enemy), h, 1, 2, 6.0) == 0.0
    @test compute(BalanceEffect(:friend_of_friend), h, 1, 2, 6.0) == 0.0
    @test compute(BalanceEffect(:enemy_of_enemy), h, 1, 2, 6.0) ≈ sqrt(1 * 2)
    @test compute(BalanceEffect(:enemy_of_enemy; root=false), h, 1, 2, 6.0) == 2.0
    # the undirected weights make the same-sign statistics symmetric
    @test compute(BalanceEffect(:enemy_of_enemy), h, 2, 1, 6.0) ≈ sqrt(2)
    custom = BalanceEffect(:friend_of_friend; positive=:ally, negative=:foe)
    @test compute(custom, build_history([Event(1, 3, 1.0; eventtype=:ally),
                                         Event(3, 2, 2.0; eventtype=:ally)]), 1, 2, 3.0) == 1.0
    @test name(BalanceEffect(:friend_of_enemy)) == "friend_of_enemy"
    @test_throws ArgumentError BalanceEffect(:frenemy)
end

@testset "Four-cycles (hand-computed)" begin
    # two-mode: senders 1–3, targets 11–13
    h = build_history([Event(1, 11, 1.0), Event(1, 11, 2.0), Event(2, 11, 3.0),
                       Event(2, 12, 4.0), Event(2, 12, 5.0), Event(2, 12, 6.0),
                       Event(3, 11, 7.0), Event(3, 12, 8.0)])
    # 1 → 12 closes 1 → 11 ← 2 → 12 (weights 2, 1, 3) and 1 → 11 ← 3 → 12 (2, 1, 1)
    @test compute(FourCycleEffect(), h, 1, 12, 9.0) == min(2, 1, 3) + min(2, 1, 1)
    @test compute(FourCycleEffect(combine=:product), h, 1, 12, 9.0) == 2 * 1 * 3 + 2 * 1 * 1
    @test compute(FourCycleEffect(combine=:count), h, 1, 12, 9.0) == 2.0
    @test compute(FourCycleEffect(combine=:product, root=true), h, 1, 12, 9.0) ≈ cbrt(8)
    @test compute(FourCycleEffect(), h, 1, 13, 9.0) == 0.0
    @test compute(FourCycleEffect(EventLayer(memory=Window(2.5))), h, 1, 12, 9.0) == 0.0
    @test name(FourCycleEffect()) == "fourcycle"
    @test_throws ArgumentError FourCycleEffect(combine=:max)
end

@testset "Recency (hand-computed)" begin
    h = build_history([Event(2, 1, 1.0), Event(3, 1, 2.0), Event(1, 4, 3.0),
                       Event(1, 2, 4.0), Event(2, 1, 5.0)])
    # who wrote to 1 most recently: 2 (t = 5), then 3 (t = 2)
    @test compute(RecencyRank(:receive), h, 1, 2, 6.0) == 1.0
    @test compute(RecencyRank(:receive), h, 1, 3, 6.0) == 0.5
    @test compute(RecencyRank(:receive), h, 1, 4, 6.0) == 0.0
    # whom 1 wrote to most recently: 2 (t = 4), then 4 (t = 3)
    @test compute(RecencyRank(:send), h, 1, 2, 6.0) == 1.0
    @test compute(RecencyRank(:send), h, 1, 4, 6.0) == 0.5
    @test compute(RecencyRank(:send), h, 1, 3, 6.0) == 0.0
    @test_throws ArgumentError RecencyRank(:both)

    @test compute(TimeSince(:dyad), h, 1, 2, 6.0) == 1 / 3           # last 1 → 2 at t = 4
    @test compute(TimeSince(:reverse), h, 1, 2, 6.0) == 1 / 2        # last 2 → 1 at t = 5
    @test compute(TimeSince(:pair), h, 1, 2, 6.0) == 1 / 2
    @test compute(TimeSince(:send_sender), h, 1, 2, 6.0) == 1 / 3    # 1 last sent at 4
    @test compute(TimeSince(:send_receiver), h, 1, 2, 6.0) == 1 / 2  # 2 last sent at 5
    @test compute(TimeSince(:receive_sender), h, 1, 2, 6.0) == 1 / 2 # 1 last received at 5
    @test compute(TimeSince(:receive_receiver), h, 1, 2, 6.0) == 1 / 3
    @test compute(TimeSince(:dyad), h, 4, 3, 6.0) == 0.0             # never
    @test compute(TimeSince(:dyad; transform=identity, empty=-1.0), h, 4, 3, 6.0) == -1.0
    @test compute(TimeSince(:dyad; transform=identity), h, 1, 2, 6.0) == 2.0
    @test compute(TimeSince(:dyad; transform=Δ -> exp(-Δ)), h, 1, 2, 6.0) ≈ exp(-2)
    # in events: 1 → 2 was the 4th of 5, so 2 events ago
    @test compute(TimeSince(:dyad; transform=identity, clock=:order), h, 1, 2, 6.0) == 2.0
    @test inverse_gap(3.0) == 0.25
    @test name(TimeSince(:dyad)) == "recencyContinue"
    @test name(TimeSince(:send_sender; transform=identity)) == "recencySendSender.gap"
    @test_throws ArgumentError TimeSince(:nowhere)

    @test compute(PShiftABAB(), h, 2, 1, 6.0) == 1.0
    @test compute(PShiftABAB(), h, 1, 2, 6.0) == 0.0
    @test compute(PShiftABAB(), build_history(Event{Float64}[]), 1, 2, 0.0) == 0.0
    # undirected shifts ignore the orientation a pair is stored in
    @test compute(UndirectedPShift(:AB_AB), h, 1, 2, 6.0) == 1.0
    @test compute(UndirectedPShift(:AB_AB), h, 2, 1, 6.0) == 1.0
    @test compute(UndirectedPShift(:AB_AB), h, 1, 3, 6.0) == 0.0
    @test compute(UndirectedPShift(:AB_AY), h, 1, 3, 6.0) == 1.0
    @test compute(UndirectedPShift(:AB_AY), h, 4, 2, 6.0) == 1.0
    @test compute(UndirectedPShift(:AB_AY), h, 3, 4, 6.0) == 0.0
    @test compute(UndirectedPShift(:AB_AY), h, 1, 2, 6.0) == 0.0
    @test name(UndirectedPShift(:AB_AY)) == "PSAB-AY.undirected"
    @test_throws ArgumentError UndirectedPShift(:AB_BA)
end

@testset "Neighbourhood statistics (hand-computed)" begin
    h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 3, 3.0),
                       Event(1, 4, 4.0), Event(4, 3, 5.0)])
    # 1 is the source of 1→2,1→3,2→3 and of 1→4,1→3,4→3
    @test compute(NodeTransitivity(role=:sender), h, 1, 5, 6.0) == 2.0
    @test compute(NodeTransitivity(), h, 5, 1, 6.0) == 2.0
    @test compute(NodeTransitivity(), h, 1, 2, 6.0) == 0.0
    # out-profiles {2,3,4} and {3}: for the pair (1, 2) the two actors are left
    # out, so the profiles are {3,4} and {3}
    @test compute(StructuralSimilarity(), h, 1, 2, 6.0) == 0.5
    @test compute(StructuralSimilarity(measure=:cosine), h, 1, 2, 6.0) ≈ 1 / sqrt(2)
    @test compute(StructuralSimilarity(direction=:in), h, 3, 4, 6.0) == 0.5   # {1,2} vs {1}
    @test compute(StructuralSimilarity(), h, 3, 5, 6.0) == 0.0
    @test_throws ArgumentError StructuralSimilarity(measure=:pearson)
    @test_throws ArgumentError NodeTransitivity(role=:both)
end

# ----------------------------------------------------------------------------
# Agreement with relevent (R) and REM.jl
# ----------------------------------------------------------------------------

@testset "relevent's design statistics (golden), both interfaces" begin
    # relevent::rem.dyad 1.2.1's own design statistics, read off
    # rem.dyad.lambda with one unit coefficient (test/fixtures/r/relevent_catalogue.R):
    # 24 columns, every candidate dyad before each of 14 events, an isolated actor
    # and repeated dyads. The Revel call is the one `effect_catalogue()` names for
    # each relevent effect; FESnd/FERec/FEInt are indicator covariates.
    g = NetworkCore.load_golden(joinpath(@__DIR__, "fixtures", "relevent_catalogue.toml"))
    n = Int(g.values["n_actors"])
    events = [Event(Int(s), Int(r), Float64(t)) for (s, r, t) in zip(
              g.values["input_sender"], g.values["input_receiver"], g.values["input_time"])]
    e0 = 1 / (n - 1)      # relevent's value before any event
    pairs = Pair{String,Any}[
        "NIDSnd" => IndegreeSender(scaling=:prop, empty=e0),
        "NIDRec" => IndegreeReceiver(scaling=:prop, empty=e0),
        "NODSnd" => OutdegreeSender(scaling=:prop, empty=e0),
        "NODRec" => OutdegreeReceiver(scaling=:prop, empty=e0),
        "NTDegSnd" => TotaldegreeSender(scaling=:prop, empty=e0),
        "NTDegRec" => TotaldegreeReceiver(scaling=:prop, empty=e0),
        "RRecSnd" => RecencyRank(:receive), "RSndSnd" => RecencyRank(:send),
        "OTPSnd" => OTP(), "ITPSnd" => ITP(), "ISPSnd" => ISP(),
        "CovEvent" => TieEffect([(3i - j) / 7 for i in 1:n, j in 1:n])]
    for k in 2:n
        push!(pairs, "FESnd_$k" => SendEffect((1:n) .== k),
                     "FERec_$k" => ReceiveEffect((1:n) .== k),
                     "FEInt_$k" => SumEffect((1:n) .== k))
    end
    @test length(pairs) == 24
    # the calls are the catalogue's (the fixed effects with k = 2)
    cat = effect_catalogue()
    for (key, call) in ("NIDSnd" => "IndegreeSender(scaling=:prop, empty=1/(n-1))",
                        "RRecSnd" => "RecencyRank(:receive)", "OTPSnd" => "OTP()",
                        "CovEvent" => "TieEffect(matrix)",
                        "FESnd" => "SendEffect((1:n) .== k)", "FEInt" => "SumEffect((1:n) .== k)")
        @test only(cat[cat.relevent .== key, :revel]) == call
    end
    for (key, stat) in pairs
        history = InteractionHistory()
        state = REM.EventNetworkState{Float64}()
        actual = Float64[]
        interface_gap = 0.0
        for e in events
            state.current_time = e.time
            for s in 1:n, r in 1:n
                s == r && continue
                v = compute(stat, history, s, r, e.time)
                push!(actual, v)
                interface_gap = max(interface_gap, abs(v - compute(stat, state, s, r)))
            end
            update_history!(history, e)
            REM.update!(state, e)
        end
        # deterministic counts, ranks and shares: the fixture's 1e-12
        @test NetworkCore.check_golden(g, key, actual)
        # the history interface and REM's state interface are one computation
        @test interface_gap == 0.0
    end
end

@testset "Half-life statistics against a direct scan, both interfaces" begin
    # Decayed volumes and recency, against the definition evaluated by rescanning
    # the whole past at every evaluation: Σ exp(−log 2 · (t − tₖ) / h) over the
    # past events that qualify.
    rng = Xoshiro(42)
    n = 6
    halflife = 9.0
    d = log(2) / halflife
    events = random_events(rng, n, 250)
    scan(past, t, keep) = sum((exp(-d * (t - e.time)) for e in past if keep(e)); init=0.0)
    h = InteractionHistory()
    state = REM.EventNetworkState{Float64}()
    stats = (out = Inertia(memory=HalfLife(halflife)),
             inc = Reciprocation(memory=HalfLife(halflife)),
             both = DyadActivity(memory=HalfLife(halflife)),
             snd = OutdegreeSender(memory=HalfLife(halflife)),
             rcv = IndegreeReceiver(memory=HalfLife(halflife)),
             last = TimeSince(:dyad; transform=Δ -> exp(-d * Δ)))
    worst = 0.0
    for (m, ev) in enumerate(events)
        t = ev.time
        state.current_time = t
        past = events[1:(m - 1)]
        for (s, r) in ((1, 2), (2, 1), (ev.sender, ev.receiver), (5, 3))
            s == r && continue
            out = scan(past, t, e -> e.sender == s && e.receiver == r)
            inc = scan(past, t, e -> e.sender == r && e.receiver == s)
            last = findlast(e -> e.sender == s && e.receiver == r, past)
            expected = (out = out, inc = inc, both = out + inc,
                        snd = scan(past, t, e -> e.sender == s),
                        rcv = scan(past, t, e -> e.receiver == r),
                        last = last === nothing ? 0.0 : exp(-d * (t - past[last].time)))
            for key in keys(stats)
                v = compute(stats[key], h, s, r, t)
                worst = max(worst, abs(v - expected[key]))
                @test compute(stats[key], state, s, r) == v
            end
        end
        update_history!(h, ev)
        REM.update!(state, ev)
    end
    @test worst < 1e-10
end

@testset "REM's state interface: what it cannot carry" begin
    state = REM.EventNetworkState{Float64}()
    # a statistic that reads the event log refuses a state that kept none
    lean = REM.EventNetworkState{Float64}(; keep_history=false)
    @test_throws ArgumentError compute(Inertia(), lean, 1, 2)
    @test compute(SendEffect([1.0, 2.0]), lean, 2, 1) == 2.0
    # event types are not carried by REM's event log
    @test_throws ArgumentError compute(Inertia(types=:a), state, 1, 2)
    @test REM.needs_history(Inertia())
    @test !REM.needs_history(SendEffect([1.0, 2.0]))
    @test !REM.needs_history(TieEffect(zeros(2, 2)))
end

# ----------------------------------------------------------------------------
# Covariates
# ----------------------------------------------------------------------------

@testset "Covariates and exogenous effects" begin
    h = build_history(Event{Float64}[])
    x = [3.0, 1.0, 4.0]
    @test compute(SendEffect(x), h, 1, 2, 0.0) == 3.0
    @test compute(ReceiveEffect(x), h, 1, 2, 0.0) == 1.0
    @test compute(DiffEffect(x), h, 2, 1, 0.0) == 2.0
    @test compute(DiffEffect(x; absolute=false), h, 2, 1, 0.0) == -2.0
    @test compute(SimEffect(x), h, 2, 1, 0.0) == -2.0
    @test compute(AverageEffect(x), h, 1, 2, 0.0) == 2.0
    @test compute(MinimumEffect(x), h, 1, 3, 0.0) == 3.0
    @test compute(MaximumEffect(x), h, 1, 3, 0.0) == 4.0
    @test compute(SumEffect(x), h, 1, 3, 0.0) == 7.0
    @test compute(ProductEffect(x), h, 1, 3, 0.0) == 12.0
    @test compute(ProductEffect(x, [0.0, 1.0, 1.0]), h, 1, 2, 0.0) == 3.0
    @test compute(MatchEffect([1, 2, 1]), h, 1, 3, 0.0) == 1.0
    @test compute(MatchEffect([1, 2, 1]), h, 1, 2, 0.0) == 0.0
    @test compute(SendEffect(x; transform=log), h, 1, 2, 0.0) ≈ log(3)
    # numbers in an untyped container are still numbers
    @test !Covariate(Any[1, 2.5, 3]).categorical
    @test name(SendEffect(Covariate(x; name="age"))) == "send.age"
    @test name(DiffEffect(x)) == "absdiff.x"
    @test name(ProductEffect(Covariate(x; name="a"), Covariate(x; name="b"))) == "product.a.b"
    @test name(SendEffect(x; name="seniority")) == "seniority"

    # categorical covariates match, and refuse arithmetic
    dept = Covariate(["ops", "legal", "ops"]; name="dept")
    @test dept.categorical && dept.levels == ["ops", "legal"]
    @test compute(MatchEffect(dept), h, 1, 3, 0.0) == 1.0
    @test compute(MatchEffect(dept), h, 1, 2, 0.0) == 0.0
    @test_throws ArgumentError SendEffect(dept)
    @test_throws ArgumentError DiffEffect(dept)
    @test occursin("categorical", sprint(show, dept))

    # time-varying: column k is in force from times[k]
    load = Covariate([0.0, 10.0], [1.0 3.0; 2.0 2.0; 0.5 0.5]; name="load")
    @test covariate_value(load, 1, -5.0) == 1.0      # before the first time: column 1
    @test covariate_value(load, 1, 9.99) == 1.0
    @test covariate_value(load, 1, 10.0) == 3.0
    @test compute(SendEffect(load), h, 1, 2, 5.0) == 1.0
    @test compute(SendEffect(load), h, 1, 2, 15.0) == 3.0
    @test compute(DiffEffect(load), h, 1, 2, 15.0) == 1.0

    # an actor the covariate does not cover is an error, never a silent zero
    @test_throws ArgumentError compute(SendEffect(x), h, 4, 1, 0.0)
    @test_throws ArgumentError compute(ReceiveEffect(x), h, 1, 0, 0.0)
    @test_throws ArgumentError Covariate(Float64[])
    @test_throws ArgumentError Covariate([1.0, NaN])
    @test_throws ArgumentError Covariate([0.0, 1.0], ones(3, 3))
    @test_throws ArgumentError Covariate([1.0, 0.0], ones(3, 2))
    @test_throws ArgumentError CovariateEffect(:ratio, x)
    @test_throws ArgumentError CovariateEffect(:send, x, [1.0, 2.0, 3.0])

    friends = [0 1 0; 1 0 0; 0 0 0]
    @test compute(TieEffect(friends), h, 1, 2, 0.0) == 1.0
    @test compute(TieEffect(friends), h, 1, 3, 0.0) == 0.0
    later = TieEffect([0.0, 10.0], [friends, 2 .* friends]; name="friend")
    @test compute(later, h, 1, 2, 5.0) == 1.0
    @test compute(later, h, 1, 2, 10.0) == 2.0
    @test compute(TieEffect(2 .* friends; transform=:indicator), h, 1, 2, 0.0) == 1.0
    @test name(later) == "friend"
    @test_throws ArgumentError compute(TieEffect(friends), h, 1, 4, 0.0)
    @test_throws ArgumentError TieEffect(ones(2, 3))
    @test_throws ArgumentError TieEffect(fill(Inf, 2, 2))
    @test_throws ArgumentError TieEffect([0.0], [friends, friends])

    weekend = GlobalEffect(t -> mod(floor(t), 7) >= 5 ? 1.0 : 0.0; name="weekend")
    @test compute(weekend, h, 1, 2, 5.5) == 1.0
    @test compute(weekend, h, 2, 3, 2.5) == 0.0
    period = GlobalEffect([0.0, 10.0], [0.0, 1.0]; name="after")
    @test compute(period, h, 1, 2, 9.0) == 0.0
    @test compute(period, h, 1, 2, 10.0) == 1.0
    @test_throws ArgumentError GlobalEffect([0.0, 1.0], [1.0])
end

@testset "Covariates read through the network (hand-computed)" begin
    power = [1.0, 5.0, 3.0, 0.0, 9.0]
    h = build_history([Event(2, 4, 1.0), Event(3, 4, 2.0), Event(3, 4, 3.0),
                       Event(4, 5, 4.0)])
    # senders to 4: actors 2 (once) and 3 (twice)
    @test compute(TertiusEffect(power), h, 1, 4, 5.0) == 4.0
    @test compute(TertiusEffect(power; aggregate=:sum), h, 1, 4, 5.0) == 8.0
    @test compute(TertiusEffect(power; aggregate=:max), h, 1, 4, 5.0) == 5.0
    @test compute(TertiusEffect(power; aggregate=:min), h, 1, 4, 5.0) == 3.0
    @test compute(TertiusEffect(power; aggregate=:range), h, 1, 4, 5.0) == 2.0
    @test compute(TertiusEffect(power; aggregate=:sd), h, 1, 4, 5.0) == 1.0
    # weighting the neighbours by their tie: (5·1 + 3·2)/3
    @test compute(TertiusEffect(power; tie_weighted=true), h, 1, 4, 5.0) ≈ 11 / 3
    # goldfish's tertiusDiff: abs(x[sender] − mean)
    @test compute(TertiusEffect(power; difference=true), h, 1, 4, 5.0) == 3.0
    # the candidate's own sender is left out of the receiver's neighbourhood
    @test compute(TertiusEffect(power), h, 2, 4, 5.0) == 3.0
    @test compute(TertiusEffect(power; exclude_other=false), h, 2, 4, 5.0) == 4.0
    # no neighbours: `empty`
    @test compute(TertiusEffect(power; empty=3.6), h, 1, 2, 5.0) == 3.6
    # the sender's out-neighbours: 4 sent to 5
    @test compute(TertiusEffect(power; role=:sender, direction=:out), h, 4, 1, 5.0) == 9.0
    @test name(TertiusEffect(power)) == "tertius.x"
    @test name(TertiusEffect(power; difference=true, aggregate=:max)) == "tertiusDiff.x.max"
    @test_throws ArgumentError TertiusEffect(power; aggregate=:median)
    @test_throws ArgumentError TertiusEffect(Covariate(["a", "b"]))

    party = [1, 1, 2, 9, 1]
    # events to 4 from party-mates of actor 1: only actor 2's single event
    @test compute(MatchedDegree(party), h, 1, 4, 5.0) == 1.0
    @test compute(MatchedDegree(party), h, 5, 4, 5.0) == 1.0
    # the candidate's own dyad never counts
    @test compute(MatchedDegree(party), h, 2, 4, 5.0) == 0.0
    # from actor 3's side: its events to actors of the receiver's party
    @test compute(MatchedDegree([1, 1, 2, 9, 9]; role=:sender), h, 4, 1, 5.0) == 0.0
    graded = MatchedDegree(power; similarity=(a, b) -> exp(-abs(a - b)))
    @test compute(graded, h, 1, 4, 5.0) ≈ exp(-4) + 2exp(-2)
    @test name(MatchedDegree(Covariate(party; name="party"))) == "matchedIndegree.party"
end

# ----------------------------------------------------------------------------
# Interactions and wrappers
# ----------------------------------------------------------------------------

@testset "Interactions, transforms, standardisation" begin
    female = [1.0, 0.0, 1.0]
    h = build_history([Event(2, 1, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)])
    inter = Interaction(Reciprocation(), SendEffect(female; name="female"))
    @test name(inter) == "reciprocity:female"
    @test compute(inter, h, 1, 2, 4.0) == 2.0
    @test compute(inter, h, 2, 1, 4.0) == 0.0
    # any statistic with the history interface can be a part: a p-shift here
    mixed = Interaction(PShift(:AB_BA), ReceiveEffect(female); name="answer_a_woman")
    @test compute(mixed, h, 3, 1, 4.0) == 1.0
    @test name(mixed) == "answer_a_woman"
    three = Interaction(Inertia(), Reciprocation(), OutdegreeSender())
    @test compute(three, h, 2, 1, 4.0) == 2 * 0 * 2
    @test_throws ArgumentError Interaction(Inertia())

    @test compute(Transformed(Inertia(), log1p), h, 2, 1, 4.0) ≈ log(3)
    @test name(Transformed(Inertia(), log1p)) == "log1p(inertia)"
    centred = Transformed(SendEffect([30.0, 50.0, 40.0]), x -> x - 40; name="age_c")
    @test compute(centred, h, 2, 1, 4.0) == 10.0
    @test compute(Transformed(PShift(:AB_BA), x -> 2x), h, 3, 1, 4.0) == 2.0

    z = Standardized(Inertia(), 3)
    vals = [compute(Inertia(), h, s, r, 4.0) for (s, r) in dyads_of(3)]
    μ, σ = mean(vals), sqrt(mean(abs2, vals .- mean(vals)))
    @test [compute(z, h, s, r, 4.0) for (s, r) in dyads_of(3)] ≈ (vals .- μ) ./ σ
    @test abs(mean(compute(z, h, s, r, 4.0) for (s, r) in dyads_of(3))) < 1e-12
    # a statistic that is constant over the risk set standardises to zero
    flat = Standardized(GlobalEffect(t -> 2.0), 3)
    @test compute(flat, h, 1, 2, 4.0) == 0.0
    # the sample-standard-deviation form (remstats' "std") is a constant rescaling
    zc = Standardized(Inertia(), 3; corrected=true)
    @test compute(zc, h, 2, 1, 4.0) ≈ compute(z, h, 2, 1, 4.0) * sqrt(5 / 6)
    @test_throws ArgumentError Standardized(Inertia(), 2; directed=false, corrected=true)
    @test name(z) == "std(inertia)"
    @test_throws ArgumentError Standardized(Inertia(), 1)

    # both interfaces, through the wrappers
    state = REM.EventNetworkState{Float64}()
    for e in h.events
        REM.update!(state, e)
    end
    state.current_time = 4.0
    for stat in (inter, three, Transformed(Inertia(), log1p), z)
        for (s, r) in dyads_of(3)
            @test compute(stat, state, s, r) ≈ compute(stat, h, s, r, 4.0)
        end
    end
end

@testset "Traits: interval-constancy and history use" begin
    ic = Revel.is_interval_constant
    @test ic(Inertia()) && ic(OTP()) && ic(OutdegreeSender()) && ic(FourCycleEffect())
    @test ic(Inertia(memory=HalfLife(Inf)))
    @test !ic(Inertia(memory=HalfLife(2.0)))
    @test !ic(OTP(memory=Window(2.0)))
    @test !ic(TimeSince(:dyad))
    @test ic(RecencyRank(:send)) && ic(PShiftABAB())
    @test ic(SendEffect([1.0, 2.0])) && ic(TieEffect(zeros(2, 2)))
    @test !ic(SendEffect(Covariate([0.0, 1.0], [1.0 2.0; 3.0 4.0])))
    @test !ic(GlobalEffect(t -> t))
    @test ic(Interaction(Inertia(), SendEffect([1.0, 2.0])))
    @test !ic(Interaction(Inertia(), TimeSince(:dyad)))
    @test ic(Interaction(PShift(:AB_BA), Inertia()))
    @test ic(Transformed(Inertia(), log1p)) && ic(Standardized(Inertia(), 3))
    @test ic(TertiusEffect([1.0, 2.0])) && !ic(TertiusEffect([1.0, 2.0]; memory=Window(1.0)))
    @test Inertia() isa AbstractRevelStatistic
    @test Inertia() isa AbstractStatistic
    @test sprint(show, Inertia()) == "DyadEffect(inertia)"
end

@testset "compute allocates nothing on a warmed history" begin
    rng = Xoshiro(5)
    events = random_events(rng, 12, 200)
    h = build_history(events)
    t = events[end].time + 1
    x = collect(1.0:12)
    stats = (Inertia(), Reciprocation(scaling=:prop), OutdegreeSender(),
             IndegreeReceiver(memory=HalfLife(5.0)), TotaldegreeDyad(), OTP(), ITP(),
             OSP(combine=:count), ISP(memory=Window(20.0)), FourCycleEffect(),
             RecencyRank(:send), TimeSince(:dyad), PShiftABAB(), SendEffect(x),
             DiffEffect(x), TieEffect(ones(12, 12)), TertiusEffect(x),
             MatchedDegree(x), Interaction(Inertia(), SendEffect(x)),
             Transformed(Inertia(), log1p))
    alloc(stat, h, t) = @allocated compute(stat, h, 3, 7, t)
    for stat in stats
        compute(stat, h, 3, 7, t); alloc(stat, h, t)        # warm up
        @test alloc(stat, h, t) == 0
    end
end

# ----------------------------------------------------------------------------
# Risk sets and designs
# ----------------------------------------------------------------------------

@testset "Risk sets" begin
    events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)]
    stats = [Inertia(), Reciprocation()]
    seen = NamedTuple[]
    applied = each_risk_set(events, stats, 3) do v
        push!(seen, (index=v.index, dyads=copy(v.dyads), X=copy(v.X), case=v.case))
    end
    @test applied === :none
    @test length(seen) == 3
    @test all(length(s.dyads) == 6 for s in seen)
    @test seen[1].dyads[seen[1].case] == (1, 2)
    @test all(iszero, seen[1].X)                       # no look-ahead
    @test seen[2].X[seen[2].case, :] == [0.0, 1.0]     # 2 → 1 reciprocates 1 → 2
    @test seen[3].X[findfirst(==((1, 2)), seen[3].dyads), :] == [1.0, 1.0]

    sizes(; kw...) = (out = Int[]; each_risk_set(v -> push!(out, length(v.dyads)),
                                                 events, stats, 3; kw...); out)
    @test sizes(riskset=:sender) == [2, 2, 2]
    @test sizes(riskset=:receiver) == [2, 2, 2]
    @test sizes(riskset=:active) == [3, 3, 3]
    @test sizes(riskset=[(1, 2), (2, 1), (1, 3), (3, 1)]) == [4, 4, 4]
    @test sizes(directed=false) == [3, 3, 3]
    @test sizes(cases=[2, 3]) == [6, 6]
    # a predicate on WHO acts restricts each case's risk set to the dyads it
    # admits (sender 1's two dyads) — one on the time does not
    @test sizes(cases=e -> e.sender == 1) == [2, 2]
    @test sizes(cases=e -> e.time > 1.5) == [6, 6]
    @test sizes(cases=[true, false, false]) == [6]
    # a time-varying risk set
    grow = (m, ev) -> m < 3 ? [(1, 2), (2, 1)] : [(1, 2), (2, 1), (1, 3)]
    @test sizes(riskset=grow) == [2, 2, 3]
    # under receiver choice the choice set is the observed sender's
    each_risk_set(events, stats, 3; riskset=:sender) do v
        @test all(d -> d[1] == v.event.sender, v.dyads)
    end
    # undirected: the case is the unordered pair
    each_risk_set(events, [Inertia(symmetric=true)], 3; directed=false) do v
        @test v.dyads[v.case] == minmax(v.event.sender, v.event.receiver)
    end
    # cases that are skipped still build the history
    each_risk_set(events, stats, 3; cases=[3]) do v
        @test v.X[findfirst(==((2, 1)), v.dyads), :] == [1.0, 1.0]
    end

    @test two_mode_dyads(1:2, 11:13) == [(1, 11), (1, 12), (1, 13), (2, 11), (2, 12), (2, 13)]
    @test_throws ArgumentError two_mode_dyads(1:3, 3:5)
    @test_throws ArgumentError each_risk_set(identity, events, stats, 1)
    @test_throws ArgumentError each_risk_set(identity, events, stats, 3; riskset=:everyone)
    @test_throws ArgumentError each_risk_set(identity, events, stats, 3; riskset=[(1, 2), (1, 2)])
    @test_throws ArgumentError each_risk_set(identity, events, stats, 3; riskset=[(1, 2), (2, 1)])
    @test_throws ArgumentError each_risk_set(identity, events, stats, 3; riskset=[(1, 3), (1, 2)])
    @test_throws ArgumentError each_risk_set(identity, events, stats, 3; cases=[4])
    @test_throws ArgumentError each_risk_set(identity, events, stats, 3; cases="all")
    @test_throws ArgumentError each_risk_set(identity, events, stats, 3;
                                             directed=false, riskset=:sender)
    # an event outside the declared universe
    @test_throws ArgumentError each_risk_set(identity, [Event(1, 4, 1.0)], stats, 3)
end

@testset "Tied event times" begin
    tied = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 2.0), Event(3, 1, 3.0)]
    stats = [Inertia(), Reciprocation()]
    @test_throws ArgumentError each_risk_set(identity, tied, stats, 3)
    @test_throws ArgumentError each_risk_set(identity, tied, stats, 3; ties=:batch)
    @test_throws ArgumentError each_risk_set(identity, tied, stats, 3; ties=:sorted)

    rows(policy) = (out = Vector{Float64}[];
                    applied = each_risk_set(v -> push!(out, v.X[v.case, :]), tied, stats, 3;
                                            ties=policy); (applied, out))
    applied, ordered = rows(:ordered)
    @test applied === :ordered
    # in sequence order the second tied event sees the first: 1 → 2 after 2 → 1
    @test ordered[3] == [1.0, 1.0]
    applied, frozen = rows(:breslow)
    @test applied === :breslow
    # with the history frozen across the tie it does not
    @test frozen[3] == [1.0, 0.0]
    @test frozen[2] == ordered[2] && frozen[4] == ordered[4]

    weights = Tuple{Vector{Int}, Float64}[]
    each_risk_set(tied, stats, 3; ties=:efron) do v
        push!(weights, (sort(v.tied), v.tie_weight))
    end
    @test weights[1] == (Int[], 1.0)
    @test weights[2][2] == 1.0 && weights[3][2] == 0.5     # 1 − (j−1)/d
    @test weights[2][1] == weights[3][1] && length(weights[2][1]) == 2
    # Efron needs the tied cases to be distinct dyads
    twice = [Event(1, 2, 1.0), Event(1, 2, 1.0), Event(2, 1, 2.0)]
    @test_throws ArgumentError each_risk_set(identity, twice, stats, 3; ties=:efron)
    @test each_risk_set(identity, twice, stats, 3; ties=:breslow) === :breslow

    # on tie-free data every policy builds the identical design
    clean = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0), Event(3, 1, 4.0)]
    base = event_design(clean, stats, 3)
    for policy in (:ordered, :breslow, :efron)
        @test event_design(clean, stats, 3; ties=policy) == base
    end
    @test metadata(base, "tie_method") == "none"
    @test metadata(event_design(tied, stats, 3; ties=:efron), "tie_method") == "efron"
end

@testset "Event design" begin
    rng = Xoshiro(2)
    events = random_events(rng, 5, 30)
    stats = [Inertia(), Reciprocation(), OTP()]
    design = event_design(events, stats, 5)
    @test size(design, 1) == 30 * 20
    @test names(design) == ["event_index", "sender", "receiver", "time", "is_event",
                            "stratum", "risk_set_size", "sampling_prob", "tie_weight",
                            "inertia", "reciprocity", "otp"]
    @test sum(design.is_event) == 30
    @test all(design.risk_set_size .== 20) && all(design.sampling_prob .== 1.0)
    @test all(design.tie_weight .== 1.0)
    @test design.stratum[design.is_event] == 1:30
    # the case rows are the observed events, in time order
    @test collect(zip(design.sender[design.is_event], design.receiver[design.is_event])) ==
          [(e.sender, e.receiver) for e in events]

    # sampled controls: the case plus n_controls distinct dyads per event,
    # reproducible from `rng` and independent of the global RNG
    s1 = event_design(events, stats, 5; n_controls=4, rng=Xoshiro(9))
    Random.seed!(1234); s2 = event_design(events, stats, 5; n_controls=4, rng=Xoshiro(9))
    Random.seed!(9999); s3 = event_design(events, stats, 5; n_controls=4, rng=Xoshiro(9))
    @test s1 == s2 == s3
    @test size(s1, 1) == 30 * 5
    @test all(s1.sampling_prob .≈ 4 / 19)
    @test all(nrow(g) == 5 && sum(g.is_event) == 1 &&
              allunique(collect(zip(g.sender, g.receiver)))
              for g in groupby(s1, :stratum))
    @test event_design(events, stats, 5; n_controls=4, rng=Xoshiro(10)) != s1
    # more controls than the risk set holds: the full risk set
    @test event_design(events, stats, 5; n_controls=500) == design
    @test_throws ArgumentError event_design(events, stats, 5; n_controls=0)

    @test_throws ArgumentError event_design(events, [Inertia(), Inertia()], 5)
    @test_throws ArgumentError event_design(events, [Inertia(name="stratum")], 5)
    @test_throws ArgumentError event_design(events, AbstractStatistic[], 5)
end

# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------

@testset "fit_revel: the three estimators agree" begin
    rng = Xoshiro(11)
    n = 7
    truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p),
             OTP(transform=:log1p)]
    events = simulate_events(truth, [0.9, 0.6, 0.3], n, 300; rng=rng)
    exact = fit_revel(events, truth, n)
    design = fit_revel(events, truth, n; engine=:design)
    @test exact.engine === :stream && design.engine === :design
    @test exact.fit isa Revel.OrdinalBPMResult && design.fit isa REM.REMResult
    # the same likelihood, maximised by the same shared Newton optimizer
    @test coef(exact) ≈ coef(design) atol = 1e-8
    @test stderror(exact) ≈ stderror(design) atol = 1e-8
    @test loglikelihood(exact) ≈ loglikelihood(design) atol = 1e-8
    @test vcov(exact) ≈ vcov(design) atol = 1e-8
    # ... and as REM.fit_rem on the full risk set, through REM's own interface
    seq = EventSequence(events; actors=ActorSet(1:n))
    rem_fit = fit_rem(seq, truth; n_controls=n * (n - 1) - 1)
    @test coef(rem_fit) ≈ coef(exact) atol = 1e-8
    # ... and as Revel.fit_obpm called directly
    @test coef(Revel.fit_obpm(events, truth, n)) == coef(exact)
    # … and a log partial likelihood written out from its definition, with
    # statistics computed afresh for every event: the fitted value, and a maximum
    function brute_loglik(θ)
        h = InteractionHistory{Float64}(); ll = 0.0
        own = [Inertia(transform=:log1p), Reciprocation(transform=:log1p),
               OTP(transform=:log1p)]
        for e in sort(events; by=x -> x.time)
            η = Dict((s, r) => sum(θ[k] * compute(own[k], h, s, r, e.time) for k in 1:3)
                     for s in 1:n for r in 1:n if s != r)
            ll += η[(e.sender, e.receiver)] - log(sum(exp, values(η)))
            update_history!(h, e)
        end
        return ll
    end
    @test brute_loglik(coef(exact)) ≈ loglikelihood(exact) rtol = 1e-10
    for k in 1:3, δ in (-1e-3, 1e-3)
        θ = copy(coef(exact)); θ[k] += δ
        @test brute_loglik(θ) < loglikelihood(exact)
    end

    # the StatsAPI surface and the result-metadata protocol
    for fit in (exact, design)
        @test all(values(NetworkCore.check_statsapi(fit;
            required=(NetworkCore.STATSAPI_VERBS..., :coefnames), strict=true)))
        @test coefnames(fit) == ["log1p(inertia)", "log1p(reciprocity)", "log1p(otp)"]
        @test coefnames(fit) == coeftable(fit).names == coefnames(fit.fit)
        # a fresh vector: a caller's edit cannot reach the fit
        labels = coefnames(fit); labels[1] = "changed"
        @test coefnames(fit)[1] == "log1p(inertia)"
        @test nobs(fit) == 300 && dof(fit) == 3
        @test aic(fit) ≈ -2loglikelihood(fit) + 6
        @test size(confint(fit)) == (3, 2)
        @test confint(fit; level=0.5)[1, 2] < confint(fit)[1, 2]
        @test coeftable(fit) isa NetworkCore.CoefficientTable
        meta = NetworkCore.fit_metadata(fit)
        @test meta.estimand === :relational_event
        @test NetworkCore.is_exact(fit)
        @test NetworkCore.tie_method(fit) === :none
        @test isempty(NetworkCore.approximations(fit))
        @test occursin("Revel relational event model", sprint(show, MIME"text/plain"(), fit))
        @test sprint(show, fit) == "RevelFit(ordinal, 300 events, 3 statistics)"
    end
    @test NetworkCore.objective(exact) === :likelihood
    @test NetworkCore.se_method(design) === :hessian
    @test revel === fit_revel
    @test fit_revel(shuffle(Xoshiro(1), events), truth, n).events == exact.events
end

@testset "fit_revel recovers known coefficients" begin
    n = 8
    x = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p),
             ReceiveEffect(x; name="x")]
    θ = [0.8, 0.5, 0.7]
    events = simulate_events(truth, θ, n, 1500; rng=Xoshiro(2026))
    fit = fit_revel(events, truth, n)
    @test fit.fit.converged
    # within four standard errors of the truth, and the standard errors are small
    @test all(abs.(coef(fit) .- θ) .< 4 .* stderror(fit))
    @test all(stderror(fit) .< 0.15)
    # the timing model recovers the baseline rate too
    base = [Inertia(transform=:log1p), ReceiveEffect(x; name="x")]
    timed = simulate_events(base, [0.8, 0.7], n, 1500; baseline=0.02, rng=Xoshiro(7))
    tfit = fit_revel(timed, base, n; model=:timing)
    @test tfit.fit isa Revel.TimingModelResult
    @test coefnames(tfit) == ["log_baseline", "log1p(inertia)", "x"]
    @test all(abs.(coef(tfit) .- [log(0.02), 0.8, 0.7]) .< 4 .* stderror(tfit))
    @test all(values(NetworkCore.check_statsapi(tfit;
        required=(NetworkCore.STATSAPI_VERBS..., :coefnames), strict=true)))
    @test coefnames(tfit) == coeftable(tfit).names == coefnames(tfit.fit)
    @test NetworkCore.estimand(tfit) === :relational_event_timing
end

@testset "fit_revel: risk sets, subsets, sampling, standard errors" begin
    n = 6
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    events = simulate_events(stats, [0.8, 0.6], n, 400; rng=Xoshiro(3))

    choice = fit_receiver_choice(events, stats, n)
    @test choice.riskset === :sender && choice.engine === :design
    @test all(choice.fit.risk_set_sizes .== n - 1)
    @test nobs(choice) == 400
    @test all(abs.(coef(choice) .- [0.8, 0.6]) .< 4 .* stderror(choice))
    # a sender covariate is constant within every choice set: not identified
    sender_only = @test_logs (:warn,) (:warn,) (:warn,) match_mode=:any fit_receiver_choice(
        events, [Inertia(transform=:log1p), SendEffect(collect(1.0:n))], n)
    @test sender_only.fit.singular
    @test isnan(stderror(sender_only)[2])
    @test !NetworkCore.is_exact(sender_only)

    # a subset of cases: the other events still build the history
    late = fit_revel(events, stats, n; cases=201:400)
    @test nobs(late) == 200 && count(late.cases) == 200
    @test late.engine === :design
    @test occursin("200 modelled as cases", sprint(show, MIME"text/plain"(), late))
    @test_throws ArgumentError fit_revel(events, stats, n; cases=e -> false)

    # sampled controls are reproducible from rng, and close to the full fit
    full = fit_revel(events, stats, n)
    a = fit_revel(events, stats, n; n_controls=10, rng=Xoshiro(4))
    b = fit_revel(events, stats, n; n_controls=10, rng=Xoshiro(4))
    @test coef(a) == coef(b)
    @test all(abs.(coef(a) .- coef(full)) .< 3 .* stderror(full))
    @test !NetworkCore.is_exact(a)
    @test all(a.fit.sampling_probs .≈ 10 / 29)

    sandwich = fit_revel(events, stats, n; se=:sandwich)
    @test sandwich.engine === :design
    @test NetworkCore.se_method(sandwich) === :sandwich
    @test coef(sandwich) ≈ coef(full) atol = 1e-8
    @test stderror(sandwich) != stderror(full)

    # undirected events on symmetric layers
    und = [Inertia(symmetric=true, transform=:log1p), SharedPartners(transform=:log1p)]
    ufit = fit_revel(events, und, n; directed=false)
    @test all(ufit.fit.risk_set_sizes .== n * (n - 1) ÷ 2)
    @test !ufit.directed && ufit.fit.converged

    # a two-mode risk set
    users, items = 1:4, 5:9
    tm_stats = [Inertia(transform=:log1p), IndegreeReceiver(transform=:log1p),
                FourCycleEffect(transform=:log1p)]
    tm = simulate_events(tm_stats, [0.7, 0.3, 0.2], 9, 300; rng=Xoshiro(5),
                         riskset=two_mode_dyads(users, items))
    @test all(e -> e.sender in users && e.receiver in items, tm)
    tfit = fit_revel(tm, tm_stats, 9; riskset=two_mode_dyads(users, items))
    @test all(tfit.fit.risk_set_sizes .== 20) && tfit.fit.converged
    # and it recovers the coefficients that generated the data
    @test all(abs.(coef(tfit) .- [0.7, 0.3, 0.2]) .< 4 .* stderror(tfit))
    @test occursin("20 listed dyads", sprint(show, MIME"text/plain"(), tfit))

    # tied data: refused by default, fitted under a named policy
    # a coarse clock: an event shares its predecessor's timestamp whenever the
    # two are different dyads (Efron needs the tied cases to be distinct)
    coarse = Event{Float64}[]
    for (k, e) in enumerate(events)
        tie = k > 1 && iseven(k) &&
              (e.sender, e.receiver) != (events[k - 1].sender, events[k - 1].receiver)
        push!(coarse, Event(e.sender, e.receiver, tie ? coarse[end].time : Float64(k)))
    end
    @test count(k -> coarse[k].time == coarse[k - 1].time, 2:400) > 50
    @test_throws ArgumentError fit_revel(coarse, stats, n)
    br = fit_revel(coarse, stats, n; ties=:breslow)
    ef = fit_revel(coarse, stats, n; ties=:efron, engine=:design)
    @test NetworkCore.tie_method(br) === :breslow && NetworkCore.tie_method(ef) === :efron
    @test !NetworkCore.is_exact(br)
    @test coef(fit_revel(coarse, stats, n; ties=:breslow, engine=:design)) ≈ coef(br) atol = 1e-8
    @test coef(fit_revel(coarse, stats, n; ties=:efron)) ≈ coef(ef) atol = 1e-8

    @test_throws ArgumentError fit_revel(events, stats, n; model=:cox)
    @test_throws ArgumentError fit_revel(events, stats, n; engine=:glm)
    @test_throws ArgumentError fit_revel(Event{Float64}[], stats, n)
    @test_throws ArgumentError fit_revel(events, [Inertia(), Inertia()], n)
    @test_throws ArgumentError fit_revel(events, stats, n; engine=:stream, riskset=:sender)
    @test_throws ArgumentError fit_revel(events, stats, n; engine=:stream, se=:sandwich)
    @test_throws ArgumentError fit_revel(events, stats, n; engine=:design, cache=:none)
    @test_throws ArgumentError fit_revel(events, stats, n; se=:bootstrap, engine=:design)
    # the timing model refuses what its likelihood cannot honour
    @test_throws ArgumentError fit_revel(events, stats, n; model=:timing, riskset=:sender)
    @test_throws ArgumentError fit_revel(events, stats, n; model=:timing, cases=1:10)
    @test_throws ArgumentError fit_revel(events, stats, n; model=:timing, se=:sandwich)
    @test_throws ArgumentError fit_revel(events, [Inertia(memory=HalfLife(5.0))], n;
                                         model=:timing)
    @test_throws ArgumentError fit_revel(events, [TimeSince(:dyad)], n; model=:timing)
end

@testset "A global covariate is identified only as a moderator" begin
    n = 5
    late = GlobalEffect(t -> t > 150 ? 1.0 : 0.0; name="late")
    truth = [Inertia(transform=:log1p), Interaction(late, Inertia(transform=:log1p))]
    events = simulate_events(truth, [0.4, 1.0], n, 300; rng=Xoshiro(21))
    fit = fit_revel(events, truth, n)
    @test coefnames(fit) == ["log1p(inertia)", "late:log1p(inertia)"]
    @test all(abs.(coef(fit) .- [0.4, 1.0]) .< 4 .* stderror(fit))
    @test coef(fit)[2] / stderror(fit)[2] > 2
    # on its own it is constant across every risk set: refused by both engines,
    # and singular when the design is fitted directly
    for engine in (:stream, :design)
        @test_throws ArgumentError fit_revel(events, [Inertia(transform=:log1p), late], n;
                                             engine=engine)
    end
    design = event_design(events, [Inertia(transform=:log1p), late], n)
    @test (@test_logs (:warn,) match_mode=:any REM.fit_rem(design, ["log1p(inertia)", "late"])).singular
    check = statistic_collinearity(events, [Inertia(), late], n)
    @test check.vif[2] == Inf && isnan(check.correlation[1, 2])
end

@testset "Moderation by refitting: strata, windows, memory profiles" begin
    n = 5
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    events = simulate_events(stats, [1.0, 0.5], n, 240; rng=Xoshiro(5))
    fits = fit_stratified(events, stats, n; by=e -> e.time <= 120 ? :early : :late)
    @test sort(collect(keys(fits))) == [:early, :late]
    @test nobs(fits[:early]) == 120 && nobs(fits[:late]) == 120
    # the late stratum is conditioned on the early events
    alone = fit_revel(events[121:240], stats, n)
    @test coef(fits[:late]) != coef(alone)
    @test coef(fits[:early]) ≈ coef(fit_revel(events[1:120], stats, n)) atol = 1e-8
    # a stratified fit is the fully interacted product-term model: the two
    # stratum log-likelihoods add up to it
    late = GlobalEffect(t -> t > 120 ? 1.0 : 0.0; name="late")
    full = fit_revel(events, [stats; [Interaction(late, s) for s in stats]], n)
    @test loglikelihood(fits[:early]) + loglikelihood(fits[:late]) ≈
          loglikelihood(full) atol = 1e-6
    @test coef(full)[1:2] ≈ coef(fits[:early]) atol = 1e-6
    @test coef(full)[1:2] .+ coef(full)[3:4] ≈ coef(fits[:late]) atol = 1e-6
    @test_throws ArgumentError fit_stratified(events, stats, n; by=:eventtype)

    table = compare_coefficients(fits)
    @test names(table) == ["group", "term", "estimate", "std_error", "z", "n_events"]
    @test size(table, 1) == 4
    @test table.z ≈ table.estimate ./ table.std_error
    @test_throws ArgumentError compare_coefficients(1)

    path = fit_moving_window(events, stats, n; width=80.0)
    @test [w.n_events for w in path] == [80, 80, 80]
    @test [w.from for w in path] == [1.0, 81.0, 161.0]
    @test size(compare_coefficients(path), 1) == 6
    overlapping = fit_moving_window(events, stats, n; width=80.0, step=40.0)
    @test length(overlapping) == 6
    @test_throws ArgumentError fit_moving_window(events, stats, n; width=0.0)
    @test_throws ArgumentError fit_moving_window(events, stats, n; width=2.0)

    # profile likelihood of a half-life: simulated at 5, the profile peaks there
    truth = [Inertia(memory=HalfLife(5.0), transform=:log1p)]
    ev = simulate_events(truth, [1.5], 6, 600; rng=Xoshiro(8))
    profile = profile_memory(ev, 6, [0.5, 5.0, 50.0, 500.0]) do h
        [Inertia(memory=HalfLife(h), transform=:log1p)]
    end
    @test names(profile) == ["value", "loglik", "aic", "bic", "converged", "in_ci", "best"]
    @test profile.value[profile.best] == [5.0]
    @test count(profile.best) == 1 && all(profile.converged)
    @test_throws ArgumentError profile_memory(h -> [Inertia()], ev, 6, Float64[])
end

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

@testset "simulate_events" begin
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    a = simulate_events(stats, [1.0, 0.5], 5, 100; rng=Xoshiro(1))
    Random.seed!(7); b = simulate_events(stats, [1.0, 0.5], 5, 100; rng=Xoshiro(1))
    @test [(e.sender, e.receiver, e.time) for e in a] ==
          [(e.sender, e.receiver, e.time) for e in b]          # rng is the only randomness
    @test [e.time for e in a] == 1.0:100.0
    @test all(e -> e.sender != e.receiver && 1 <= e.sender <= 5, a)
    @test simulate_events(stats, [1.0, 0.5], 5, 0; rng=Xoshiro(1)) == Event{Float64}[]

    times = cumsum(fill(0.5, 40))
    given = simulate_events(stats, [1.0, 0.5], 5, 40; times=times, rng=Xoshiro(1))
    @test [e.time for e in given] == times
    timed = simulate_events(stats, [1.0, 0.5], 5, 40; baseline=0.1, rng=Xoshiro(1))
    @test issorted(e.time for e in timed) && timed[1].time > 0

    marked = simulate_events(stats, [1.0, 0.5], 5, 3; rng=Xoshiro(1),
                             eventtype=[:a, :b, :a], weights=[1.0, 2.0, 3.0])
    @test [e.eventtype for e in marked] == [:a, :b, :a]
    @test [e.weight for e in marked] == [1.0, 2.0, 3.0]

    seeded = simulate_events([Inertia()], [50.0], 4, 5; rng=Xoshiro(1),
                             history=[Event(3, 4, 0.5)])
    @test all(e -> (e.sender, e.receiver) == (3, 4), seeded)   # overwhelming inertia

    choices = simulate_events(stats, [1.0, 0.5], 5, 6; riskset=:sender,
                              senders=[1, 2, 3, 4, 5, 1], rng=Xoshiro(1))
    @test [e.sender for e in choices] == [1, 2, 3, 4, 5, 1]
    und = simulate_events([Inertia(symmetric=true)], [0.1], 5, 30; directed=false,
                          rng=Xoshiro(1))
    @test all(e -> e.sender < e.receiver, und)

    @test_throws ArgumentError simulate_events(stats, [1.0], 5, 10)
    @test_throws ArgumentError simulate_events(stats, [1.0, 0.5], 5, 10; times=[1.0])
    @test_throws ArgumentError simulate_events(stats, [1.0, 0.5], 5, 2; times=[2.0, 1.0])
    @test_throws ArgumentError simulate_events(stats, [1.0, 0.5], 5, 2; times=[1.0, 2.0],
                                               baseline=1.0)
    @test_throws ArgumentError simulate_events(stats, [1.0, 0.5], 5, 2; baseline=-1.0)
    @test_throws ArgumentError simulate_events([Inertia(memory=HalfLife(2.0))], [1.0], 5, 2;
                                               baseline=1.0)
    @test_throws ArgumentError simulate_events(stats, [1.0, 0.5], 5, 2; riskset=:sender)
    @test_throws ArgumentError simulate_events(stats, [1.0, 0.5], 5, 2; riskset=:active)
    @test_throws ArgumentError simulate_events(stats, [1.0, 0.5], 5, 2; eventtype=[:a])
    # raw cumulative statistics with a large coefficient explode, and say so
    @test_throws ArgumentError simulate_events([Inertia(transform=x -> exp(50x))], [50.0],
                                               4, 50; rng=Xoshiro(1))
end

# ----------------------------------------------------------------------------
# Goodness of fit and diagnostics
# ----------------------------------------------------------------------------

@testset "Event diagnostics and prediction" begin
    n = 6
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    events = simulate_events(stats, [1.0, 0.5], n, 300; rng=Xoshiro(1))
    fit = fit_revel(events, stats, n)
    d = event_diagnostics(fit)
    @test size(d, 1) == 300
    @test names(d) == ["event_index", "time", "sender", "receiver", "risk_set_size",
                       "probability", "rank", "n_above", "n_tied", "reciprocal_rank",
                       "rank_fraction", "deviance_residual", "null_residual", "surprise"]
    @test d.rank == d.n_above .+ (d.n_tied .+ 1) ./ 2
    @test all(0 .< d.probability .<= 1)
    @test all(1 .<= d.rank .<= 30)
    # the deviance residuals add up to the fitted deviance
    @test sum(d.deviance_residual) ≈ -2loglikelihood(fit)
    @test all(d.null_residual .≈ 2log(30))
    # before any event every dyad is equally likely and shares the average rank;
    # its expected reciprocal rank under random tie-breaking is the mean of 1/k
    @test d.probability[1] ≈ 1 / 30 && d.rank[1] == 15.5
    @test d.n_above[1] == 0 && d.n_tied[1] == 30
    @test d.reciprocal_rank[1] ≈ sum(1 ./ (1:30)) / 30
    @test d.surprise ≈ -log2.(d.probability)

    s = prediction_summary(fit; ks=(1, 5, 0.2))
    @test s.n_events == 300 && s.ks == [1, 5, 0.2]
    # recall credits a tie as a random tie-break would: the expected recall
    expected(cut) = sum(clamp.((cut .- d.n_above) ./ d.n_tied, 0, 1)) / 300
    @test issorted(s.recall) && s.recall[1] ≈ expected(1)
    @test s.recall[3] ≈ expected(6)                           # 20 % of 30 dyads
    # … so the event tied with the whole risk set earns 1/30, not 0, at k = 1
    @test expected(1) > count(<=(1), d.rank) / 300
    @test s.mean_reciprocal_rank ≈ mean(d.reciprocal_rank)
    @test s.deviance ≈ -2loglikelihood(fit)
    @test s.null_deviance ≈ 600log(30)
    @test 0 < s.pseudo_r2 < 1
    @test 1 <= s.perplexity <= 30
    @test s.mean_rank ≈ mean(d.rank)
    @test_throws ArgumentError prediction_summary(fit; ks=(0,))
    @test_throws ArgumentError prediction_summary(fit; ks=(1.5,))

    # the design engine scores identically
    @test event_diagnostics(fit_revel(events, stats, n; engine=:design)).probability ≈
          d.probability

    # out of sample: fit on the first 200 events, score the last 100
    train = fit_revel(events, stats, n; cases=1:200)
    held = event_diagnostics(train; cases=201:300)
    @test held.event_index == 201:300
    inside = event_diagnostics(train)
    @test inside.event_index == 1:200
    @test prediction_summary(train; cases=201:300).n_events == 100
    # the receiver-choice fit is scored on its own choice sets
    choice = fit_receiver_choice(events, stats, n)
    @test all(event_diagnostics(choice).risk_set_size .== n - 1)
    # a timing fit is scored on its event-choice part
    timing = fit_revel(events, stats, n; model=:timing)
    @test size(event_diagnostics(timing), 1) == 300
end

@testset "Score processes and score tests" begin
    n = 6
    truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    events = simulate_events(truth, [1.0, 1.0], n, 400; rng=Xoshiro(1))
    fit = fit_revel(events, truth, n)
    table, process = score_process_test(fit; n_sim=300, rng=Xoshiro(2), return_process=true)
    @test table.term == ["log1p(inertia)", "log1p(reciprocity)", "GLOBAL"]
    @test names(table) == ["term", "statistic", "p_value", "p_kolmogorov", "at_event"]
    @test all(0 .< table.p_value .<= 1) && all(0 .<= table.p_kolmogorov .<= 1)
    @test size(process) == (400, 2)
    # the score is zero at the maximum: the process returns to the origin
    @test maximum(abs, process[end, :]) < 1e-6
    @test table.statistic[1] == maximum(abs, process[:, 1])
    @test table.statistic[3] == maximum(table.statistic[1:2])
    # correctly specified: no evidence against constant effects
    @test table.p_value[3] > 0.05
    @test score_process_test(fit; n_sim=50, rng=Xoshiro(3)) ==
          score_process_test(fit; n_sim=50, rng=Xoshiro(3))

    # an effect that switches on half-way through is detected ...
    late = GlobalEffect(t -> t > 200 ? 1.0 : 0.0; name="late")
    shifting = [Inertia(transform=:log1p),
                Interaction(late, Inertia(transform=:log1p); name="late_inertia")]
    ev2 = simulate_events(shifting, [0.0, 2.0], n, 400; rng=Xoshiro(4))
    wrong = fit_revel(ev2, [Inertia(transform=:log1p)], n)
    drift = score_process_test(wrong; n_sim=300, rng=Xoshiro(5))
    @test drift.p_value[1] < 0.01
    # ... and located near the change
    @test 120 <= drift.at_event[1] <= 280

    # the score test finds the omitted effect without refitting
    small = fit_revel(events, truth[1:1], n)
    screen = score_test(small, [truth[2], OTP(transform=:log1p)])
    @test screen.term == ["log1p(reciprocity)", "log1p(otp)"]
    @test screen.p_value[1] < 1e-6 && screen.direction[1] == 1
    @test all(screen.variance .> 0) && all(screen.chisq .>= 0)
    # for one candidate it is the score statistic of the larger model at the
    # restricted estimate: U' I⁻¹ U with the full information matrix
    bigger = [truth[1], truth[2]]
    U = zeros(2); info = zeros(2, 2)
    θ0 = [coef(small)[1], 0.0]
    each_risk_set(events, bigger, n) do v
        η = v.X * θ0
        p = exp.(η .- maximum(η)); p ./= sum(p)
        xbar = v.X' * p
        U .+= v.X[v.case, :] .- xbar
        info .+= v.X' * (p .* v.X) .- xbar * xbar'
    end
    @test screen.chisq[1] ≈ dot(U, info \ U) rtol = 1e-8
    # ... and close to the likelihood-ratio statistic it approximates
    lr = 2 * (loglikelihood(fit) - loglikelihood(small))
    @test 0.5 < screen.chisq[1] / lr < 2
    @test score_test(small, truth[2]).chisq == screen.chisq[1:1]
    # a candidate that duplicates a fitted effect has nothing left to explain
    again = score_test(small, [Inertia(transform=:log1p, name="again")])
    @test isnan(again.chisq[1]) && isnan(again.p_value[1])
    @test_throws ArgumentError score_test(small, AbstractStatistic[])
    @test_throws ArgumentError score_test(small, [Inertia(transform=:log1p)])

    timing = fit_revel(events, truth, n; model=:timing)
    @test_throws ArgumentError score_process_test(timing)
    @test_throws ArgumentError score_test(timing, [OTP()])
    sampled = fit_revel(events, truth, n; n_controls=5, rng=Xoshiro(1))
    @test_throws ArgumentError score_process_test(sampled)
    @test_throws ArgumentError score_process_test(fit; n_sim=0)
end

@testset "Sequence metrics (hand-computed)" begin
    events = [Event(1, 2, 1.0), Event(2, 1, 4.0), Event(1, 2, 5.0), Event(2, 3, 6.0),
              Event(1, 3, 8.0), Event(3, 1, 8.5)]
    shares = mechanism_shares(events)
    @test keys(shares) == (:repetition, :reciprocation, :transitive, :cyclic,
                           :shared_out, :shared_in)
    @test shares.repetition == 1 / 6         # the second 1 → 2
    @test shares.reciprocation == 3 / 6      # 2 → 1, 1 → 2, 3 → 1
    @test shares.transitive == 1 / 6         # 1 → 3 closes 1 → 2 → 3
    @test shares.cyclic == 1 / 6             # 3 → 1 closes 1 → 2 → 3 → 1
    @test shares.shared_out == 0.0           # no two actors ever sent to a common third
    @test shares.shared_in == 2 / 6          # 2 had sent to both ends of 1 → 3 and 3 → 1

    @test closing_times(events) == [3.0, 1.0, 0.5]
    @test closing_times(events; reference=:first) == [3.0, 1.0, 0.5]
    @test closing_times(events; mechanism=:repetition) == [4.0]
    @test closing_times(events; clock=:order) == [1.0, 1.0, 1.0]
    # 1 → 3 at t = 8 closes 1 → 2 (last at 5) → 3 (at 6): the two-path exists from 6
    @test closing_times(events; mechanism=:transitive) == [2.0]
    @test closing_times(events; mechanism=:cyclic) == [2.5]
    more = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 1, 7.0)]
    @test closing_times(more; reference=:last) == [1.0, 1.0, 4.0]
    @test closing_times(more; reference=:first) == [1.0, 1.0, 6.0]   # Amati et al.
    @test closing_times([Event(1, 2, 1.0)]) == Float64[]
    @test_throws ArgumentError closing_times(events; mechanism=:gossip)
    @test_throws ArgumentError closing_times(events; reference=:median)
    @test_throws ArgumentError closing_times(events; clock=:wall)
    @test_throws ArgumentError mechanism_shares(Event{Float64}[])
end

@testset "Simulation-based goodness of fit" begin
    n = 6
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    events = simulate_events(stats, [1.0, 0.8], n, 200; rng=Xoshiro(1))
    fit = fit_revel(events, stats, n)
    result = gof(fit; n_sim=40, rng=Xoshiro(2))
    @test result isa NetworkCore.GOFResult
    @test n_simulations(result) == 40
    @test [s.name for s in result.statistics] ==
          ["mechanism shares", "degree concentration", "closing times (events)"]
    @test result.statistics[1].observed ≈ collect(values(mechanism_shares(events)))
    @test all(0 .< s.p_values[k] <= 1 for s in result.statistics for k in eachindex(s.p_values))
    Random.seed!(1); r1 = gof(fit; n_sim=10, rng=Xoshiro(3))
    Random.seed!(2); r2 = gof(fit; n_sim=10, rng=Xoshiro(3))
    @test r1.statistics[1].simulated == r2.statistics[1].simulated
    @test occursin("Goodness-of-fit", sprint(show, result))

    @test 0 < result.p_overall <= 1

    # A model without reciprocity cannot reproduce data generated with strong
    # reciprocity, and the joint (Mahalanobis) test says so. Calibrated when this
    # test was written, over fresh seeds: the joint test rejected the poor model
    # in 40 of 40 data sets and the true one in 7 of 320 (2 %), so asking for two
    # rejections (acceptances) out of three data sets fails by chance about once
    # in 300 runs — no seed is tuned. (The per-statistic p-values are pointwise
    # and far weaker: a single auxiliary caught the omission in 19 of 40.)
    rejected = map(1:3) do k
        strong = simulate_events(stats, [0.2, 2.5], n, 300; rng=Xoshiro(40 + k))
        poor = gof(fit_revel(strong, stats[1:1], n); n_sim=100, rng=Xoshiro(50 + k))
        good = gof(fit_revel(strong, stats, n); n_sim=100, rng=Xoshiro(60 + k))
        (poor.p_overall <= 0.05, good.p_overall <= 0.05)
    end
    @test count(first, rejected) >= 2
    @test count(last, rejected) <= 1

    own = gof(fit; n_sim=10, rng=Xoshiro(6),
              auxiliary=[("events by actor 1", ["sent"],
                          ev -> [count(e -> e.sender == 1, ev)])])
    @test own.statistics[1].observed == [count(e -> e.sender == 1, events)]

    # conditional on the observed senders for a choice fit; own clock for timing
    @test n_simulations(gof(fit_receiver_choice(events, stats, n); n_sim=5,
                            rng=Xoshiro(7))) == 5
    @test n_simulations(gof(fit_revel(events, stats, n; model=:timing); n_sim=5,
                            rng=Xoshiro(8))) == 5
    @test_throws ArgumentError gof(fit_revel(events, stats, n; cases=1:100))
    @test_throws ArgumentError gof(fit; n_sim=0)
    grow = (m, ev) -> [(s, r) for s in 1:n for r in 1:n if s != r]
    @test_throws ArgumentError gof(fit_revel(events, stats, n; riskset=grow))
end

@testset "Collinearity among statistics" begin
    rng = Xoshiro(1)
    n = 6
    events = random_events(rng, n, 150)
    check = statistic_collinearity(events, [Inertia(), OutdegreeSender(), OTP()], n)
    @test check.names == ["inertia", "outdegreeSender", "otp"]
    @test check.correlation ≈ check.correlation' && all(diag(check.correlation) .≈ 1)
    @test all(check.vif .>= 1) && isfinite(check.condition_number)
    # inertia is part of out-degree: they are positively related within risk sets
    @test check.correlation[1, 2] > 0
    # an exactly collinear specification
    x = collect(1.0:n)
    exact = statistic_collinearity(events, [SendEffect(x), ReceiveEffect(x), SumEffect(x)], n)
    @test all(exact.vif .== Inf) && exact.condition_number == Inf
    # in a receiver-choice set the sender's covariate does not vary at all
    choice = statistic_collinearity(events, [SendEffect(x), ReceiveEffect(x)], n;
                                    riskset=:sender)
    @test choice.vif[1] == Inf && choice.vif[2] ≈ 1.0
    @test isnan(choice.correlation[1, 2])
    # the within-risk-set correlation is the one a by-hand computation gives
    design = event_design(events, [Inertia(), OutdegreeSender()], n)
    centred = transform(groupby(design, :stratum),
                        :inertia => (v -> v .- mean(v)) => :a,
                        :outdegreeSender => (v -> v .- mean(v)) => :b)
    @test statistic_collinearity(events, [Inertia(), OutdegreeSender()], n).correlation[1, 2] ≈
          cor(centred.a, centred.b)
end

# ----------------------------------------------------------------------------
# Regressions: scale, thread safety, refusals, disclosures
# ----------------------------------------------------------------------------

@testset "Layers: sparse storage for large actor IDs" begin
    # The same sequence on actors 1–4 and on actors shifted beyond the dense
    # threshold (the dyad → slot map becomes a hash map): identical statistics
    rng = Xoshiro(31)
    small = random_events(rng, 4, 60)
    shift = Revel._DENSE_MAX + 1000
    big = [Event(e.sender + shift, e.receiver + shift, e.time) for e in small]
    for mk in (() -> Inertia(), () -> Reciprocation(memory=Window(5.0)),
               () -> OTP(memory=HalfLife(3.0)), () -> RecencyRank(:send),
               () -> OutdegreeSender(scaling=:prop), () -> TimeSince(:pair))
        a, b = mk(), mk()
        hs, hb = build_history(small), build_history(big)
        t = small[end].time + 1
        @test [compute(a, hs, i, j, t) for i in 1:4, j in 1:4 if i != j] ==
              [compute(b, hb, i + shift, j + shift, t) for i in 1:4, j in 1:4 if i != j]
    end
    # memory grows with the dyads that have a history, not with the largest ID
    stat = Inertia()
    compute(stat, build_history(big), 1 + shift, 2 + shift, 1e6)
    @test Base.summarysize(stat) < 2^20
    @test_throws ArgumentError compute(Inertia(), build_history([Event(2^31, 1, 1.0)]),
                                       1, 2, 2.0)
end

@testset "Layers: private copies, shared layers, thread safety" begin
    n = 6
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p), OTP()]
    sequences = [simulate_events(stats[1:2], [0.8, 0.5], n, 120; rng=Xoshiro(70 + k))
                 for k in 1:8]
    serial = [coef(fit_revel(ev, stats, n)) for ev in sequences]
    # the fit worked on copies: the user's statistics hold no cache
    @test all(isempty(s.layer.states) for s in stats[1:2]) && isempty(stats[3].leg1.states)
    # one specification fitted from several tasks at once
    parallel = fetch.([Threads.@spawn coef(fit_revel(ev, stats, n)) for ev in sequences])
    @test parallel == serial
    parallel = fetch.([Threads.@spawn coef(fit_revel(ev, stats, n; engine=:design))
                       for ev in sequences])
    @test parallel ≈ serial atol = 1e-8
    # a copy merges layers with the same keywords (one index of the history)
    copies = Revel._fresh(stats)
    @test copies[1].layer === copies[2].layer === copies[3].leg1
    @test copies[1].layer !== stats[1].layer
    # … but keeps apart layers that differ
    other = Revel._fresh([Inertia(), Inertia(memory=Window(2.0), name="w")])
    @test other[1].layer !== other[2].layer
end

# A statistic that counts its evaluations (the counter is global: the fitters
# evaluate copies of the statistics)
const EVALUATIONS = Ref(0)
struct CountingStatistic <: AbstractRevelStatistic
    label::String
end
Revel._value(::CountingStatistic, events, s::Int, r::Int, t::Float64) =
    (EVALUATIONS[] += 1; Float64(s == 1))
Revel._uses_history(::CountingStatistic) = false

@testset "Sampled controls cost the rows drawn, not the risk set" begin
    n = 12
    events = random_events(Xoshiro(2), n, 50)
    EVALUATIONS[] = 0
    event_design(events, [CountingStatistic("c")], n; n_controls=3, rng=Xoshiro(1))
    @test EVALUATIONS[] == 50 * 4
    EVALUATIONS[] = 0
    full = event_design(events, [CountingStatistic("c")], n)
    @test EVALUATIONS[] == 50 * n * (n - 1)
    # the design is the same frame either way: case first, then the controls
    sampled = event_design(events, [Inertia(), Reciprocation()], n; n_controls=3,
                           rng=Xoshiro(1))
    @test size(sampled, 1) == 50 * 4 && all(sampled.risk_set_size .== n * (n - 1))
    @test all(sampled.sampling_prob .≈ 3 / (n * (n - 1) - 1))
    for g in groupby(sampled, :stratum)
        @test count(g.is_event) == 1 && g.is_event[1]
        @test allunique(zip(g.sender, g.receiver))
    end
    # a sampled row carries the statistic the full design gives that dyad
    key(df) = Dict((r.event_index, r.sender, r.receiver) => (r.inertia, r.reciprocity)
                   for r in eachrow(df))
    fullkey = key(event_design(events, [Inertia(), Reciprocation()], n))
    @test all(fullkey[k] == v for (k, v) in key(sampled))
end

@testset "Refusals: inputs that would give a silently wrong fit" begin
    n = 4
    events = random_events(Xoshiro(3), n, 40)
    stats = [Inertia()]
    # a REM.jl statistic has no history interface
    err = try fit_revel(events, [REM.Repetition()], n) catch e e end
    @test err isa ArgumentError && occursin("REM.jl statistic", err.msg)
    # a self-loop, a non-finite time
    @test_throws ArgumentError fit_revel([events; Event(2, 2, 100.0)], stats, n)
    @test_throws ArgumentError fit_revel([events; Event(1, 2, NaN)], stats, n)
    @test_throws ArgumentError event_design([events; Event(1, 2, Inf)], stats, n)
    # a sender outside the actors of a receiver-choice risk set
    @test_throws ArgumentError event_design([Event(0, 1, 0.5); events], stats, n;
                                            riskset=:sender)
    @test_throws ArgumentError simulate_events(stats, [0.0], n, 2; riskset=:sender,
                                               senders=[1, 9])
    # a function risk set that lists a dyad twice
    twice = (m, e) -> [(e.sender, e.receiver), (e.sender, e.receiver), (1, 2), (2, 1)]
    @test_throws ArgumentError event_design(events, stats, n; riskset=twice)
    # keep= reads the event type, which REM's event log does not carry
    state = REM.EventNetworkState{Float64}(); state.keep_history = true
    REM.update!(state, events[1]); state.current_time = 10.0
    @test_throws ArgumentError compute(Inertia(keep=(s, r, t, w, ty) -> ty === :email),
                                       state, 1, 2)
    # ranks and recencies take no memory kernel
    @test_throws ArgumentError RecencyRank(:send; memory=Window(2.0))
    @test_throws ArgumentError TimeSince(:dyad; memory=HalfLife(2.0))
    @test_throws ArgumentError RecencyRank(; layer=EventLayer(memory=Window(2.0)))
    # a stratifier with no key for an event
    err = try fit_stratified(events, stats, n; by=e -> e.time < 5 ? missing : :a) catch e e end
    @test err isa ArgumentError && occursin("stratum key", err.msg)
    # diagnostics of a fit that did not converge
    capped = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        fit_revel(events, [Inertia(), Reciprocation()], n; maxiter=1, engine=:design)
    end
    if !capped.fit.converged
        @test_throws ArgumentError score_test(capped, [OTP()])
        @test_throws ArgumentError score_process_test(capped)
        @test_throws ArgumentError gof(capped)
        @test_throws ArgumentError event_diagnostics(capped)
    end
end

@testset "Standardized standardises over the model's risk set" begin
    n = 5
    stats = [Inertia(transform=:log1p)]
    events = simulate_events(stats, [1.0], n, 80; rng=Xoshiro(4))
    @test fit_revel(events, [Standardized(Inertia(), n)], n).fit.converged
    for bad in [(Standardized(Inertia(), n + 1), (;)),
                (Standardized(Inertia(), n), (; directed=false)),
                (Standardized(Inertia(), n), (; riskset=:sender)),
                (Standardized(Inertia(), n), (; riskset=:active))]
        @test_throws ArgumentError fit_revel(events, [bad[1]], n; bad[2]...)
    end
    # an explicit set of dyads: the two-mode risk set
    users, items = 1:2, 3:5
    tm = simulate_events([Inertia(transform=:log1p)], [1.0], n, 60;
                         riskset=two_mode_dyads(users, items), rng=Xoshiro(5))
    set = two_mode_dyads(users, items)
    z = Standardized(Inertia(), set)
    fz = fit_revel(tm, [z], n; riskset=set)
    @test fz.fit.converged
    design = event_design(tm, [z], n; riskset=set)
    # z-scored within every risk set: mean 0 over the set (population sd)
    @test all(abs(mean(g[!, "std(inertia)"])) < 1e-12 for g in groupby(design, :stratum))
    @test_throws ArgumentError fit_revel(tm, [z], n; riskset=set[1:4])
    # the wrapper accepts a fractional time on an integer clock, like a bare statistic
    hi = build_history([Event(1, 2, 1), Event(1, 2, 2)])
    @test compute(Standardized(Inertia(), 3), hi, 1, 2, 2.5) ≈
          compute(Standardized(Inertia(), 3), hi, 1, 2, 3)
    @test compute(Interaction(Inertia(), Inertia(name="b")), hi, 1, 2, 2.5) == 4.0
    @test compute(Transformed(Inertia(), log1p), hi, 1, 2, 2.5) == log1p(2.0)
end

@testset "Timing model: statistics on the event clock are admissible" begin
    n = 5
    events = simulate_events([Inertia(transform=:log1p)], [0.8], n, 80; baseline=0.5,
                             rng=Xoshiro(6))
    for stat in (Inertia(memory=HalfLife(5.0), clock=:order, transform=:log1p),
                 TimeSince(:dyad; clock=:order))
        @test Revel.is_interval_constant(stat)
        @test fit_revel(events, [stat], n; model=:timing).fit.converged
    end
    @test !Revel.is_interval_constant(Inertia(memory=HalfLife(5.0)))
    @test_throws ArgumentError fit_revel(events, [TimeSince(:dyad)], n; model=:timing)
    # the fit records the start of its clock, and gof simulates from it
    late = [Event(e.sender, e.receiver, e.time + 100) for e in events]
    f = fit_revel(late, [Inertia(transform=:log1p)], n; model=:timing, t0=90.0)
    @test f.t0 == 90.0 && f.t_end === nothing
    g = gof(f; n_sim=5, rng=Xoshiro(7))
    @test [s.name for s in g.statistics][end] == "waiting times"
end

@testset "Simulation freezes the history across a tie block when asked" begin
    stats = [Inertia(transform=:log1p)]
    times = fill(1.0, 12)
    ordered = simulate_events(stats, [30.0], 5, 12; times=times, rng=Xoshiro(8))
    frozen = simulate_events(stats, [30.0], 5, 12; times=times, ties=:breslow,
                             rng=Xoshiro(8))
    # with the history growing inside the block, the first dyad repeats; frozen,
    # every draw is from the empty history, so dyads vary
    @test length(unique((e.sender, e.receiver) for e in ordered)) == 1
    @test length(unique((e.sender, e.receiver) for e in frozen)) > 1
    @test_throws ArgumentError simulate_events(stats, [1.0], 5, 2; ties=:batch)
    @test_throws ArgumentError simulate_events(stats, [1.0], 5, 2; t0=1.0)
end

@testset "Diagnostics: refusals and disclosures" begin
    n = 6
    truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    events = simulate_events(truth, [1.0, 0.8], n, 300; rng=Xoshiro(9))
    fit = fit_revel(events, truth, n)
    # GLOBAL: the largest standardised supremum, resampled as a maximum
    t = score_process_test(fit; n_sim=200, rng=Xoshiro(1))
    @test t.statistic[end] == maximum(t.statistic[1:end-1])
    @test 0 < t.p_value[end] <= 1
    @test all(t.p_kolmogorov .> 0)
    # the score test of a candidate close to — but not inside — the model's span
    small = fit_revel(events, truth[1:1], n)
    near = score_test(small, [Inertia(memory=HalfLife(1e6), transform=:log1p,
                                      name="nearly")])
    @test isfinite(near.chisq[1]) && 0 < near.residual_share[1] < 1e-3
    same = score_test(small, [Inertia(transform=:log1p, name="copy")])
    @test isnan(same.chisq[1])
    # collinearity: an exact duplicate makes only the duplicated pair infinite
    dup = statistic_collinearity(events, [Inertia(), Inertia(name="c"), OTP()], n)
    @test dup.vif[1] == dup.vif[2] == Inf && isfinite(dup.vif[3])
    at_fit = statistic_collinearity(fit)
    info = inv(vcov(fit))
    @test at_fit.vif ≈ diag(info) .* diag(inv(info)) rtol = 1e-6
    # compare_coefficients: Wald tests of stratum differences
    fits = fit_stratified(events, truth, n; by=e -> e.time <= 150 ? :early : :late)
    table = compare_coefficients(fits; reference=:early)
    late = table[table.group .== :late, :]
    early = table[table.group .== :early, :]
    @test late.difference ≈ late.estimate .- early.estimate
    @test late.se_difference ≈ sqrt.(late.std_error .^ 2 .+ early.std_error .^ 2)
    @test all(ismissing, early.p_difference) && all(0 .< late.p_difference .<= 1)
    @test_throws ArgumentError compare_coefficients(fits; reference=:middle)
    windows = fit_moving_window(events, truth, n; width=150.0, step=75.0)
    @test_throws ArgumentError compare_coefficients(windows; reference=windows[1].from)
    # windows start at t_first + k * step exactly (no accumulated drift)
    tiny = fit_moving_window(events, truth[1:1], n; width=30.0, step=0.3, min_events=20)
    froms = [w.from for w in tiny]
    ks = round.(Int, (froms .- events[1].time) ./ 0.3)
    @test froms == events[1].time .+ ks .* 0.3
    # profile_memory: one draw of controls for every grid value
    prof = profile_memory(events, n, [5.0, 5.0, 5.0]; n_controls=5, rng=Xoshiro(3)) do h
        [Inertia(memory=HalfLife(h), transform=:log1p)]
    end
    @test prof.loglik[1] == prof.loglik[2] == prof.loglik[3]
    @test count(prof.best) == 1 && all(prof.in_ci)
    fit5 = fit_revel(events, [Inertia(memory=HalfLife(5.0), transform=:log1p)], n)
    prof = profile_memory(events, n, [5.0]) do h
        [Inertia(memory=HalfLife(h), transform=:log1p)]
    end
    @test prof.aic[1] ≈ aic(fit5) + 2
    # a failing grid value is recorded, not fatal
    failing = @test_logs (:warn, r"failed") match_mode=:any profile_memory(
        h -> [Inertia(memory=HalfLife(h), transform=:log1p)], events, n, [5.0, -1.0])
    @test isnan(failing.loglik[2]) && !failing.converged[2] && failing.best[1]
end

@testset "Sequence measures: definitions and edge cases" begin
    h = build_history([Event(1, 2, 1.0), Event(1, 2, 2.0), Event(1, 3, 3.0),
                       Event(3, 2, 4.0)])
    # intensity: events per distinct partner (Vu et al. 2017)
    @test compute(OutdegreeSender(measure=:intensity), h, 1, 4, 5.0) == 1.5
    @test compute(OutdegreeSender(measure=:intensity), h, 4, 1, 5.0) == 0.0
    @test name(OutdegreeSender(measure=:intensity)) == "outdegreeSender.intensity"
    @test_throws ArgumentError OutdegreeSender(measure=:intensity, scaling=:prop)
    @test compute(DegreeAssortativity(measure=:partners), h, 1, 2, 5.0) == 2.0 * 2.0
    # the harmonic combination of two legs: 2ab/(a + b)
    @test compute(OTP(combine=:harmonic), h, 1, 2, 5.0) ≈ 2 * 1 * 1 / 2
    hh = build_history([Event(1, 3, 1.0), Event(1, 3, 2.0), Event(1, 3, 2.5), Event(3, 2, 3.0)])
    @test compute(OTP(combine=:harmonic), hh, 1, 2, 4.0) ≈ 2 * 3 * 1 / 4
    # tertius diversity: Shannon entropy of the neighbours' categories
    party = [:x, :green, :red, :x, :green]
    ht = build_history([Event(2, 4, 1.0), Event(3, 4, 2.0), Event(5, 4, 3.0)])
    p = [2 / 3, 1 / 3]
    @test compute(TertiusEffect(party; aggregate=:entropy), ht, 1, 4, 4.0) ≈ -sum(p .* log.(p))
    @test_throws ArgumentError TertiusEffect([1.0, 2.0, 3.0]; aggregate=:entropy)
    @test_throws ArgumentError TertiusEffect(party; aggregate=:mean)
    @test_throws ArgumentError TertiusEffect(party; aggregate=:entropy, difference=true)
    # :sd without cancellation for large values with a small spread
    big = [0.0, 1e9, 1e9 + 1, 0.0, 0.0]
    hb = build_history([Event(2, 4, 1.0), Event(3, 4, 2.0)])
    @test compute(TertiusEffect(big; aggregate=:sd), hb, 1, 4, 3.0) ≈ 0.5
    # a similarity function sees the categories, not their codes
    seen = Any[]
    sim = (a, b) -> (push!(seen, (a, b)); 1.0)
    compute(MatchedDegree(party; similarity=sim), ht, 1, 4, 4.0)
    @test all(x -> x[1] isa Symbol && x[2] isa Symbol, seen)
    # split_by_type takes a single type
    @test name.(split_by_type(Reciprocation, :praise)) == ["reciprocity[types=praise]"]
end

@testset "a statistic reads only the past of its evaluation time" begin
    h = build_history([Event(1, 2, 1.0), Event(1, 2, 3.0), Event(2, 1, 4.0)])
    @test compute(Inertia(), h, 1, 2, 2.0) == 1.0            # the event at 3 is later
    @test compute(Inertia(memory=HalfLife(1.0)), h, 1, 2, 2.0) == 0.5
    @test compute(Inertia(memory=Window(5.0)), h, 1, 2, 2.0) == 1.0
    @test compute(TimeSince(:dyad; transform=identity), h, 1, 2, 2.0) == 1.0
    @test compute(PShiftABAB(), h, 1, 2, 2.0) == 1.0         # the last event by t = 2
    @test compute(UndirectedPShift(:AB_AB), h, 2, 1, 3.5) == 1.0
    @test compute(Inertia(), h, 1, 2, 0.5) == 0.0
    @test compute(Inertia(), h, 1, 2, 3.0) == 2.0            # an event at t counts
    # the same statistic evaluated forwards, backwards and forwards again
    stat = OTP(memory=HalfLife(2.0))
    vals = [compute(stat, h, 1, 1, t) for t in (5.0, 2.0, 5.0)]
    @test vals[1] == vals[3]
    @test_throws ArgumentError compute(Inertia(), build_history([Event(1, 2, 3.0),
                                                                 Event(2, 1, 1.0)]), 1, 2, 4.0)
    @test_throws ArgumentError compute(Inertia(), build_history([Event(1, 2, NaN)]), 1, 2, 4.0)
    @test_throws ArgumentError compute(Inertia(), h, 1, 2, NaN)
end

@testset "half-life memory on a far-negative clock" begin
    events = [Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 4.0), Event(2, 1, 5.0)]
    shifted = [Event(e.sender, e.receiver, e.time - 1.0e5) for e in events]
    for stat in (Inertia(memory=HalfLife(7.0)), OutdegreeSender(memory=HalfLife(7.0)),
                 Inertia(memory=HalfLife(7.0), scaling=:prop, empty=-1.0),
                 OutdegreeSender(memory=HalfLife(7.0), scaling=:prop, empty=-1.0))
        a = compute(stat, build_history(events), 1, 2, 6.0)
        b = compute(stat, build_history(shifted), 1, 2, 6.0 - 1.0e5)
        @test isfinite(b) && b ≈ a
    end
end

@testset "Separation: every route warns, flags and withholds inference" begin
    # Every event is sent by actor 3, the actor with the largest covariate: the
    # sender effect runs to +Inf (quasi-complete separation: actor 3's two
    # dyads tie). Each route asks NetworkCore's shared verdict on the risk sets
    # it fits and follows the shared policy.
    quasi = [Event(3, isodd(k) ? 1 : 2, Float64(k)) for k in 1:12]
    sender = SendEffect([0.0, 1.0, 2.0]; name="seniority")
    sep_warning = (:warn, r"does not exist \(separation\)")
    routes = (
        () -> fit_revel(quasi, [sender], 3),                         # Revel.fit_obpm
        () -> fit_revel(quasi, [sender], 3; engine=:design),         # REM.fit_rem
        () -> fit_revel(quasi, [sender], 3; model=:timing, t_end=13.0),  # Revel.fit_timing
    )
    for route in routes
        fit = @test_logs sep_warning match_mode=:any route()
        @test !fit.fit.converged
        @test "seniority" in fit.fit.separated
        @test Revel._separated(fit)
        table = coeftable(fit)
        @test all(isnan, table.z_values) && all(isnan, table.p_values)
        @test all(isnan, confint(fit))
        @test any(occursin("separation", a) for a in NetworkCore.approximations(fit))
        # the diagnostics refuse it, naming the statistic and the reason
        err = try event_diagnostics(fit); nothing catch e; e end
        @test err isa ArgumentError && occursin("separation", err.msg) &&
              occursin("seniority", err.msg)
    end
    # compare_coefficients withholds the Wald tests that involve a separated fit
    ok = fit_revel(simulate_events([Inertia(transform=:log1p)], [1.0], 3, 40;
                                   rng=Xoshiro(1)), [sender], 3)
    sep = @test_logs sep_warning match_mode=:any fit_revel(quasi, [sender], 3)
    table = compare_coefficients(["ok" => ok, "sep" => sep]; reference="ok")
    @test isnan(table.z[2]) && !isnan(table.z[1])
    @test isnan(table.p_difference[2])
end

# Kolmogorov distance between the empirical distribution of `p` and U(0, 1).
# Under a calibrated test the p-values of true models are uniform; 40 uniform
# p-values exceed a distance of 0.3 with probability about 1.5e-3. The bound is
# two-sided: p-values piled near 1 (a test that never rejects) fail it as well
# as p-values piled near 0.
function _ks_uniform(p)
    q = sort(p)
    n = length(q)
    return maximum(max(i / n - q[i], q[i] - (i - 1) / n) for i in 1:n)
end

# Two-sided size check over 200 p-values of true models. At the 5 % level the
# rejections must lie in 2:21: under an exact 5 % test a binomial(200, 0.05)
# falls below 2 with probability 4.0e-4 and above 21 with probability 4.8e-4,
# while a test of true size 1 % passes with probability 0.60 and one that never
# rejects never passes. The Kolmogorov distance must stay below 0.15
# (asymptotic false-alarm rate 2.5e-4; the resampling p-values move in steps of
# 0.01). The distance alone would not do: p-values uniform on (0.06, 1) never
# reject and sit at D ≈ 0.06.
_size_two_sided(p) = 2 <= count(<(0.05), p) <= 21 && _ks_uniform(p) < 0.15

@testset "the Kolmogorov p-value is calibrated" begin
    # inertia and reciprocity are strongly correlated within risk sets; the old
    # √(I⁻¹)ₖₖ scaling rejected these true models about a third of the time
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    rejections = 0
    for seed in 1:40
        ev = simulate_events(stats, [1.0, 1.0], 6, 200; rng=Xoshiro(seed))
        t = score_process_test(fit_revel(ev, stats, 6); n_sim=1, rng=Xoshiro(seed))
        rejections += any(t.p_kolmogorov[1:2] .< 0.05)
    end
    @test rejections <= 6
    # … and so are the resampling p-value and the score test, in both directions:
    # 200 data sets from the fitted model (measured: 5 and 13 rejections,
    # D = 0.075 and 0.061)
    spt_p = Float64[]; st_p = Float64[]
    for seed in 101:300
        ev = simulate_events(stats, [1.0, 1.0], 6, 200; rng=Xoshiro(seed))
        f = fit_revel(ev, stats, 6)
        push!(spt_p, score_process_test(f; n_sim=99, rng=Xoshiro(seed)).p_value[end])
        push!(st_p, score_test(f, [OTP(transform=:log1p)]).p_value[1])
    end
    @test _size_two_sided(spt_p) && _size_two_sided(st_p)
    # the check catches a conservative test that the distance alone passes
    conservative = 0.06 .+ 0.94 .* rand(Xoshiro(1), 200)
    @test _ks_uniform(conservative) < 0.15 && !_size_two_sided(conservative)
    @test _size_two_sided(rand(Xoshiro(1), 200))
    # with one statistic the standardised process ends at zero and its scale is
    # the information's
    ev = simulate_events(stats[1:1], [1.0], 6, 200; rng=Xoshiro(3))
    fit = fit_revel(ev, stats[1:1], 6)
    table, process = score_process_test(fit; n_sim=10, rng=Xoshiro(1), return_process=true)
    @test abs(process[end, 1]) < 1e-6
    @test table.statistic[1] ≈ maximum(abs, process[:, 1])
end

@testset "Efron ties in the event diagnostics" begin
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    base = simulate_events(stats, [0.8, 0.6], 6, 300; rng=Xoshiro(3))
    coarse = Event{Float64}[]
    for (k, e) in enumerate(base)
        tie = k > 1 && iseven(k) &&
              (e.sender, e.receiver) != (base[k - 1].sender, base[k - 1].receiver)
        push!(coarse, Event(e.sender, e.receiver, tie ? coarse[end].time : Float64(k)))
    end
    for engine in (:stream, :design)
        fit = fit_revel(coarse, stats, 6; ties=:efron, engine=engine)
        d = event_diagnostics(fit)
        # the deviance residuals add up to the fitted deviance under Efron too
        @test sum(d.deviance_residual) ≈ -2loglikelihood(fit)
        @test all(0 .< d.probability .<= 1)
        # the null model's residual uses the Efron-weighted risk-set size
        @test any(d.null_residual .< 2log(30) - 1e-9) && all(d.null_residual .<= 2log(30) + 1e-12)
        s = prediction_summary(fit)
        @test s.deviance ≈ -2loglikelihood(fit)
    end
end

@testset "goodness of fit for undirected events" begin
    stats = [Inertia(symmetric=true, transform=:log1p), SharedPartners(transform=:log1p)]
    ev = simulate_events(stats, [1.0, 0.3], 6, 200; directed=false, rng=Xoshiro(4))
    # store half of the pairs the other way round: nothing may depend on it
    flipped = [isodd(k) ? Event(e.receiver, e.sender, e.time) : e for (k, e) in enumerate(ev)]
    fit = fit_revel(flipped, stats, 6; directed=false)
    @test coef(fit) ≈ coef(fit_revel(ev, stats, 6; directed=false)) atol = 1e-10
    result = gof(fit; n_sim=40, rng=Xoshiro(5))
    @test [s.name for s in result.statistics] ==
          ["mechanism shares (undirected)", "degree concentration (undirected)",
           "closing times (events)"]
    p = reduce(vcat, [s.p_values for s in result.statistics])
    @test count(p .< 0.05) <= 2              # a correctly specified model
    # directed layers on undirected data read the pair in ID order, not in the
    # order the pair was stored
    @test coef(fit_revel(flipped, [Inertia(transform=:log1p)], 6; directed=false)) ≈
          coef(fit_revel(ev, [Inertia(transform=:log1p)], 6; directed=false)) atol = 1e-10
end

@testset "wrappers keep their own arguments" begin
    stats = [Inertia(transform=:log1p)]
    ev = simulate_events(stats, [1.0], 5, 120; rng=Xoshiro(1))
    @test_throws ArgumentError fit_stratified(ev, stats, 5; by=e -> e.time <= 60, cases=1:120)
    @test_throws ArgumentError fit_moving_window(ev, stats, 5; width=40.0, cases=1:120)
    @test_throws ArgumentError fit_receiver_choice(ev, stats, 5; riskset=:full)
end

@testset "strata defined by who acts get their own risk sets" begin
    n = 8
    group = [1, 1, 1, 1, 2, 2, 2, 2]
    x = [0.0, 1.0, 2.0, 0.5, 1.5, 0.0, 2.5, 1.0]
    truth = [Inertia(transform=:log1p), SendEffect(x; name="x")]
    ev = simulate_events(truth, [0.8, 0.5], n, 1200; rng=Xoshiro(9))
    fits = fit_stratified(ev, truth, n; by=e -> group[e.sender])
    for g in (1, 2)
        fit = fits[g]
        # the dyads that could have produced an event of this stratum: the
        # group's four senders times seven receivers
        @test all(fit.fit.risk_set_sizes .== 4 * (n - 1))
        @test abs(coef(fit)[2] - 0.5) < 4 * stderror(fit)[2]
    end
    # a stratifier that does not depend on the dyad keeps the full risk set
    early = fit_stratified(ev, truth, n; by=e -> e.time <= 600)
    @test all(early[true].fit.risk_set_sizes .== n * (n - 1))
end

@testset "missing covariate values are refused" begin
    @test_throws ArgumentError Covariate([missing, 1, 2])
    @test_throws ArgumentError MatchEffect([missing, missing, "a"])
    @test_throws ArgumentError Covariate([0.0, 1.0], [1.0 missing; 2.0 3.0])
end

@testset "hyperevents with actors who join during the sequence" begin
    events = [HyperEvent([1], [2, 3], 1.0), HyperEvent([2], [1, 3], 2.0),
              HyperEvent([5], [1, 4], 3.0), HyperEvent([4], [5, 1], 4.0)]
    # actor 5 joins at t = 3
    at_risk = (m, e) -> e.time < 3 ? (1:4) : (1:5)
    d = hyper_design(events, [HyperReciprocation()], 5; n_controls=50, actors=at_risk)
    # the sender is fixed: C(3,2) = 3 receiver pairs among the other actors of
    # 1–4, then C(4,2) = 6 among the other actors of 1–5
    @test d.risk_set_size[d.is_event] == [3, 3, 6, 6]
    @test all(r -> all(<=(4), r), d.receivers[d.stratum .<= 2])
    # an event involving an actor who is not yet at risk is refused
    @test_throws ArgumentError hyper_design(events, [HyperReciprocation()], 5;
                                            actors=(m, e) -> 1:4)
end

# ----------------------------------------------------------------------------
# The concordance, the documentation and the namespace
# ----------------------------------------------------------------------------

module CatalogueScope
    using Revel
    const n = 5
    const h = 2.0
    const x = collect(1.0:5)
    const y = collect(5.0:-1:1)
    const matrix = ones(5, 5)
    const f = t -> 1.0
    const layer_a = EventLayer(types=:a)
    const layer_b = EventLayer(types=:b)
    const p = 2
    const q = 1
    const k = 2
    evaluate(call::String) = Core.eval(@__MODULE__, Meta.parse(call))
end

@testset "effect_catalogue" begin
    cat = effect_catalogue()
    @test names(cat) == ["revel", "family", "configuration", "relevent", "remstats", "rem",
                         "goldfish", "eventnet", "source"]
    @test allunique(cat.revel)
    @test Set(cat.family) == Set(["endogenous", "exogenous", "interaction", "hyperevent",
                                  "memory", "scaling"])
    @test all(!isempty, cat.configuration) && all(!isempty, cat.source)
    # every endogenous and exogenous entry is a call that builds a statistic
    for row in eachrow(cat)
        row.family in ("endogenous", "exogenous", "hyperevent") || continue
        stat = CatalogueScope.evaluate(row.revel)
        @test stat isa AbstractStatistic
    end
    # the equivalences the review singles out as traps
    lookup(col, value) = only(cat[cat[!, col] .== value, :revel])
    @test lookup(:relevent, "FrPSndSnd †") == "Inertia(scaling=:prop, empty=1/(n-1))"
    # relevent's three names whose 1.2.1 output departs from their documentation
    # are marked, and no cell claims what the package refuses
    @test sort(filter(x -> endswith(x, "†"), cat.relevent)) == ["FrPSndSnd †", "FrRecSnd †", "OSPSnd †"]
    @test !any(occursin("\"interact\"", c) for c in cat.remstats)
    # the zero-history share differs between relevent (1/(n-1)) and remstats (1/n)
    @test lookup(:relevent, "NODSnd") == "OutdegreeSender(scaling=:prop, empty=1/(n-1))"
    @test lookup(:remstats, "outdegreeSender(scaling = \"prop\")") ==
          "OutdegreeSender(scaling=:prop, empty=1/n)"
    @test lookup(:remstats, "inertia()") == "Inertia()"
    @test lookup(:goldfish, "commonReceiver") == "OSP(combine=:count)"
    @test lookup(:goldfish, "commonSender") == "ISP(combine=:count)"
    @test lookup(:relevent, "CovInt") == "SumEffect(x)"
    @test lookup(:rem, "triadStat") ==
          "OTP(combine=:product, root=true, memory=HalfLife(h; normalized=true))"
    # the documentation's concordance tables are the catalogue, rendered
    # (repair with `julia --project=docs docs/render_concordance.jl`)
    page = read(joinpath(pkgdir(Revel), "docs", "src", "guide", "concordance.md"), String)
    @test all(occursin("| `" * call * "` |", page) for call in cat.revel)
end

@testset "The bibliography covers every source the guide names" begin
    docs = joinpath(pkgdir(Revel), "docs")
    bib = read(joinpath(docs, "references.bib"), String)
    keys = Set(m.captures[1] for m in eachmatch(r"^@\w+\{([^,\s]+),"m, bib))
    @test length(keys) >= 260
    page = read(joinpath(docs, "src", "guide", "literature.md"), String)
    cited = [m.captures[1] for m in eachmatch(r"`([a-z]+[0-9]{4}[a-z]+)`", page)]
    @test length(cited) >= 40
    @test isempty(setdiff(cited, keys))
    # the package copy carries no pointers to local files
    @test !occursin(r"^\s*file\s*="m, bib)
end

@testset "Every exported docstring carries a runnable example" begin
    # Each Revel-owned docstring of an exported binding — including the ones
    # Revel attaches to the shared NetworkCore/StatsAPI generics (`compute`, `name`,
    # `gof`, `coef`, …) — must contain a fenced ```julia block, and every block
    # must run in a fresh module. Names re-exported unchanged from REM and
    # NetworkCore are documented where they are defined.
    reexported = (:Event, :n_simulations)
    meta = Base.Docs.meta(Revel)
    undocumented = String[]; missing_example = String[]
    blocks = Tuple{String,String}[]
    for nm in names(Revel)
        nm === :Revel && continue
        b = Base.Docs.Binding(Revel, nm)
        if nm in reexported
            @test b.mod !== Revel && haskey(Base.Docs.meta(b.mod), b)
            continue
        end
        if !haskey(meta, b)
            push!(undocumented, string(nm))
            continue
        end
        has_example = false
        for (_, ds) in meta[b].docs
            txt = ds.text isa AbstractString ? ds.text : join(string.(ds.text), "\n")
            for m in eachmatch(r"```julia\n(.*?)```"s, txt)
                has_example = true
                push!(blocks, (string(nm), String(m.captures[1])))
            end
        end
        has_example || push!(missing_example, string(nm))
    end
    @test isempty(undocumented)
    @test isempty(missing_example)
    failed = String[]; wrong = String[]; claims = 0
    for (nm, block) in blocks
        mod = Module()
        try
            c, bad = check_block(mod, block; where="docstring of $nm")
            claims += c; append!(wrong, bad)
        catch err
            @error "docstring example of $nm failed" exception = (err, catch_backtrace())
            push!(failed, nm)
        end
    end
    @test isempty(failed)
    @test length(blocks) >= 90
    # … and the values their comments state are the values they compute
    foreach(w -> @error(w), wrong)
    @test isempty(wrong)
    @test claims >= 250
end

@testset "Guide pages run, and the values they state hold" begin
    # Every Julia block of the README and of the documentation pages runs, one
    # module per page (the pages' own `@example` scoping), and every `# value`
    # comment is checked as for the docstrings
    root = pkgdir(Revel)
    pages = [joinpath(root, "README.md");
             joinpath.(root, "docs", "src", ["index.md", "getting_started.md"]);
             [joinpath(root, "docs", "src", "guide", f)
              for f in readdir(joinpath(root, "docs", "src", "guide")) if endswith(f, ".md")]]
    claims = 0; blocks = 0
    for page in pages
        mod = Module()
        Core.eval(mod, :(using Revel))
        for m in eachmatch(r"```(?:julia|@example[^\n]*)\n(.*?)```"s, read(page, String))
            blocks += 1
            c, bad = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                redirect_stdout(devnull) do
                    check_block(mod, m.captures[1]; where=basename(page))
                end
            end
            claims += c
            foreach(w -> @error(w), bad)
            @test isempty(bad)
        end
    end
    @test blocks >= 80 && claims >= 90
end

@testset "Namespace" begin
    # no export shadows Base
    @test isempty([s for s in names(Revel) if isdefined(Base, s) && Base.isexported(Base, s) &&
                                              getfield(Base, s) !== getfield(Revel, s)])
    # a name Revel shares with REM or NetworkCore is the SAME binding, so
    # co-loading leaves every name usable unqualified
    for pkg in (REM, NetworkCore), sym in names(Revel)
        sym === :Revel && continue
        if sym in names(pkg)
            @test getfield(pkg, sym) === getfield(Revel, sym)
        end
    end
    @test Revel.compute === REM.compute === NetworkCore.compute
    @test Revel.gof === NetworkCore.gof
    @test Revel.coeftable === NetworkCore.coeftable
end

@testset "Namespace: co-loading with ERGM, SNA and Siena" begin
    # Two packages exporting different bindings under one name leave that name
    # UNDEFINED for anyone who loads both — and `using ERGM, SNA, Siena` beside a
    # relational event package is the statnet workflow. Checked in a fresh
    # process, where the names are resolved as a user's session resolves them.
    # (ERGM, SNA and Siena are test-only dependencies. Revel's `SameEffect`,
    # `DifferenceEffect` and `SimilarityEffect` were renamed `MatchEffect`,
    # `DiffEffect` and `SimEffect` because Siena exports those three names.)
    script = """
        using ERGM, SNA, Siena, REM, Revel
        bad = Symbol[]
        for nm in names(Revel)
            # `public` names (Revel.fit_obpm, …) are not brought in by `using`
            (nm === :Revel || !Base.isexported(Revel, nm)) && continue
            ok = try
                getfield(Main, nm) === getfield(Revel, nm)
            catch
                false
            end
            ok || push!(bad, nm)
        end
        isempty(bad) || error("names that do not survive co-loading: \$bad")
        h = build_history([Event(1, 2, 1.0)])
        compute(Inertia(), h, 1, 2, 2.0) == 1.0 || error("compute is not usable unqualified")
        print("ok")
        """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
    @test read(cmd, String) == "ok"
end

# ----------------------------------------------------------------------------
# Relational hyperevents
# ----------------------------------------------------------------------------

@testset "Hyperevents: HyperEvent construction" begin
    meeting = HyperEvent([3, 1, 2], 1.0)
    @test meeting.senders == [1, 2, 3]            # stored sorted
    @test isempty(meeting.receivers)
    @test !is_directed(meeting)
    @test participants(meeting) == [1, 2, 3]
    @test meeting.eventtype === :event && meeting.weight == 1.0
    @test meeting isa HyperEvent{Float64}

    mail = HyperEvent([4], [2, 1], 7; eventtype=:email, weight=2.5)
    @test mail isa HyperEvent{Int}
    @test mail.senders == [4] && mail.receivers == [1, 2]
    @test is_directed(mail)
    @test participants(mail) == [1, 2, 4]
    @test mail.eventtype === :email && mail.weight == 2.5
    @test HyperEvent(1, [2, 3], 1.0).senders == [1]      # a single sender as a scalar

    @test sprint(show, meeting) == "HyperEvent({1, 2, 3} @ 1.0)"
    @test sprint(show, mail) == "HyperEvent({4} → {1, 2} @ 7)"
    @test HyperEvent([1, 2], 1.0) == HyperEvent([2, 1], 1.0)
    @test hash(HyperEvent([1, 2], 1.0)) == hash(HyperEvent([2, 1], 1.0))
    @test HyperEvent([1, 2], 1.0) != HyperEvent([1], [2], 1.0)

    @test_throws ArgumentError HyperEvent(Int[], 1.0)               # nobody
    @test_throws ArgumentError HyperEvent(Int[], [1, 2], 1.0)       # no sender
    @test_throws ArgumentError HyperEvent([1, 1, 2], 1.0)           # a set, not a list
    @test_throws ArgumentError HyperEvent([0, 1], 1.0)              # positive IDs
    @test_throws ArgumentError HyperEvent([1], [1, 2], 1.0)         # loop
    @test_throws ArgumentError HyperEvent([1, 2], 1.0; weight=NaN)
end

@testset "Hyperevents: history and lazy indices" begin
    events = [HyperEvent([1, 2, 3], 1.0), HyperEvent([1, 2], 2.0)]
    h = build_hyper_history(events)
    @test h isa HyperHistory{Float64}
    @test length(h) == 2
    @test isempty(h.indices)                      # nothing indexed until asked for
    @test sprint(show, h) == "HyperHistory(2 undirected hyperevents)"

    @test compute(SubsetRepetition(2), h, [1, 2], Int[], 3.0) == 2.0
    @test collect(keys(h.indices)) == [(:und, 2, 0)]   # only the order asked for
    @test compute(ExactRepetition(), h, [1, 2], Int[], 3.0) == 1.0
    @test haskey(h.indices, (:exact, 0, 0)) && !haskey(h.indices, (:und, 1, 0))

    # an index built earlier is kept up to date by later events
    update_hyper_history!(h, HyperEvent([1, 2], 3.0))
    @test compute(SubsetRepetition(2), h, [1, 2], Int[], 4.0) == 3.0
    @test compute(ExactRepetition(), h, [1, 2], Int[], 4.0) == 2.0
    # … and agrees with a history that indexes everything afterwards
    fresh = build_hyper_history([events; HyperEvent([1, 2], 3.0)])
    for stat in (SubsetRepetition(1), SubsetRepetition(2), ExactRepetition(),
                 HyperClosure(), SharedPriorEvents(2))
        @test compute(stat, h, [1, 2, 3], Int[], 4.0) ==
              compute(stat, fresh, [1, 2, 3], Int[], 4.0)
    end

    # events later than the evaluation time have not happened yet
    @test compute(SubsetRepetition(2), h, [1, 2], Int[], 2.5) == 2.0
    @test compute(SubsetRepetition(2), h, [1, 2], Int[], 0.5) == 0.0

    # build_hyper_history sorts; update_hyper_history! insists on time order
    shuffled = build_hyper_history([HyperEvent([1, 2], 2.0), HyperEvent([1, 2, 3], 1.0)])
    @test [e.time for e in shuffled.events] == [1.0, 2.0]
    @test_throws ArgumentError update_hyper_history!(h, HyperEvent([1, 2], 0.5))
    # one history, one kind of hyperevent
    @test_throws ArgumentError update_hyper_history!(h, HyperEvent([1], [2], 9.0))
    # an empty history has no past
    empty_history = HyperHistory()
    @test empty_history isa HyperHistory{Float64}
    @test compute(SubsetRepetition(1), empty_history, [1, 2], Int[], 1.0) == 0.0
    @test compute(HyperClosure(), empty_history, [1, 2], Int[], 1.0) == 0.0

    # a candidate must be a loopless hyperedge with sorted actor sets
    @test_throws ArgumentError compute(SubsetRepetition(1), h, [2, 1], Int[], 4.0)
    @test_throws ArgumentError compute(SubsetRepetition(1), h, [1, 1], Int[], 4.0)
    @test_throws ArgumentError compute(SubsetRepetition(1), h, Int[], [1], 4.0)
    @test_throws ArgumentError compute(SubsetRepetition(1), h, [1], [1, 2], 4.0)
    # the candidate-as-event form is the same call
    @test compute(SubsetRepetition(2), h, HyperEvent([2, 1], 4.0)) == 3.0
end

@testset "Hyperevents: hyperedge size" begin
    h = HyperHistory()
    @test compute(HyperedgeSize(), h, [1, 2, 3], Int[], 1.0) == 3.0
    @test compute(HyperedgeSize(), h, [1], [2, 3, 4], 1.0) == 4.0
    @test compute(HyperedgeSize(endpoint=:senders), h, [1], [2, 3, 4], 1.0) == 1.0
    @test compute(HyperedgeSize(endpoint=:receivers), h, [1], [2, 3, 4], 1.0) == 3.0
    @test compute(HyperedgeSize(endpoint=:receivers), h, [1, 2], Int[], 1.0) == 0.0
    @test compute(HyperedgeSize(transform=x -> x^2), h, [1, 2, 3], Int[], 1.0) == 9.0
    @test name(HyperedgeSize()) == "size"
    @test name(HyperedgeSize(endpoint=:receivers)) == "size.receivers"
    @test HyperedgeSize() isa AbstractHyperStatistic
    @test HyperedgeSize() isa REM.AbstractStatistic
    @test_throws ArgumentError HyperedgeSize(endpoint=:both)
end

@testset "Hyperevents: subset repetition (hand-computed)" begin
    # e1 {1,2,3} @1   e2 {1,2} @2 (weight 2, type :b)   e3 {2,3,4} @3   e4 {1,2,3} @5
    h = build_hyper_history([HyperEvent([1, 2, 3], 1.0),
                             HyperEvent([1, 2], 2.0; weight=2.0, eventtype=:b),
                             HyperEvent([2, 3, 4], 3.0), HyperEvent([1, 2, 3], 5.0)])
    c = [1, 2, 3]; none = Int[]
    # order 1 (activity): deg(1) = {e1,e2,e4} = 3, deg(2) = 4, deg(3) = {e1,e3,e4} = 3
    @test compute(SubsetRepetition(1), h, c, none, 6.0) ≈ 10 / 3
    @test compute(SubsetRepetition(1; aggregate=:sum), h, c, none, 6.0) == 10.0
    @test compute(SubsetRepetition(1; aggregate=:min), h, c, none, 6.0) == 3.0
    @test compute(SubsetRepetition(1; aggregate=:max), h, c, none, 6.0) == 4.0
    # population sd of (3, 4, 3): mean 10/3, squared deviations 1/9 + 4/9 + 1/9 = 6/9,
    # variance (6/9)/3 = 2/9
    @test compute(SubsetRepetition(1; aggregate=:sd), h, c, none, 6.0) ≈ sqrt(2) / 3
    # pairs of (3, 4, 3): |3−4| + |3−3| + |4−3| = 2; mean over 3 pairs, or minus the sum
    @test compute(SubsetRepetition(1; aggregate=:absdiff), h, c, none, 6.0) ≈ 2 / 3
    @test compute(SubsetRepetition(1; aggregate=:assortativity), h, c, none, 6.0) == -2.0

    # order 2: deg{1,2} = {e1,e2,e4} = 3, deg{1,3} = {e1,e4} = 2, deg{2,3} = {e1,e3,e4} = 3
    @test compute(SubsetRepetition(2), h, c, none, 6.0) ≈ 8 / 3
    @test compute(SubsetRepetition(2; aggregate=:sum), h, c, none, 6.0) == 8.0
    # order 3: deg{1,2,3} = {e1,e4} = 2
    @test compute(SubsetRepetition(3), h, c, none, 6.0) == 2.0
    # order 3 on {1,2,3,4}: {1,2,3} → 2, {1,2,4} → 0, {1,3,4} → 0, {2,3,4} → 1 (e3)
    @test compute(SubsetRepetition(3), h, [1, 2, 3, 4], none, 6.0) == 3 / 4
    @test compute(SubsetRepetition(3; aggregate=:max), h, [1, 2, 3, 4], none, 6.0) == 2.0
    @test compute(SubsetRepetition(3; aggregate=:min), h, [1, 2, 3, 4], none, 6.0) == 0.0
    # no 4-subset of a 3-hyperedge, no pair of 1-subsets of a singleton
    @test compute(SubsetRepetition(4), h, c, none, 6.0) == 0.0
    @test compute(SubsetRepetition(1; aggregate=:assortativity), h, [1], none, 6.0) == 0.0
    # an actor never seen
    @test compute(SubsetRepetition(1), h, [7, 8], none, 6.0) == 0.0

    # memory kernels, read at t = 6: ages are e1 5, e2 4, e3 3, e4 1
    # half-life 1 on {1,2}: 2^-5 + 2^-4 + 2^-1 = 0.59375
    @test compute(SubsetRepetition(2; memory=HalfLife(1.0)), h, [1, 2], none, 6.0) ≈ 0.59375
    # window 3 keeps e3 (exactly 3 old) and e4: deg{2,3} = 2, deg{1,2} = 1
    @test compute(SubsetRepetition(2; memory=Window(3.0)), h, [2, 3], none, 6.0) == 2.0
    @test compute(SubsetRepetition(2; memory=Window(3.0)), h, [1, 2], none, 6.0) == 1.0
    # interval (1, 4] keeps e2 and e3 only (e4 is exactly 1 old: excluded)
    @test compute(SubsetRepetition(2; memory=IntervalMemory(1.0, 4.0)), h, [1, 2], none, 6.0) == 1.0
    @test compute(SubsetRepetition(2; memory=IntervalMemory(1.0, 4.0)), h, [2, 3], none, 6.0) == 1.0
    # linear decay over 10: (1 − 5/10) + (1 − 4/10) + (1 − 1/10) = 2.0
    @test compute(SubsetRepetition(2; memory=LinearDecay(10.0)), h, [1, 2], none, 6.0) ≈ 2.0
    # power law, exponent 1: 1/5 + 1/4 + 1/1 = 1.45
    @test compute(SubsetRepetition(2; memory=PowerLaw(1.0)), h, [1, 2], none, 6.0) ≈ 1.45
    # a user kernel: weight 1 up to age 2, else 0 → e4 only
    @test compute(SubsetRepetition(2; memory=KernelMemory(a -> a <= 2 ? 1.0 : 0.0; support=2.0)),
                  h, [1, 2], none, 6.0) == 1.0

    # event weights: e1 1 + e2 2 + e4 1 on {1,2}
    @test compute(SubsetRepetition(2; weighted=true), h, [1, 2], none, 6.0) == 4.0
    # the event-type filter: only e2 is of type :b
    @test compute(SubsetRepetition(2; types=:b), h, [1, 2], none, 6.0) == 1.0
    @test compute(SubsetRepetition(2; types=[:b], weighted=true), h, [1, 2], none, 6.0) == 2.0
    @test compute(SubsetRepetition(2; types=:event), h, [1, 2], none, 6.0) == 2.0
    @test compute(SubsetRepetition(2; types=:nope), h, [1, 2], none, 6.0) == 0.0
    # the transform is applied to the aggregate
    @test compute(SubsetRepetition(2; transform=:log1p), h, c, none, 6.0) ≈ log1p(8 / 3)

    @test name(SubsetRepetition(2)) == "subrep(2)"
    @test name(SubsetRepetition(2; aggregate=:sum)) == "subrep(2).sum"
    @test name(SubsetRepetition(1; memory=HalfLife(7.0), types=:b, weighted=true,
                                transform=:log1p)) ==
          "log1p(subrep(1)[halflife=7.0,types=b,weighted])"
    @test name(SubsetRepetition(2; name="familiarity")) == "familiarity"
end

@testset "Hyperevents: exact repetition, shared prior events, prior success" begin
    h = build_hyper_history([HyperEvent([1, 2, 3], 1.0),
                             HyperEvent([1, 2], 2.0; weight=2.0, eventtype=:b),
                             HyperEvent([2, 3, 4], 3.0), HyperEvent([1, 2, 3], 5.0)])
    none = Int[]
    # exactly {1,2,3}: e1 and e4; exactly {1,2}: e2 only; exactly {2,3}: never
    @test compute(ExactRepetition(), h, [1, 2, 3], none, 6.0) == 2.0
    @test compute(ExactRepetition(), h, [1, 2], none, 6.0) == 1.0
    @test compute(ExactRepetition(), h, [2, 3], none, 6.0) == 0.0
    @test compute(ExactRepetition(weighted=true), h, [1, 2], none, 6.0) == 2.0
    # half-life 2 on {1,2,3}: 2^(-5/2) + 2^(-1/2)
    @test compute(ExactRepetition(memory=HalfLife(2.0)), h, [1, 2, 3], none, 6.0) ≈
          2.0^(-2.5) + 2.0^(-0.5)
    # the reverse of an undirected hyperedge is itself, and so is its participant set
    @test compute(ExactRepetition(direction=:in), h, [1, 2, 3], none, 6.0) == 2.0
    @test compute(ExactRepetition(direction=:sym), h, [1, 2, 3], none, 6.0) == 2.0
    @test compute(UnorderedRepetition(), h, [1, 2], none, 6.0) == 1.0
    @test name(ExactRepetition()) == "exact.rep"
    @test name(ExactRepetition(direction=:in)) == "exact.recip"
    @test name(UnorderedRepetition()) == "unordered.rep"

    # shared prior events: candidate {1,2,4} shares {1,2} with e1, e2, e4 and {2,4}
    # with e3 → 4 events share ≥ 2; none shares all three
    @test compute(SharedPriorEvents(2), h, [1, 2, 4], none, 6.0) == 4.0
    @test compute(SharedPriorEvents(3), h, [1, 2, 4], none, 6.0) == 0.0
    # candidate {3,4}: e1, e3, e4 contain 3 or 4; only e3 contains both
    @test compute(SharedPriorEvents(1), h, [3, 4], none, 6.0) == 3.0
    @test compute(SharedPriorEvents(2), h, [3, 4], none, 6.0) == 1.0
    # each past event counts once here but C(|overlap|, 2) times in the summed
    # subset repetition: on {1,2,3} the overlaps are 3, 2, 2, 3 → 4 versus 3+1+1+3 = 8
    @test compute(SharedPriorEvents(2), h, [1, 2, 3], none, 6.0) == 4.0
    @test compute(SubsetRepetition(2; aggregate=:sum), h, [1, 2, 3], none, 6.0) == 8.0
    # weights, types and a window (ages 5, 4, 3, 1): window 3.5 keeps e3, e4
    @test compute(SharedPriorEvents(2; weighted=true), h, [1, 2, 4], none, 6.0) == 5.0
    @test compute(SharedPriorEvents(2; types=:b), h, [1, 2, 4], none, 6.0) == 1.0
    @test compute(SharedPriorEvents(2; memory=Window(3.5)), h, [1, 2, 4], none, 6.0) == 2.0
    @test compute(SharedPriorEvents(5), h, [1, 2], none, 6.0) == 0.0
    @test name(SharedPriorEvents(2)) == "shared.events(2)"

    # prior success: the weight is the outcome.
    # p1 {1,2} y=10   p2 {2,3} y=4   p3 {1,2,3} y=1
    papers = build_hyper_history([HyperEvent([1, 2], 1.0; weight=10.0),
                                  HyperEvent([2, 3], 2.0; weight=4.0),
                                  HyperEvent([1, 2, 3], 3.0; weight=1.0)])
    # order 1: outcomes 11, 15, 5 (sum 31) over 2, 3, 2 events (sum 7)
    @test compute(PriorSuccess(1), papers, [1, 2, 3], none, 4.0) ≈ 31 / 7
    # order 2: {1,2} 11 over 2, {1,3} 1 over 1, {2,3} 5 over 2 → 17/5
    @test compute(PriorSuccess(2), papers, [1, 2, 3], none, 4.0) ≈ 17 / 5
    # order 3: {1,2,3} 1 over 1
    @test compute(PriorSuccess(3), papers, [1, 2, 3], none, 4.0) == 1.0
    # no past event: 0, not NaN
    @test compute(PriorSuccess(2), papers, [1, 4], none, 4.0) == 0.0
    @test compute(PriorSuccess(1), papers, [4, 5], none, 4.0) == 0.0
    # it is the ratio of the outcome-weighted to the unweighted summed subset repetition
    @test compute(PriorSuccess(2), papers, [1, 2, 3], none, 4.0) ≈
          compute(SubsetRepetition(2; weighted=true, aggregate=:sum), papers, [1, 2, 3], none, 4.0) /
          compute(SubsetRepetition(2; aggregate=:sum), papers, [1, 2, 3], none, 4.0)
    @test name(PriorSuccess(2)) == "prior.success(2)"
end

@testset "Hyperevents: directed subset repetition and the named effects" begin
    # m1 1 → {2,3} @1   m2 1 → {2,4} @2   m3 2 → {1,3} @3   m4 4 → {2,3} @4
    h = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([1], [2, 4], 2.0),
                             HyperEvent([2], [1, 3], 3.0), HyperEvent([4], [2, 3], 4.0)])
    @test sprint(show, h) == "HyperHistory(4 directed hyperevents)"
    S = [1]; R = [2, 3]
    # (1,1): deg(1→2) = {m1,m2} = 2, deg(1→3) = {m1} = 1
    @test compute(DirectedSubsetRepetition(1, 1), h, S, R, 5.0) == 1.5
    @test compute(DirectedSubsetRepetition(1, 1; aggregate=:sum), h, S, R, 5.0) == 3.0
    @test compute(DirectedSubsetRepetition(1, 1; aggregate=:min), h, S, R, 5.0) == 1.0
    @test compute(SenderReceiverSetRepetition(1), h, S, R, 5.0) == 1.5
    # (1,2): sender 1 addressed {2,3} together once (m1)
    @test compute(DirectedSubsetRepetition(1, 2), h, S, R, 5.0) == 1.0
    @test compute(SenderReceiverSetRepetition(2), h, S, R, 5.0) == 1.0
    # (0,2): {2,3} were addressed together by anyone in m1 and m4
    @test compute(DirectedSubsetRepetition(0, 2), h, S, R, 5.0) == 2.0
    @test compute(ReceiverSetRepetition(2), h, S, R, 5.0) == 2.0
    # (0,1): in-degrees 2 ← {m1,m2,m4} = 3, 3 ← {m1,m3,m4} = 3, 4 ← {m2} = 1
    @test compute(ReceiverSetRepetition(1), h, S, R, 5.0) == 3.0
    @test compute(HyperReceiverPopularity(), h, S, [2, 4], 5.0) == 2.0
    # (1,0): sender 1 sent m1 and m2
    @test compute(HyperSenderActivity(), h, S, R, 5.0) == 2.0
    @test compute(HyperSenderActivity(), h, [3], [1, 2], 5.0) == 0.0
    # several senders are averaged: out-degrees of 1 and 2 are 2 and 1
    @test compute(HyperSenderActivity(), h, [1, 2], [3], 5.0) == 1.5
    # (2,0): 1 and 2 never co-sent
    @test compute(DirectedSubsetRepetition(2, 0), h, [1, 2], [3], 5.0) == 0.0

    # reciprocation = (1,1) on the reversed hyperedge: deg(2→1) = {m3} = 1, deg(3→1) = 0
    @test compute(HyperReciprocation(), h, S, R, 5.0) == 0.5
    @test compute(DirectedSubsetRepetition(1, 1; direction=:in), h, S, R, 5.0) == 0.5
    # out-in popularity = (1,0) reversed: out-degrees of the receivers, 2 → 1, 3 → 0, 4 → 1
    @test compute(OutInPopularity(), h, S, R, 5.0) == 0.5
    @test compute(OutInPopularity(), h, S, [2, 4], 5.0) == 1.0
    # (0,1) reversed: the in-degree of the sender, 1 ← {m3} = 1
    @test compute(DirectedSubsetRepetition(0, 1; direction=:in), h, S, R, 5.0) == 1.0

    # :sym ignores roles: order 1 + 1 = 2 on the participant sets {1,2,3}, {1,2,4},
    # {1,2,3}, {2,3,4}: deg{1,2} = 3, deg{1,3} = 2, deg{2,3} = 3
    @test compute(DirectedSubsetRepetition(1, 1; direction=:sym), h, S, R, 5.0) ≈ 8 / 3
    @test compute(SubsetRepetition(2), h, S, R, 5.0) ≈ 8 / 3
    # an undirected statistic on directed events reads participant sets
    @test compute(SubsetRepetition(1), h, [1], [4], 5.0) == (3 + 2) / 2

    # exact repetition needs both sets to match
    @test compute(ExactRepetition(), h, [1], [2, 3], 5.0) == 1.0
    @test compute(ExactRepetition(), h, [4], [2, 3], 5.0) == 1.0
    @test compute(ExactRepetition(), h, [1], [2], 5.0) == 0.0
    @test compute(ExactRepetition(), h, [3], [1, 2], 5.0) == 0.0
    # exact reciprocation: the reverse of {1,3} → {2} is 2 → {1,3} = m3
    @test compute(ExactRepetition(direction=:in), h, [1, 3], [2], 5.0) == 1.0
    @test compute(ExactRepetition(direction=:in), h, [1], [2, 3], 5.0) == 0.0
    # unordered repetition: m1 and m3 were both among exactly {1,2,3}
    @test compute(UnorderedRepetition(), h, [3], [1, 2], 5.0) == 2.0
    @test compute(UnorderedRepetition(), h, [2], [3, 4], 5.0) == 1.0

    # memory: half-life 1 at t = 5 on deg(1→2): 2^-4 + 2^-3
    @test compute(DirectedSubsetRepetition(1, 1; memory=HalfLife(1.0)), h, [1], [2], 5.0) ≈
          2.0^-4 + 2.0^-3
    # too few senders or receivers for the order: no sub-hyperedge
    @test compute(DirectedSubsetRepetition(0, 3), h, S, R, 5.0) == 0.0
    @test compute(DirectedSubsetRepetition(2, 1), h, S, R, 5.0) == 0.0

    @test name(DirectedSubsetRepetition(1, 2)) == "subrep(1,2)"
    @test name(DirectedSubsetRepetition(1, 1; direction=:in)) == "subrecip(1,1)"
    @test name(ReceiverSetRepetition(2)) == "rec.subrep(2)"
    @test name(SenderReceiverSetRepetition(2)) == "send.rec.subrep(2)"
    @test name(HyperSenderActivity()) == "sender.activity"
    @test name(HyperReceiverPopularity()) == "receiver.popularity"
    @test name(HyperReciprocation()) == "reciprocation"
    @test name(OutInPopularity()) == "out.in.popularity"
end

@testset "Hyperevents: interaction among receivers" begin
    # m1 1 → {2,3}   m2 1 → {2,4}   m3 2 → {1,3}   m4 4 → {2,3}
    h = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([1], [2, 4], 2.0),
                             HyperEvent([2], [1, 3], 3.0), HyperEvent([4], [2, 3], 4.0)])
    # candidate 5 → {1,2,3}, order 1: for each receiver j, the events j sent to each
    # other receiver: j=1: deg(1→2) = 2, deg(1→3) = 1; j=2: deg(2→1) = 1, deg(2→3) = 1;
    # j=3: 0, 0.  Sum 5 over |J|·C(|J|−1, 1) = 3·2 = 6 terms.
    @test compute(InteractionAmongReceivers(1), h, [5], [1, 2, 3], 5.0) ≈ 5 / 6
    @test compute(InteractionAmongReceivers(), h, [5], [1, 2, 3], 5.0) ≈ 5 / 6
    @test compute(InteractionAmongReceivers(1; aggregate=:sum), h, [5], [1, 2, 3], 5.0) == 5.0
    # order 2: j=1: deg(1→{2,3}) = 1 (m1); j=2: deg(2→{1,3}) = 1 (m3); j=3: 0.
    # Sum 2 over 3·C(2,2) = 3 terms.
    @test compute(InteractionAmongReceivers(2), h, [5], [1, 2, 3], 5.0) ≈ 2 / 3
    # a receiver set too small for the order
    @test compute(InteractionAmongReceivers(2), h, [5], [1, 2], 5.0) == 0.0
    @test compute(InteractionAmongReceivers(1), h, [5], [1], 5.0) == 0.0
    # window 1.5 at t = 5 keeps m4 only: j=4 is not a receiver here → 0
    @test compute(InteractionAmongReceivers(1; memory=Window(1.5)), h, [5], [1, 2, 3], 5.0) == 0.0
    @test name(InteractionAmongReceivers(2)) == "interact.rec(2)"
end

@testset "Hyperevents: closure" begin
    none = Int[]
    # undirected: u1 {1,3}  u2 {1,3}  u3 {2,3,4}  u4 {1,4}
    # projection: W13 = 2, W23 = W24 = W34 = 1, W14 = 1
    h = build_hyper_history([HyperEvent([1, 3], 1.0), HyperEvent([1, 3], 2.0),
                             HyperEvent([2, 3, 4], 3.0), HyperEvent([1, 4], 4.0)])
    # pair {1,2}: through 3, min(W13, W23) = min(2, 1) = 1; through 4, min(W14, W24) = 1
    @test compute(HyperClosure(), h, [1, 2], none, 5.0) == 2.0
    # product: 2·1 + 1·1 = 3; the largest path instead of their sum: 1 (min), 2 (product)
    @test compute(HyperClosure(combine=:product), h, [1, 2], none, 5.0) == 3.0
    @test compute(HyperClosure(parallel=:max), h, [1, 2], none, 5.0) == 1.0
    @test compute(HyperClosure(combine=:product, parallel=:max), h, [1, 2], none, 5.0) == 2.0
    # candidate {1,2,3} (Lerner et al. 2021: sum over pairs and third actors of the
    # min, divided by the number of pairs):
    #   {1,2} → 2;  {1,3}: through 2 min(W12 = 0, ·) = 0, through 4 min(W14, W34) = 1;
    #   {2,3}: through 1 min(W21 = 0, ·) = 0, through 4 min(W24, W34) = 1
    @test compute(HyperClosure(), h, [1, 2, 3], none, 5.0) ≈ 4 / 3
    # the unnormalised form of Lerner, Hâncean & Perc (2025)
    @test compute(HyperClosure(aggregate=:sum), h, [1, 2, 3], none, 5.0) == 4.0
    @test compute(HyperClosure(aggregate=:max), h, [1, 2, 3], none, 5.0) == 2.0
    @test compute(HyperClosure(aggregate=:min), h, [1, 2, 3], none, 5.0) == 1.0
    # the 2019 normalisation: also divide by the n − 2 possible third actors
    @test compute(HyperClosure(normalize=:thirds, n_actors=5), h, [1, 2, 3], none, 5.0) ≈ 4 / 9
    # a window of 1.5 at t = 5 keeps u4 only: no two-path left
    @test compute(HyperClosure(memory=Window(1.5)), h, [1, 2], none, 5.0) == 0.0
    # half-life 1 at t = 5: W13 = 2^-4 + 2^-3, W23 = W24 = 2^-2, W14 = 2^-1
    @test compute(HyperClosure(memory=HalfLife(1.0)), h, [1, 2], none, 5.0) ≈
          min(2.0^-4 + 2.0^-3, 2.0^-2) + min(2.0^-1, 2.0^-2)
    # a singleton has no pair; directions are ignored on undirected hyperevents
    @test compute(HyperClosure(), h, [1], none, 5.0) == 0.0
    @test compute(HyperClosure(:cyclic), h, [1, 2], none, 5.0) == 2.0

    # directed: m1 1 → {2,3}   m2 1 → {2,4}   m3 2 → {1,3}   m4 4 → {2,3}
    # projection: W12 = 2, W13 = 1, W14 = 1, W21 = 1, W23 = 1, W42 = 1, W43 = 1
    d = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([1], [2, 4], 2.0),
                             HyperEvent([2], [1, 3], 3.0), HyperEvent([4], [2, 3], 4.0)])
    # candidate 1 → {3}, third actors 2 and 4
    #   transitive  W(1,k), W(k,3): min(2, 1) + min(1, 1) = 2
    @test compute(HyperClosure(:transitive), d, [1], [3], 5.0) == 2.0
    @test compute(HyperClosure(), d, [1], [3], 5.0) == 2.0            # the default
    @test compute(HyperClosure(:transitive; combine=:product), d, [1], [3], 5.0) == 3.0
    @test compute(HyperClosure(:transitive; parallel=:max), d, [1], [3], 5.0) == 1.0
    #   cyclic  W(k,1), W(3,k): actor 3 never sent → 0
    @test compute(HyperClosure(:cyclic), d, [1], [3], 5.0) == 0.0
    #   shared senders  W(k,1), W(k,3): k=2 min(1, 1) = 1; k=4 W41 = 0
    @test compute(HyperClosure(:shared_senders), d, [1], [3], 5.0) == 1.0
    #   shared receivers  W(1,k), W(3,k): 0
    @test compute(HyperClosure(:shared_receivers), d, [1], [3], 5.0) == 0.0
    # candidate 2 → {1}: cyclic through 4, min(W42, W14) = 1 (1 → 4 → 2 closes 2 → 1)
    @test compute(HyperClosure(:cyclic), d, [2], [1], 5.0) == 1.0
    @test compute(HyperClosure(:transitive), d, [2], [1], 5.0) == 0.0
    # candidate 1 → {2}: shared receivers through 3, min(W13, W23) = 1
    @test compute(HyperClosure(:shared_receivers), d, [1], [2], 5.0) == 1.0
    # symmetric legs for 1 → {3}: k=2 min(W12 + W21, W32 + W23) = min(3, 1);
    # k=4 min(W14 + W41, W34 + W43) = min(1, 1)
    @test compute(HyperClosure(dir1=:sym, dir2=:sym), d, [1], [3], 5.0) == 2.0
    # averaged over the (sender, receiver) pairs (Lerner & Lomi 2023: Σ_j … / |J|):
    # 1 → {3,4}: pair (1,3) → 2; pair (1,4): W(k,4) = 0 for k = 2, 3 → 0
    @test compute(HyperClosure(:transitive), d, [1], [3, 4], 5.0) == 1.0
    @test compute(HyperClosure(:transitive; aggregate=:sum), d, [1], [3, 4], 5.0) == 2.0
    @test compute(HyperClosure(:transitive; normalize=:thirds, n_actors=5), d, [1], [3], 5.0) ≈ 2 / 3

    @test name(HyperClosure()) == "closure(out,in)"
    @test name(HyperClosure(:transitive)) == "closure.transitive"
    @test name(HyperClosure(:shared_senders; combine=:product)) == "closure.shared_senders.product"
    @test_throws ArgumentError HyperClosure(:balance)
    @test_throws ArgumentError HyperClosure(combine=:sum)
    @test_throws ArgumentError HyperClosure(parallel=:mean)
    @test_throws ArgumentError HyperClosure(dir1=:up)
    @test_throws ArgumentError HyperClosure(normalize=:pairs)
    @test_throws ArgumentError HyperClosure(normalize=:thirds)             # needs n_actors
    @test_throws ArgumentError HyperClosure(n_actors=5)                    # unused otherwise
    @test_throws ArgumentError HyperClosure(aggregate=:sd)
end

@testset "Hyperevents: covariates on a hyperedge" begin
    h = HyperHistory()
    x = [1.0, 2.0, 4.0, 8.0, 16.0]
    none = Int[]
    c = [1, 2, 4]                      # values 1, 2, 8
    @test compute(HyperCovariate(x), h, c, none, 0.0) ≈ 11 / 3
    @test compute(HyperCovariate(x; aggregate=:sum), h, c, none, 0.0) == 11.0
    @test compute(HyperCovariate(x; aggregate=:min), h, c, none, 0.0) == 1.0
    @test compute(HyperCovariate(x; aggregate=:max), h, c, none, 0.0) == 8.0
    # deviations from 11/3: −8/3, −5/3, 13/3; squares sum to 258/9
    @test compute(HyperCovariate(x; aggregate=:sd), h, c, none, 0.0) ≈ sqrt(258 / 9 / 3)
    @test compute(HyperCovariate(x; aggregate=:samplesd), h, c, none, 0.0) ≈ sqrt(258 / 9 / 2)
    @test compute(HyperCovariate(x; aggregate=:samplesd), h, c, none, 0.0) ≈ std([1.0, 2.0, 8.0])
    # pairs: |1−2| + |1−8| + |2−8| = 14 over 3 pairs
    @test compute(HyperCovariate(x; aggregate=:absdiff), h, c, none, 0.0) ≈ 14 / 3
    @test compute(HyperCovariate(x; aggregate=:absdiff), h, [3], none, 0.0) == 0.0

    # directed 1 → {2,4}
    S = [1]; R = [2, 4]
    @test compute(HyperCovariate(x), h, S, R, 0.0) ≈ 11 / 3
    @test compute(HyperCovariate(x; endpoint=:senders), h, S, R, 0.0) == 1.0
    # receiver-set average (Lerner & Lomi 2023)
    @test compute(HyperCovariate(x; endpoint=:receivers), h, S, R, 0.0) == 5.0
    # sender–receiver heterophily: (|1−2| + |1−8|)/2 — pairs run across the two sets
    @test compute(HyperCovariate(x; aggregate=:absdiff), h, S, R, 0.0) == 4.0
    # receiver-set heterophily: |2−8|
    @test compute(HyperCovariate(x; endpoint=:receivers, aggregate=:absdiff), h, S, R, 0.0) == 6.0
    @test compute(HyperCovariate(x; endpoint=:senders, aggregate=:absdiff), h, S, R, 0.0) == 0.0

    # categorical: a, a, b, b, a
    group = Covariate([:a, :a, :b, :b, :a]; name="group")
    # {1,2,4}: (1,2) same, (1,4) and (2,4) differ
    @test compute(HyperCovariate(group; aggregate=:catdiff), h, c, none, 0.0) ≈ 2 / 3
    # 1 → {2,4}: (1,2) same, (1,4) differ; within the receivers (2,4) differ
    @test compute(HyperCovariate(group; aggregate=:catdiff), h, S, R, 0.0) == 0.5
    @test compute(HyperCovariate(group; endpoint=:receivers, aggregate=:catdiff), h, S, R, 0.0) == 1.0
    @test_throws ArgumentError HyperCovariate(group)                  # no mean of categories
    @test_throws ArgumentError HyperCovariate(group; aggregate=:absdiff)

    # a time-varying covariate is read at the candidate's time
    load = Covariate([0.0, 5.0], [1.0 10.0; 2.0 20.0]; name="load")
    @test compute(HyperCovariate(load), h, [1, 2], none, 1.0) == 1.5
    @test compute(HyperCovariate(load), h, [1, 2], none, 6.0) == 15.0

    @test name(HyperCovariate(x)) == "mean.x"
    @test name(HyperCovariate(Covariate(x; name="age"); endpoint=:receivers, aggregate=:absdiff)) ==
          "absdiff.age.receivers"
    @test_throws ArgumentError HyperCovariate(x; aggregate=:median)
    @test_throws ArgumentError HyperCovariate(x; endpoint=:targets)
    # an undirected hyperedge has no receivers; an actor without a value is an error
    @test_throws ArgumentError compute(HyperCovariate(x; endpoint=:receivers), h, c, none, 0.0)
    @test_throws ArgumentError compute(HyperCovariate(x), h, [1, 9], none, 0.0)
end

@testset "Hyperevents: interactions and transforms" begin
    h = build_hyper_history([HyperEvent([1, 2, 3], 1.0), HyperEvent([1, 2], 2.0)])
    none = Int[]
    # deg{1,2} = 2, deg{1,3} = deg{2,3} = 1 → mean 4/3; size 3
    by_size = Interaction(HyperedgeSize(), SubsetRepetition(2))
    @test compute(by_size, h, [1, 2, 3], none, 3.0) ≈ 3 * 4 / 3
    @test name(by_size) == "size:subrep(2)"
    centred = Transformed(SubsetRepetition(2), x -> x - 1)
    @test compute(centred, h, [1, 2], none, 3.0) == 1.0
    @test compute(Transformed(SubsetRepetition(2), sqrt), h, [1, 2], none, 3.0) ≈ sqrt(2)
    @test name(Transformed(SubsetRepetition(2), sqrt)) == "sqrt(subrep(2))"
    @test compute(by_size, h, HyperEvent([1, 2, 3], 3.0)) ≈ 4.0
end

@testset "Hyperevents: argument validation" begin
    @test_throws ArgumentError SubsetRepetition(0)
    @test_throws ArgumentError SubsetRepetition(2; aggregate=:median)
    @test_throws ArgumentError SubsetRepetition(2; memory=5.0)
    @test_throws ArgumentError SubsetRepetition(2; transform=:cube)
    @test_throws ArgumentError SharedPriorEvents(0)
    @test_throws ArgumentError PriorSuccess(0)
    @test_throws ArgumentError ExactRepetition(direction=:both)
    @test_throws ArgumentError DirectedSubsetRepetition(0, 0)
    @test_throws ArgumentError DirectedSubsetRepetition(-1, 2)
    @test_throws ArgumentError DirectedSubsetRepetition(1, 1; direction=:back)
    @test_throws ArgumentError DirectedSubsetRepetition(1, 1; aggregate=:catdiff)
    @test_throws ArgumentError ReceiverSetRepetition(0)
    @test_throws ArgumentError SenderReceiverSetRepetition(0)
    @test_throws ArgumentError InteractionAmongReceivers(0)
    @test_throws ArgumentError InteractionAmongReceivers(1; aggregate=:sd)

    # not implemented: refused with the reason, never silently approximated
    err = try SubsetRepetition(2; aggregate=:gw) catch e; e end
    @test err isa ArgumentError && occursin("geometrically weighted", err.msg)
    @test_throws ArgumentError DirectedSubsetRepetition(1, 1; aggregate=:geometric)

    events = [HyperEvent([1, 2], 1.0), HyperEvent([2, 3], 2.0), HyperEvent([1, 2], 3.0)]
    stats = [SubsetRepetition(1)]
    @test_throws ArgumentError hyper_design(HyperEvent{Float64}[], stats, 4)
    @test_throws ArgumentError hyper_design(events, stats, 1)
    @test_throws ArgumentError hyper_design(events, stats, 4; n_controls=0)
    @test_throws ArgumentError hyper_design(events, stats, 4; sampler=:importance)
    @test_throws ArgumentError hyper_design(events, [], 4)
    @test_throws ArgumentError hyper_design(events, [SubsetRepetition(1), SubsetRepetition(1)], 4)
    @test_throws ArgumentError hyper_design(events, [SubsetRepetition(1; name="senders")], 4)
    @test_throws ArgumentError hyper_design(events, [SubsetRepetition(1; name="stratum")], 4)
    # an observed participant outside the actors at risk
    @test_throws ArgumentError hyper_design(events, stats, 2)
    @test_throws ArgumentError hyper_design(events, stats, 4; actors=[1, 2, 4])
    @test_throws ArgumentError hyper_design(events, stats, 4; actors=[1, 2, 3, 9])
    # the only possible hyperedge of its size has no alternative
    @test_throws ArgumentError hyper_design([HyperEvent([1, 2, 3], 1.0)], stats, 3)
    # receiver-set sampling needs receivers
    @test_throws ArgumentError hyper_design(events, stats, 4; sampler=:receivers)
    # a dyadic statistic has no value on a hyperedge, and vice versa
    err = try hyper_design(events, [Inertia()], 4) catch e; e end
    @test err isa ArgumentError && occursin("not a hyperevent statistic", err.msg)
    @test_throws ArgumentError hyper_design(events, [Interaction(HyperedgeSize(), Inertia())], 4)
    @test_throws ArgumentError compute(SubsetRepetition(1), build_history(Event{Float64}[]), 1, 2, 0.0)
    # two-mode hyperevents (two actor sets) are not implemented
    err = try hyper_design(events, stats, 6; actors=(1:3, 4:6)) catch e; e end
    @test err isa ArgumentError && occursin("two-mode", err.msg)

    # tie policies: the shared vocabulary, with the unsupported ones refused
    tied = [HyperEvent([1, 2], 1.0), HyperEvent([2, 3], 1.0), HyperEvent([1, 2], 2.0)]
    @test_throws ArgumentError hyper_design(tied, stats, 4)               # ties=:error
    err = try hyper_design(tied, stats, 4; ties=:efron) catch e; e end
    @test err isa ArgumentError && occursin("ties=:breslow", err.msg)
    @test_throws ArgumentError hyper_design(tied, stats, 4; ties=:batch)
    @test_throws ArgumentError hyper_design(tied, stats, 4; ties=:whatever)

    @test_throws ArgumentError fit_rhem(HyperEvent{Float64}[], stats, 4)
    @test_throws ArgumentError fit_rhem(events, stats, 4; se=:bootstrap)
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0, 2.0], 5, 10; sizes=[2])
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0], 5, 10; sizes=Int[])
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0], 5, 10; sizes=[6])
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0], 5, 10; sizes=[0])
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0], 5, 10; sizes=[(1, 2)], directed=false)
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0], 5, 10; sizes=[(1, 0)])
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0], 5, 10; sizes=[2, (1, 2)])
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0], 5, -1; sizes=[2])
    @test_throws ArgumentError simulate_hyperevents(stats, [1.0], 5, 10; sizes=[2], candidates=1)
    @test_throws ArgumentError simulate_hyperevents([Inertia()], [1.0], 5, 10; sizes=[2])
end

@testset "Hyperevents: the case-control design" begin
    events = [HyperEvent([1, 2], 1.0), HyperEvent([1, 2, 3], 2.0), HyperEvent([1, 2], 3.0),
              HyperEvent([4, 5], 4.0)]
    stats = [SubsetRepetition(2), ExactRepetition()]
    design = hyper_design(events, stats, 6; n_controls=4, rng=Xoshiro(11))

    @test names(design) == ["event_index", "time", "is_event", "stratum", "risk_set_size",
                            "sampling_prob", "tie_weight", "senders", "receivers",
                            "subrep(2)", "exact.rep"]
    @test size(design, 1) == 4 * (1 + 4)
    @test design.event_index == repeat(1:4; inner=5)
    @test design.stratum == design.event_index
    @test design.time == repeat([1.0, 2.0, 3.0, 4.0]; inner=5)
    @test design.is_event == repeat([true, false, false, false, false]; outer=4)
    @test all(==(1.0), design.tie_weight)
    @test DataFrames.metadata(design, "tie_method") == "none"
    # the number of hyperedges of the observed size: C(6,2) = 15, C(6,3) = 20
    @test design.risk_set_size == repeat([15, 20, 15, 15]; inner=5)
    # four of the 14 (19) alternatives were kept
    @test design.sampling_prob ≈ repeat([4 / 14, 4 / 19, 4 / 14, 4 / 14]; inner=5)
    # the case row is the observed hyperedge
    @test design.senders[design.is_event] == [[1, 2], [1, 2, 3], [1, 2], [4, 5]]
    @test all(isempty, design.receivers)
    for m in 1:4
        rows = design[design.stratum .== m, :]
        observed = rows.senders[1]
        controls = rows.senders[2:end]
        @test all(c -> length(c) == length(observed), controls)      # the same size
        @test all(c -> c != observed, controls)                      # never the case
        @test allunique(controls)                                    # without replacement
        @test all(c -> issorted(c) && allunique(c) && all(in(1:6), c), controls)
    end
    # statistics are read strictly before the event:
    #   event 1: no history → 0, 0
    #   event 2 {1,2,3}: deg{1,2} = 1, the other pairs 0 → 1/3; exact 0
    #   event 3 {1,2}: deg{1,2} = 2 (events 1 and 2); exact 1 (event 1)
    #   event 4 {4,5}: 0, 0
    @test design[design.is_event, "subrep(2)"] ≈ [0.0, 1 / 3, 2.0, 0.0]
    @test design[design.is_event, "exact.rep"] == [0.0, 0.0, 1.0, 0.0]
    # every row carries the statistic of its own hyperedge at its event's history
    h = build_hyper_history(events[1:2])
    for row in eachrow(design[design.stratum .== 3, :])
        @test row["subrep(2)"] == compute(stats[1], h, row.senders, row.receivers, 3.0)
    end

    # all randomness comes from rng
    again = hyper_design(events, stats, 6; n_controls=4, rng=Xoshiro(11))
    @test again.senders == design.senders && again[!, "subrep(2)"] == design[!, "subrep(2)"]
    other = hyper_design(events, stats, 6; n_controls=4, rng=Xoshiro(12))
    @test other.senders != design.senders
    @test other.senders[other.is_event] == design.senders[design.is_event]

    # when the alternatives number n_controls or fewer, all are enumerated: the
    # risk set of each size is complete and no randomness is used
    full = hyper_design(events, stats, 6; n_controls=19, rng=Xoshiro(1))
    @test size(full, 1) == 15 + 20 + 15 + 15
    @test all(==(1.0), full.sampling_prob)
    for (m, N) in zip(1:4, (15, 20, 15, 15))
        rows = full.senders[full.stratum .== m]
        @test length(rows) == N && allunique(rows)
    end
    @test hyper_design(events, stats, 6; n_controls=500, rng=Xoshiro(2)).senders == full.senders
    # one control fewer than the alternatives of the triple: that stratum is sampled
    mixed = hyper_design(events, stats, 6; n_controls=18, rng=Xoshiro(1))
    @test count(mixed.stratum .== 2) == 19 && count(mixed.stratum .== 1) == 15
    @test mixed.sampling_prob[mixed.stratum .== 2][1] ≈ 18 / 19

    # a size-only statistic is constant within every stratum (not identified)
    sized = hyper_design(events, [HyperedgeSize()], 6; n_controls=4, rng=Xoshiro(3))
    @test all(m -> allequal(sized.size[sized.stratum .== m]), 1:4)

    # a restricted set of actors at risk
    pooled = hyper_design(events[1:3], stats, 6; n_controls=2, rng=Xoshiro(4),
                          actors=[1, 2, 3, 6])
    @test all(c -> all(in((1, 2, 3, 6)), c), pooled.senders)
    @test pooled.risk_set_size[pooled.is_event] == [6, 4, 6]        # C(4,2), C(4,3)

    # the sampler is uniform over the alternatives: 3000 draws of one control for
    # {1,2} among the five other pairs of four actors; each expects 600, with
    # binomial sd sqrt(3000 · 0.2 · 0.8) ≈ 21.9 — five sd is ±110
    many = [HyperEvent([1, 2], Float64(t)) for t in 1:3000]
    drawn = hyper_design(many, [HyperedgeSize()], 4; n_controls=1, rng=Xoshiro(5))
    controls = drawn.senders[.!drawn.is_event]
    alternatives = [[1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    @test sort(unique(controls)) == alternatives
    @test all(a -> abs(count(==(a), controls) - 600) < 110, alternatives)
end

@testset "Hyperevents: the design for directed hyperevents" begin
    events = [HyperEvent([1], [2, 3], 1.0), HyperEvent([2], [1], 2.0),
              HyperEvent([1], [2, 3], 3.0)]
    stats = [SenderReceiverSetRepetition(1), HyperReciprocation()]
    # directed hyperevents default to the sender-stratified design of Lerner &
    # Lomi (2023); the uniform design is asked for by name
    @test hyper_design(events, stats, 5; n_controls=6, rng=Xoshiro(21)).risk_set_size[1] == 6
    design = hyper_design(events, stats, 5; n_controls=6, rng=Xoshiro(21), sampler=:uniform)
    # C(5,1)·C(4,2) = 30 hyperedges with one sender and two receivers; C(5,1)·C(4,1) = 20
    @test design.risk_set_size[design.is_event] == [30, 20, 30]
    @test design.sampling_prob[design.is_event] ≈ [6 / 29, 6 / 19, 6 / 29]
    for m in 1:3
        rows = design[design.stratum .== m, :]
        case = (rows.senders[1], rows.receivers[1])
        controls = collect(zip(rows.senders[2:end], rows.receivers[2:end]))
        @test length(controls) == 6 && allunique(controls)
        @test all(c -> c != case, controls)
        @test all(c -> length(c[1]) == length(case[1]) && length(c[2]) == length(case[2]), controls)
        @test all(c -> isempty(intersect(c[1], c[2])), controls)     # loopless
        @test all(c -> issorted(c[2]) && all(in(1:5), c[1]) && all(in(1:5), c[2]), controls)
    end
    # event 3 (1 → {2,3}): sender 1 addressed 2 and 3 once each → 1.0; 2 replied
    # to 1 once, 3 never → 0.5
    @test design[design.is_event, "send.rec.subrep(1)"] == [0.0, 0.0, 1.0]
    @test design[design.is_event, "reciprocation"] == [0.0, 1.0, 0.5]

    # full enumeration
    full = hyper_design(events, stats, 5; n_controls=29, sampler=:uniform)
    @test [count(full.stratum .== m) for m in 1:3] == [30, 20, 30]
    @test allunique(collect(zip(full.senders, full.receivers))[full.stratum .== 1])

    # sampler=:receivers keeps the observed sender (Lerner & Lomi 2023): C(4,2) = 6
    # receiver sets for the pairs, C(4,1) = 4 for the single receiver
    fixed = hyper_design(events, stats, 5; n_controls=3, rng=Xoshiro(22), sampler=:receivers)
    @test fixed.risk_set_size[fixed.is_event] == [6, 4, 6]
    @test fixed.sampling_prob[fixed.is_event] ≈ [3 / 5, 3 / 3, 3 / 5]
    @test all(m -> allequal(fixed.senders[fixed.stratum .== m]), 1:3)
    for m in 1:3
        rows = fixed[fixed.stratum .== m, :]
        @test allunique(rows.receivers)
        @test all(r -> !(rows.senders[1][1] in r), rows.receivers)
    end
    @test count(fixed.stratum .== 2) == 4                     # all four enumerated
    # a statistic of the sender alone is constant within such a stratum
    act = hyper_design(events, [HyperSenderActivity()], 5; n_controls=3, rng=Xoshiro(23),
                       sampler=:receivers)
    @test all(m -> allequal(act[act.stratum .== m, "sender.activity"]), 1:3)
    # an alias of the design's RNG stream: reproducible
    @test hyper_design(events, stats, 5; n_controls=3, rng=Xoshiro(22),
                       sampler=:receivers).receivers == fixed.receivers
end

@testset "Hyperevents: risk-set sizes saturate instead of overflowing" begin
    @test Revel._binomial_sat(10, 3) == 120
    @test Revel._binomial_sat(10, 0) == 1 && Revel._binomial_sat(10, 10) == 1
    @test Revel._binomial_sat(10, 11) == 0 && Revel._binomial_sat(10, -1) == 0
    # the largest central binomial coefficient that fits an Int64, and the next one
    @test Revel._binomial_sat(66, 33) == binomial(big(66), 33) == 7219428434016265740
    @test binomial(big(67), 33) > typemax(Int)
    @test Revel._binomial_sat(67, 33) == typemax(Int)
    @test Revel._binomial_sat(1000, 500) == typemax(Int)
    @test Revel._binomial_sat(10^6, 2) == binomial(10^6, 2)
    # directed: C(n, p) · C(n − p, q); with the senders fixed, C(n − p, q)
    @test Revel._n_hyperedges(5, 1, 2, false) == 30
    @test Revel._n_hyperedges(5, 1, 2, true) == 6
    @test Revel._n_hyperedges(100, 30, 30, false) == typemax(Int)
    @test Revel._n_hyperedges(5, 3, 0, false) == 10

    # a design on a network far too large to enumerate
    big_events = [HyperEvent(collect(1:60), 1.0), HyperEvent(collect(31:90), 2.0)]
    design = hyper_design(big_events, [SubsetRepetition(1)], 200; n_controls=3, rng=Xoshiro(1))
    @test all(==(typemax(Int)), design.risk_set_size)
    @test all(p -> 0 < p < 1e-15, design.sampling_prob)
    @test size(design, 1) == 8 && all(s -> length(s) == 60, design.senders)
    # event 2 {31..90}: 30 of its 60 participants took part in event 1
    @test design[design.is_event, "subrep(1)"] == [0.0, 0.5]
end

@testset "Hyperevents: tied timestamps" begin
    tied = [HyperEvent([1, 2], 1.0), HyperEvent([1, 2], 1.0), HyperEvent([1, 2], 2.0)]
    stats = [ExactRepetition()]
    # :ordered — sequence order, the second tied event sees the first
    ordered = hyper_design(tied, stats, 4; n_controls=2, rng=Xoshiro(1), ties=:ordered)
    @test ordered[ordered.is_event, "exact.rep"] == [0.0, 1.0, 2.0]
    @test DataFrames.metadata(ordered, "tie_method") == "ordered"
    # :breslow — the history is frozen across the tied block
    breslow = hyper_design(tied, stats, 4; n_controls=2, rng=Xoshiro(1), ties=:breslow)
    @test breslow[breslow.is_event, "exact.rep"] == [0.0, 0.0, 2.0]
    @test DataFrames.metadata(breslow, "tie_method") == "breslow"
    @test breslow.stratum == repeat(1:3; inner=3)             # one stratum per event
    # without ties in the data the policy that applied is :none
    untied = [HyperEvent([1, 2], 1.0), HyperEvent([1, 2], 2.0)]
    @test DataFrames.metadata(hyper_design(untied, stats, 4; ties=:breslow), "tie_method") == "none"
    fit = fit_rhem([tied; [HyperEvent([3, 4], 3.0), HyperEvent([1, 2], 4.0),
                           HyperEvent([3, 4], 5.0), HyperEvent([2, 3], 6.0)]],
                   [ExactRepetition(transform=:log1p)], 4; ties=:breslow)
    @test NetworkCore.tie_method(fit) === :breslow
    @test fit.ties === :breslow
end

@testset "Hyperevents: simulation" begin
    stats = [SubsetRepetition(2; transform=:log1p), HyperCovariate([0.0, 1.0, 0.0, 1.0, 0.5, 2.0])]
    a = simulate_hyperevents(stats, [1.0, 0.5], 6, 60; sizes=[2, 3], rng=Xoshiro(42))
    b = simulate_hyperevents(stats, [1.0, 0.5], 6, 60; sizes=[2, 3], rng=Xoshiro(42))
    c = simulate_hyperevents(stats, [1.0, 0.5], 6, 60; sizes=[2, 3], rng=Xoshiro(43))
    @test a == b                                   # all randomness comes from rng
    @test a != c
    @test a isa Vector{HyperEvent{Float64}} && length(a) == 60
    @test [e.time for e in a] == collect(1.0:60.0)
    @test all(e -> !is_directed(e) && length(e.senders) in (2, 3), a)
    @test all(e -> all(in(1:6), e.senders), a)
    @test Set(length(e.senders) for e in a) == Set([2, 3])
    @test isempty(simulate_hyperevents(stats, [1.0, 0.5], 6, 0; sizes=[2]))

    # directed: size pairs, or one sender and k receivers with directed=true
    dstats = [ReceiverSetRepetition(1; transform=:log1p)]
    mails = simulate_hyperevents(dstats, [0.5], 6, 40; sizes=[(1, 2), (2, 1)], rng=Xoshiro(1))
    @test all(is_directed, mails)
    @test Set((length(e.senders), length(e.receivers)) for e in mails) == Set([(1, 2), (2, 1)])
    @test all(e -> isempty(intersect(e.senders, e.receivers)), mails)
    casts = simulate_hyperevents(dstats, [0.5], 6, 20; sizes=[3], directed=true, rng=Xoshiro(1))
    @test all(e -> length(e.senders) == 1 && length(e.receivers) == 3, casts)

    # a restricted actor set
    few = simulate_hyperevents(stats[1:1], [1.0], 6, 30; sizes=[2], actors=[2, 4, 6],
                               rng=Xoshiro(2))
    @test all(e -> all(in((2, 4, 6)), e.senders), few)

    # too many hyperedges to enumerate: a sampled candidate set, still reproducible
    s1 = simulate_hyperevents(stats[1:1], [1.0], 30, 25; sizes=[4], candidates=50, rng=Xoshiro(3))
    s2 = simulate_hyperevents(stats[1:1], [1.0], 30, 25; sizes=[4], candidates=50, rng=Xoshiro(3))
    @test s1 == s2 && all(e -> length(e.senders) == 4, s1)

    # a strong covariate effect shows: actor 6 (x = 2) is in most pairs, against
    # 1/3 of them under uniform choice
    pairs = simulate_hyperevents([HyperCovariate([0.0, 0.0, 0.0, 0.0, 0.0, 2.0]; aggregate=:sum)],
                                 [3.0], 6, 200; sizes=[2], rng=Xoshiro(4))
    @test count(e -> 6 in e.senders, pairs) > 150
end

@testset "Hyperevents: simulate and recover (undirected, exact risk set)" begin
    # Six actors and hyperedges of size 2 or 3: C(6,2) = 15 and C(6,3) = 20
    # hyperedges, so n_controls=20 enumerates every alternative and the
    # size-stratified conditional likelihood is exact. Across 100 replications of
    # this design at 300, 1000 and 3000 events the z-scores (estimate − truth)/se
    # had |mean| ≤ 0.18 and standard deviation between 0.95 and 1.13, so 4
    # standard errors is roughly a 1e-4 event per coefficient; the seed makes
    # the test deterministic.
    x = [0.0, 1.0, 0.0, 1.0, 0.5, -1.0]
    truth = [0.8, 0.6]
    stats = [SubsetRepetition(2; memory=HalfLife(30.0), transform=:log1p),
             HyperCovariate(x; name="x")]
    events = simulate_hyperevents(stats, truth, 6, 600; sizes=[2, 3], rng=Xoshiro(2024))
    fit = fit_rhem(events, stats, 6; n_controls=20, rng=Xoshiro(1))
    @test fit.fit.converged
    @test all(isfinite, stderror(fit)) && all(>(0), stderror(fit))
    @test all(abs.(coef(fit) .- truth) .< 4 .* stderror(fit))
    @test all(stderror(fit) .< 0.25)          # and the estimates are informative
    # every alternative was enumerated, so the control draw does not matter
    @test coef(fit_rhem(events, stats, 6; n_controls=20, rng=Xoshiro(99))) == coef(fit)
    @test all(==(1.0), fit.fit.sampling_probs)
end

@testset "Hyperevents: simulate and recover (sampled controls, closure)" begin
    # Eight actors, triples: C(8,3) = 56 hyperedges, 10 sampled controls per event.
    # Case-control sampling keeps the estimator consistent (Lerner & Lomi 2023);
    # the tolerance is again 4 standard errors of the sampled fit.
    x = [0.0, 1.0, 0.0, 1.0, 0.5, -1.0, 2.0, 0.3]
    truth = [0.8, 0.6, -0.4]
    stats = [SubsetRepetition(2; memory=HalfLife(30.0), transform=:log1p),
             HyperCovariate(x; name="x"),
             HyperClosure(memory=HalfLife(30.0), transform=:log1p)]
    events = simulate_hyperevents(stats, truth, 8, 600; sizes=[3], rng=Xoshiro(7))
    fit = fit_rhem(events, stats, 8; n_controls=10, rng=Xoshiro(8))
    @test fit.fit.converged
    @test all(abs.(coef(fit) .- truth) .< 4 .* stderror(fit))
    @test all(p -> p ≈ 10 / 55, fit.fit.sampling_probs)
    @test fit.fit.risk_set_sizes == fill(56, 600)
    # more controls: the same estimand, a different draw
    more = fit_rhem(events, stats, 8; n_controls=40, rng=Xoshiro(9))
    @test all(abs.(coef(more) .- truth) .< 4 .* stderror(more))
    @test all(abs.(coef(more) .- coef(fit)) .< 4 .* stderror(fit))
end

@testset "Hyperevents: simulate and recover (directed)" begin
    # Five actors, one sender and two or three receivers: 5·C(4,2) = 30 and
    # 5·C(4,3) = 20 hyperedges, all enumerated with n_controls=30.
    truth = [0.7, 0.5, 0.4]
    stats = [SenderReceiverSetRepetition(1; transform=:log1p),
             HyperReciprocation(transform=:log1p),
             ReceiverSetRepetition(2; transform=:log1p)]
    events = simulate_hyperevents(stats, truth, 5, 600; sizes=[(1, 2), (1, 3)], rng=Xoshiro(5))
    fit = fit_rhem(events, stats, 5; n_controls=30, rng=Xoshiro(6), sampler=:uniform)
    @test fit.fit.converged
    @test all(abs.(coef(fit) .- truth) .< 4 .* stderror(fit))
    @test Set(fit.fit.risk_set_sizes) == Set([30, 20])
    # the receiver-choice design of Lerner & Lomi (2023) targets the same
    # coefficients for statistics that vary across receiver sets
    choice = fit_rhem(events, stats, 5; n_controls=30, sampler=:receivers, rng=Xoshiro(6))
    @test Set(choice.fit.risk_set_sizes) == Set([6, 4])
    @test all(abs.(coef(choice) .- truth) .< 4 .* stderror(choice))
end

@testset "Hyperevents: dyadic special cases agree with the dyadic statistics" begin
    # Directed hyperevents with one sender and one receiver ARE dyadic events
    dyadic = simulate_events([Inertia(transform=:log1p), Reciprocation(transform=:log1p)],
                             [0.8, 0.6], 5, 80; rng=Xoshiro(3))
    hyper = [HyperEvent([e.sender], [e.receiver], e.time) for e in dyadic]
    for memory in (FullMemory(), HalfLife(10.0), Window(15.0))
        pairs = [
            DirectedSubsetRepetition(1, 1; memory=memory) => Inertia(memory=memory),
            ExactRepetition(memory=memory) => Inertia(memory=memory),
            HyperReciprocation(memory=memory) => Reciprocation(memory=memory),
            ExactRepetition(direction=:in, memory=memory) => Reciprocation(memory=memory),
            HyperSenderActivity(memory=memory) => OutdegreeSender(memory=memory),
            HyperReceiverPopularity(memory=memory) => IndegreeReceiver(memory=memory),
            OutInPopularity(memory=memory) => OutdegreeReceiver(memory=memory),
            HyperClosure(:transitive; memory=memory) => OTP(memory=memory),
            HyperClosure(:cyclic; memory=memory) => ITP(memory=memory),
            HyperClosure(:shared_receivers; memory=memory) => OSP(memory=memory),
            HyperClosure(:shared_senders; memory=memory) => ISP(memory=memory),
            HyperClosure(:transitive; combine=:product, memory=memory) =>
                OTP(combine=:product, memory=memory),
        ]
        for cut in (10, 45, 80)
            dh = build_history(dyadic[1:cut])
            hh = build_hyper_history(hyper[1:cut])
            t = dyadic[cut].time + 1.0
            for (hstat, dstat) in pairs, s in 1:5, r in 1:5
                s == r && continue
                @test compute(hstat, hh, [s], [r], t) ≈ compute(dstat, dh, s, r, t) atol = 1e-12
            end
        end
    end

    # … and the fits agree: with every alternative enumerated (5·4 = 20 dyads) the
    # uniform hyperevent design is the full dyadic risk set, and the default
    # sender-stratified one is the receiver-choice model
    dstats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    hstats = [DirectedSubsetRepetition(1, 1; transform=:log1p),
              HyperReciprocation(transform=:log1p)]
    dfit = fit_revel(dyadic, dstats, 5)
    hfit = fit_rhem(hyper, hstats, 5; n_controls=19, sampler=:uniform)
    @test coef(hfit) ≈ coef(dfit) atol = 1e-6
    @test stderror(hfit) ≈ stderror(dfit) atol = 1e-6
    @test loglikelihood(hfit) ≈ loglikelihood(dfit) atol = 1e-6
    @test nobs(hfit) == nobs(dfit) == 80
    cfit = fit_rhem(hyper, hstats, 5; n_controls=19)
    @test cfit.sampler === :receivers
    choice = fit_receiver_choice(dyadic, dstats, 5)
    @test coef(cfit) ≈ coef(choice) atol = 1e-6
    @test loglikelihood(cfit) ≈ loglikelihood(choice) atol = 1e-6

    # Undirected hyperevents with two participants are undirected dyadic events
    und = simulate_events([Inertia(symmetric=true, transform=:log1p)], [1.0], 5, 80;
                          rng=Xoshiro(4), directed=false)
    pairs2 = [HyperEvent([e.sender, e.receiver], e.time) for e in und]
    for memory in (FullMemory(), HalfLife(10.0)), cut in (10, 45, 80)
        dh = build_history(und[1:cut])
        hh = build_hyper_history(pairs2[1:cut])
        t = und[cut].time + 1.0
        inertia = Inertia(symmetric=true, memory=memory)
        for i in 1:5, j in (i + 1):5
            @test compute(ExactRepetition(memory=memory), hh, [i, j], Int[], t) ≈
                  compute(inertia, dh, i, j, t) atol = 1e-12
            @test compute(SubsetRepetition(2; memory=memory), hh, [i, j], Int[], t) ≈
                  compute(inertia, dh, i, j, t) atol = 1e-12
        end
    end
    ufit = fit_revel(und, [Inertia(symmetric=true, transform=:log1p)], 5; directed=false)
    pfit = fit_rhem(pairs2, [ExactRepetition(transform=:log1p)], 5; n_controls=9)
    @test coef(pfit) ≈ coef(ufit) atol = 1e-6
    @test loglikelihood(pfit) ≈ loglikelihood(ufit) atol = 1e-6
end

@testset "Hyperevents: the worked examples of Lerner & Lomi (2023)" begin
    # Figures 2–7 of Lerner & Lomi (2023, pp. 9–12): each figure gives a past
    # and a candidate hyperevent and the value of one statistic, computed by hand
    # in the paper. Actors A–G are 1–7.
    A, B, C, D, E, F, G = 1:7
    # Fig. 2 — partial receiver-set repetition of orders 1, 2, 3
    h = build_hyper_history([HyperEvent([A], [B, C, D, E], 1.0)])
    @test [compute(ReceiverSetRepetition(p), h, [A], [C, D, E, F], 2.0) for p in 1:3] ≈
          [3 / 4, 3 / 6, 1 / 4]
    @test all(iszero, [compute(SenderReceiverSetRepetition(p), h, [G], [C, D, E, F], 2.0)
                       for p in 1:3])
    # Fig. 3 — interaction among receivers of orders 1 and 2
    h = build_hyper_history([HyperEvent([A], [C, D, E], 1.0)])
    @test [compute(InteractionAmongReceivers(p), h, [F], [A, B, C, D], 2.0) for p in 1:2] ≈
          [2 / 12, 1 / 12]
    # Fig. 4 — reciprocation and out-in popularity
    h = build_hyper_history([HyperEvent([A], [D, E, F], 1.0), HyperEvent([B], [A, C], 2.0)])
    @test compute(HyperReciprocation(), h, [D], [A, B, C], 3.0) ≈ 1 / 3
    @test compute(OutInPopularity(), h, [D], [A, B, C], 3.0) ≈ 2 / 3
    # Fig. 5 — transitive and cyclic closure
    h = build_hyper_history([HyperEvent([A], [B, C], 1.0), HyperEvent([C], [D, E], 2.0)])
    @test compute(HyperClosure(:transitive), h, [A], [D, E], 3.0) ≈ 1.0
    @test compute(HyperClosure(:cyclic), h, [E], [A, F], 3.0) ≈ 1 / 2
    # Figs. 6 and 7 — incoming and outgoing balance
    h = build_hyper_history([HyperEvent([C], [A, B], 1.0), HyperEvent([C], [D, E], 2.0)])
    @test compute(HyperClosure(:shared_senders), h, [A], [D, E], 3.0) ≈ 1.0
    h = build_hyper_history([HyperEvent([A], [B, C], 1.0), HyperEvent([E], [C, D], 2.0)])
    @test compute(HyperClosure(:shared_receivers), h, [A], [D, E], 3.0) ≈ 1 / 2

    # Lerner et al. (2021): covariate homogeneity of a binary covariate, by its
    # definition, for every split of hyperedges of sizes 2–5
    xhom(k, n) = (v = abs(n - 2k) / n; isodd(n) ? (v - 1 / n) / (1 - 1 / n) : v)
    none = HyperHistory{Float64}()
    for n in 2:5, k in 0:n
        x = [ones(k); zeros(n - k)]
        @test compute(HyperCovariate(x; aggregate=:homogeneity), none, collect(1:n),
                      Int[], 0.0) ≈ xhom(k, n) atol = 1e-12
    end
    @test_throws ArgumentError HyperCovariate([0.0, 2.0]; aggregate=:homogeneity)
    # Lerner & Hâncean (2023): prior success disparity, the sample sd of the
    # members' summed outcomes
    papers = build_hyper_history([HyperEvent([1, 2], 1.0; weight=10.0),
                                  HyperEvent([2, 3], 2.0; weight=4.0)])
    @test compute(SubsetRepetition(1; weighted=true, aggregate=:samplesd), papers,
                  [1, 2, 3], Int[], 3.0) ≈ std([10.0, 14.0, 4.0])
end

@testset "Hyperevents: the fit wrapper" begin
    stats = [SubsetRepetition(1; transform=:log1p), SubsetRepetition(2; transform=:log1p)]
    events = simulate_hyperevents(stats, [0.4, 0.9], 7, 150; sizes=[2, 3], rng=Xoshiro(10))
    fit = fit_rhem(events, stats, 7; n_controls=15, rng=Xoshiro(11))
    @test fit isa HyperFit
    @test rhem === fit_rhem
    @test fit.fit isa REM.REMResult
    @test fit.events == events && fit.n_actors == 7 && fit.n_controls == 15
    @test fit.sampler === :uniform && fit.ties === :error
    @test coefnames(fit) == ["log1p(subrep(1))", "log1p(subrep(2))"]
    @test coef(fit) == coef(fit.fit) && length(coef(fit)) == 2
    @test stderror(fit) == stderror(fit.fit)
    @test vcov(fit) == vcov(fit.fit) && size(vcov(fit)) == (2, 2)
    @test confint(fit) == confint(fit.fit)
    @test loglikelihood(fit) == loglikelihood(fit.fit)
    @test nobs(fit) == 150 && dof(fit) == 2
    @test aic(fit) ≈ -2 * loglikelihood(fit) + 4
    @test bic(fit) == bic(fit.fit)
    @test coeftable(fit) isa NetworkCore.CoefficientTable
    # the StatsAPI surface and the result-metadata protocol, as for every model
    @test all(values(NetworkCore.check_statsapi(fit;
        required=(NetworkCore.STATSAPI_VERBS..., :coefnames), strict=true)))
    @test coefnames(fit) == coeftable(fit).names == coefnames(fit.fit)
    @test coefnames === NetworkCore.coefnames === REM.coefnames
    @test NetworkCore.estimand(fit) == NetworkCore.estimand(fit.fit)
    @test NetworkCore.objective(fit) == NetworkCore.objective(fit.fit)
    @test NetworkCore.is_exact(fit) == NetworkCore.is_exact(fit.fit)
    @test NetworkCore.se_method(fit) === :hessian
    @test NetworkCore.tie_method(fit) === :none
    @test NetworkCore.missing_method(fit) == NetworkCore.missing_method(fit.fit)
    # REM's sampling note speaks of dyads and suggests a full risk set; the
    # hyperevent fit says it in hyperedges and leaves the other notes alone
    notes = NetworkCore.approximations(fit)
    @test length(notes) == length(NetworkCore.approximations(fit.fit))
    @test any(occursin("case-control sampling of hyperedges", x) for x in notes)
    @test !any(occursin("control_draw_cov", x) || occursin("dyad", x) for x in notes)
    for (what, call) in ["event_diagnostics" => () -> event_diagnostics(fit),
                         "prediction_summary" => () -> prediction_summary(fit),
                         "score_process_test" => () -> score_process_test(fit),
                         "score_test" => () -> score_test(fit, stats[1:1])]
        @test_throws ArgumentError call()
    end
    @test NetworkCore.fit_metadata(fit) isa NetworkCore.ResultMetadata
    @test occursin("relational hyperevent model", sprint(show, MIME"text/plain"(), fit))
    @test !occursin("control_draw_cov", sprint(show, MIME"text/plain"(), fit))

    # the fit is the conditional logit of its own design
    design = hyper_design(events, stats, 7; n_controls=15, rng=Xoshiro(11))
    direct = REM.fit_rem(design, [name(s) for s in stats])
    @test coef(fit) == coef(direct)
    # reproducible from rng; a different draw of controls moves the estimate a little
    @test coef(fit_rhem(events, stats, 7; n_controls=15, rng=Xoshiro(11))) == coef(fit)
    @test coef(fit_rhem(events, stats, 7; n_controls=15, rng=Xoshiro(12))) != coef(fit)
    # events in any order: the fit sorts them
    @test coef(fit_rhem(reverse(events), stats, 7; n_controls=15, rng=Xoshiro(11))) == coef(fit)
    # sandwich standard errors come from REM.fit_rem too
    robust = fit_rhem(events, stats, 7; n_controls=15, rng=Xoshiro(11), se=:sandwich)
    @test NetworkCore.se_method(robust) === :sandwich && coef(robust) == coef(fit)

    # goodness of fit is not implemented for hyperevent fits
    err = try gof(fit) catch e; e end
    @test err isa ArgumentError && occursin("not implemented", err.msg)
end

# ----------------------------------------------------------------------------
# Golden fixture: remstats 4.1.0
# ----------------------------------------------------------------------------
#
# test/fixtures/revel_remstats.toml is generated by test/fixtures/r/revel_remstats.R
# from R remstats 4.1.0 / remify 4.1.0:
#
#   Rscript test/fixtures/r/revel_remstats.R > test/fixtures/revel_remstats.toml

@testset "remstats design parity (golden)" begin
    g = NetworkCore.load_golden(joinpath(pkgdir(Revel), "test", "fixtures",
                                      "revel_remstats.toml"))
    n = Int(g.values["n_actors"])
    times = Float64.(g.values["input_time"])
    senders = Int.(g.values["input_sender"])
    receivers = Int.(g.values["input_receiver"])
    weights = Float64.(g.values["input_weight"])
    events = [Event(s, r, t) for (s, r, t) in zip(senders, receivers, times)]
    weighted_events = [Event(s, r, t; weight=w)
                       for (s, r, t, w) in zip(senders, receivers, times, weights)]
    x = Float64.(g.values["covariate_x"])
    grp = Covariate(String.(g.values["covariate_g"]); name="g")
    d = permutedims(reshape(Float64.(g.values["covariate_d"]), n, n))   # stored row-major
    window = Window(g.values["window_width"])
    decay = HalfLife(g.values["decay_halflife"])
    interval = IntervalMemory(g.values["interval_lo"], g.values["interval_hi"])

    # The fixture's layout: event-major, then dyads sender 1..n, receiver 1..n,
    # sender != receiver (directed) or (i, j), i < j (undirected). Each row is
    # computed from the events BEFORE it; `at(k)` is the clock the k-th row is
    # evaluated at (the k-th event's own time unless stated otherwise).
    event_time(evs) = k -> evs[k].time
    function via_history(stat, evs; directed=true, at=event_time(evs))
        history = InteractionHistory()
        out = Float64[]
        for (k, event) in enumerate(evs)
            t = at(k)
            for s in 1:n, r in (directed ? (1:n) : ((s + 1):n))
                s == r && continue
                push!(out, compute(stat, history, s, r, t))
            end
            update_history!(history, event)
        end
        return out
    end
    # The same design through REM's interface. As in REM's own fitters, the
    # state's clock is moved to the evaluation time before the statistics are read.
    function via_state(stat, evs; directed=true, at=event_time(evs))
        state = REM.EventNetworkState{Float64}()
        out = Float64[]
        for (k, event) in enumerate(evs)
            state.current_time = at(k)
            for s in 1:n, r in (directed ? (1:n) : ((s + 1):n))
                s == r && continue
                push!(out, compute(stat, state, s, r))
            end
            REM.update!(state, event)
        end
        return out
    end

    covered = Set{String}()
    # One fixture key: Revel reproduces remstats through the history interface,
    # and the REM interface returns the identical array.
    function parity(key, stat; evs=events, directed=true)
        push!(covered, key)
        actual = via_history(stat, evs; directed=directed)
        @testset "$key" begin
            @test actual == via_state(stat, evs; directed=directed)
            @test NetworkCore.check_golden(g, key, actual)
        end
        return actual
    end

    p_dyad, p_node = 1 / (n - 1), 1 / n     # remstats' zero-history values
    n_dyads = n * (n - 1)

    # ---- full memory, directed ------------------------------------------------
    for (key, stat) in [
            "full_inertia" => Inertia(),
            "full_reciprocity" => Reciprocation(),
            "full_indegreeSender" => IndegreeSender(),
            "full_indegreeReceiver" => IndegreeReceiver(),
            "full_outdegreeSender" => OutdegreeSender(),
            "full_outdegreeReceiver" => OutdegreeReceiver(),
            "full_totaldegreeSender" => TotaldegreeSender(),
            "full_totaldegreeReceiver" => TotaldegreeReceiver(),
            "full_totaldegreeDyad" => TotaldegreeDyad(),
            "full_otp" => OTP(), "full_itp" => ITP(),
            "full_osp" => OSP(), "full_isp" => ISP(),
            "full_otp_unique" => OTP(combine=:count),
            "full_itp_unique" => ITP(combine=:count),
            "full_osp_unique" => OSP(combine=:count),
            "full_isp_unique" => ISP(combine=:count),
            "full_psABBA" => PShift(:AB_BA), "full_psABBY" => PShift(:AB_BY),
            "full_psABXA" => PShift(:AB_XA), "full_psABXB" => PShift(:AB_XB),
            "full_psABXY" => PShift(:AB_XY), "full_psABAY" => PShift(:AB_AY),
            "full_psABAB" => PShiftABAB(),
            "full_rrankSend" => RecencyRank(:send),
            "full_rrankReceive" => RecencyRank(:receive),
            # 1 / (time since … + 1), measured to the current event's time; 0
            # when there has been no such event (both packages)
            "full_recencyContinue" => TimeSince(:dyad),
            "full_recencySendSender" => TimeSince(:send_sender),
            "full_recencySendReceiver" => TimeSince(:send_receiver),
            "full_recencyReceiveSender" => TimeSince(:receive_sender),
            "full_recencyReceiveReceiver" => TimeSince(:receive_receiver),
            # scaling = "prop": 1/(n-1) for a sender without sends (inertia) or
            # receipts (reciprocity); 1/n before the first event (degrees)
            "full_inertia_prop" => Inertia(scaling=:prop, empty=p_dyad),
            "full_reciprocity_prop" => Reciprocation(scaling=:prop, empty=p_dyad),
            "full_indegreeSender_prop" => IndegreeSender(scaling=:prop, empty=p_node),
            "full_indegreeReceiver_prop" => IndegreeReceiver(scaling=:prop, empty=p_node),
            "full_outdegreeSender_prop" => OutdegreeSender(scaling=:prop, empty=p_node),
            "full_outdegreeReceiver_prop" => OutdegreeReceiver(scaling=:prop, empty=p_node),
            "full_totaldegreeSender_prop" => TotaldegreeSender(scaling=:prop, empty=p_node),
            "full_totaldegreeReceiver_prop" => TotaldegreeReceiver(scaling=:prop, empty=p_node),
            "full_totaldegreeDyad_prop" => TotaldegreeDyad(scaling=:prop, empty=p_node),
            # exogenous
            "exo_send_x" => SendEffect(x), "exo_receive_x" => ReceiveEffect(x),
            "exo_same_g" => MatchEffect(grp),
            "exo_difference_x_abs" => DiffEffect(x),
            "exo_difference_x" => DiffEffect(x; absolute=false),   # x[sender] - x[receiver]
            "exo_average_x" => AverageEffect(x), "exo_minimum_x" => MinimumEffect(x),
            "exo_maximum_x" => MaximumEffect(x), "exo_tie_d" => TieEffect(d),
            # a:b
            "full_inertia_x_send_x" => Interaction(Inertia(), SendEffect(x)),
            "full_outdegreeSender_x_indegreeReceiver" =>
                Interaction(OutdegreeSender(), IndegreeReceiver())]
        parity(key, stat)
    end

    # ---- scaling = "std" --------------------------------------------------------
    # remstats divides by the SAMPLE standard deviation of the risk set
    # (arma::stddev, denominator D - 1): `Standardized(…; corrected=true)`. The
    # default population form (denominator D) differs by the constant
    # sqrt((D - 1) / D). Both return 0 for a statistic that is constant across
    # the risk set.
    for (key, stat) in ["full_inertia_std" => Inertia(),
                        "full_indegreeReceiver_std" => IndegreeReceiver(),
                        "full_otp_std" => OTP(), "full_send_x_std" => SendEffect(x)]
        parity(key, Standardized(stat, n; corrected=true))
        plain = via_history(Standardized(stat, n), events)
        @test maximum(abs, plain .* sqrt((n_dyads - 1) / n_dyads) .- g.values[key]) < 1e-12
        @test !NetworkCore.check_golden(g, key, plain)     # the population form is NOT remstats' "std"
    end

    # ---- memory = "window" and "interval" ---------------------------------------
    # remstats keeps an event of age a when a <= width ("window") and when
    # lo < a <= hi ("interval"): exactly `Window` and `IntervalMemory`. The time grid
    # makes ages equal to the width and to both interval bounds occur.
    # Ranks, recencies and participation shifts ignore the memory in both packages.
    memory_effects(memory) = [
        "inertia" => Inertia(memory=memory),
        "reciprocity" => Reciprocation(memory=memory),
        "outdegreeSender" => OutdegreeSender(memory=memory),
        "indegreeReceiver" => IndegreeReceiver(memory=memory),
        "totaldegreeDyad" => TotaldegreeDyad(memory=memory),
        "otp" => OTP(memory=memory), "itp" => ITP(memory=memory),
        "osp" => OSP(memory=memory), "isp" => ISP(memory=memory),
        "otp_unique" => OTP(memory=memory, combine=:count),
        "inertia_prop" => Inertia(memory=memory, scaling=:prop, empty=p_dyad),
        "outdegreeSender_prop" => OutdegreeSender(memory=memory, scaling=:prop, empty=p_node),
        "rrankSend" => RecencyRank(:send),               # takes no memory
        "recencyContinue" => TimeSince(:dyad),           # takes no memory
        "psABBA" => PShift(:AB_BA)]
    for (key, stat) in memory_effects(window)
        parity("window_" * key, stat)
    end
    for (key, stat) in memory_effects(interval)
        key == "outdegreeSender_prop" && continue
        parity("interval_" * key, stat)
    end
    # The boundary conventions are really exercised by the fixture:
    @test !NetworkCore.check_golden(g, "window_inertia", via_history(
        Inertia(memory=KernelMemory(a -> a < window.width ? 1.0 : 0.0)), events))
    @test !NetworkCore.check_golden(g, "interval_inertia", via_history(
        Inertia(memory=KernelMemory(a -> interval.lo <= a <= interval.hi ? 1.0 : 0.0)), events))
    @test !NetworkCore.check_golden(g, "interval_inertia", via_history(
        Inertia(memory=KernelMemory(a -> interval.lo < a < interval.hi ? 1.0 : 0.0)), events))

    let key = "interval_outdegreeSender_prop"
        push!(covered, key)
        actual = via_history(OutdegreeSender(memory=interval, scaling=:prop, empty=p_node), events)
        # A deliberate difference. remstats returns 1/n at the FIRST time point
        # only and 0 at a later one whose memory holds no event (0/0 → NaN → 0);
        # Revel returns `empty` whenever there is nothing to take a share of.
        @test !NetworkCore.check_golden(g, key, actual)
        # … and that is the whole difference: zero the later empty-memory rows
        in_memory = reshape(via_history(Inertia(memory=interval), events), n_dyads, :)
        patched = reshape(copy(actual), n_dyads, :)
        for k in 2:length(events)
            iszero(sum(view(in_memory, :, k))) && (patched[:, k] .= 0.0)
        end
        @test NetworkCore.check_golden(g, key, vec(patched))
    end

    # ---- memory = "decay" -------------------------------------------------------
    # remstats weighs a past event by exp(-(t_prev - t_event)·ln2/halflife), with
    # t_prev the time of the PREVIOUS event, not of the event being explained
    # (its help page says "the elapsed time between t and the past event"), and
    # does not multiply by ln2/halflife. `HalfLife(h)` evaluated at the previous
    # event's time therefore reproduces it; evaluated at the event's own time —
    # what Revel's fitters do — it is smaller by 2^(-(t - t_prev)/halflife).
    previous_time(evs) = k -> k == 1 ? 0.0 : evs[k - 1].time
    function decay_parity(key, stat; evs=events)
        push!(covered, key)
        @testset "$key" begin
            lagged = via_history(stat, evs; at=previous_time(evs))
            @test lagged == via_state(stat, evs; at=previous_time(evs))
            @test NetworkCore.check_golden(g, key, lagged)
            # A deliberate difference: Revel evaluates the decay at the time of
            # the event being explained, as remstats documents. The two then
            # differ by exactly the decay over the last waiting time.
            own = via_history(stat, evs)
            @test !NetworkCore.check_golden(g, key, own)
            gap = [evs[k].time - previous_time(evs)(k) for k in eachindex(evs)]
            factor = repeat(2.0 .^ (-gap ./ decay.halflife); inner=n_dyads)
            @test maximum(abs, own .- factor .* g.values[key]) < 1e-10
        end
    end
    for (key, stat) in memory_effects(decay)
        if key in ("otp_unique", "inertia_prop", "outdegreeSender_prop", "rrankSend",
                   "recencyContinue", "psABBA")
            # ratios, supports, ranks and recencies do not depend on where the
            # decay is evaluated
            parity("decay_" * key, stat)
        else
            decay_parity("decay_" * key, stat)
        end
    end

    # ---- ordinal = TRUE: remify replaces the times by the event index -----------
    for (key, stat) in [
            "ordinal_window_inertia" => Inertia(memory=window, clock=:order),
            "ordinal_window_otp" => OTP(memory=window, clock=:order),
            "ordinal_recencyContinue" => TimeSince(:dyad; clock=:order),
            "ordinal_recencyReceiveSender" => TimeSince(:receive_sender; clock=:order)]
        parity(key, stat)
    end

    # ---- undirected events (dyads (i, j), i < j) --------------------------------
    # remstats' undirected degree of an actor is the number of events it took
    # part in, which is what every degree kind of a `symmetric=true` layer is.
    for (key, stat) in [
            "undirected_inertia" => Inertia(symmetric=true),
            "undirected_totaldegreeDyad" => TotaldegreeDyad(symmetric=true),
            "undirected_degreeMin" => DegreeMin(symmetric=true),
            "undirected_degreeMax" => DegreeMax(symmetric=true),
            "undirected_degreeDiff" => DegreeDiff(symmetric=true),
            "undirected_sp" => SharedPartners(),
            "undirected_sp_unique" => SharedPartners(combine=:count),
            "undirected_recencyContinue" => TimeSince(:dyad; symmetric=true),
            # the previous event's pair again, in either orientation
            "undirected_psABAB" => UndirectedPShift(:AB_AB),
            # exactly one actor in common with the previous event
            "undirected_psABAY" => UndirectedPShift(:AB_AY),
            # remstats divides degreeMin/degreeMax by the number of past events …
            "undirected_degreeMin_prop" =>
                DegreeMin(symmetric=true, scaling=:prop, empty=p_node),
            "undirected_degreeMax_prop" =>
                DegreeMax(symmetric=true, scaling=:prop, empty=p_node),
            # … but totaldegreeDyad by TWICE that number: the total degrees of a
            # plain (directed) layer, whose in + out is the same involvement count
            "undirected_totaldegreeDyad_prop" => TotaldegreeDyad(scaling=:prop, empty=p_node),
            "undirected_window_inertia" => Inertia(symmetric=true, memory=window),
            "undirected_window_degreeMin" => DegreeMin(symmetric=true, memory=window),
            "undirected_window_sp" => SharedPartners(memory=window)]
        parity(key, stat; directed=false)
    end
    # The constructions on a plain (directed) layer that also reproduce remstats:
    for (key, stat) in ["undirected_inertia" => DyadActivity(),
                        "undirected_totaldegreeDyad" => TotaldegreeDyad(),
                        "undirected_degreeMin" => DegreeMin(),
                        "undirected_degreeMax" => DegreeMax(),
                        "undirected_degreeDiff" => DegreeDiff(),
                        "undirected_recencyContinue" => TimeSince(:pair)]
        @test NetworkCore.check_golden(g, key, via_history(stat, events; directed=false))
    end
    # remstats' two undirected `prop` degrees disagree with each other; Revel's
    # symmetric layer divides both by the number of past events, so its
    # totaldegreeDyad share is exactly twice remstats' after the first event
    let twice = via_history(TotaldegreeDyad(symmetric=true, scaling=:prop, empty=p_node),
                            events; directed=false)
        rows = (n_dyads ÷ 2 + 1):length(twice)
        @test maximum(abs, twice[rows] .- 2 .* g.values["undirected_totaldegreeDyad_prop"][rows]) < 1e-12
    end
    # a directed participation shift misses a pair stored the other way round
    @test !NetworkCore.check_golden(g, "undirected_psABAB",
                                 via_history(PShiftABAB(), events; directed=false))

    # ---- event weights (a `weight` column in remify's edgelist) -------------------
    for (key, stat) in [
            "weighted_inertia" => Inertia(weighted=true),
            "weighted_outdegreeSender" => OutdegreeSender(weighted=true),
            "weighted_indegreeReceiver" => IndegreeReceiver(weighted=true),
            "weighted_reciprocity" => Reciprocation(weighted=true),
            "weighted_otp" => OTP(weighted=true),
            "weighted_otp_unique" => OTP(weighted=true, combine=:count),
            "weighted_inertia_prop" => Inertia(weighted=true, scaling=:prop, empty=p_dyad),
            "weighted_outdegreeSender_prop" =>
                OutdegreeSender(weighted=true, scaling=:prop, empty=p_node),
            "weighted_recencyContinue" => TimeSince(:dyad)]
        parity(key, stat; evs=weighted_events)
    end
    decay_parity("weighted_decay_inertia", Inertia(weighted=true, memory=decay);
                 evs=weighted_events)
    decay_parity("weighted_decay_outdegreeSender", OutdegreeSender(weighted=true, memory=decay);
                 evs=weighted_events)

    # ---- products and event attributes (remstats `a:b`, `event()`) ---------------
    parity("full_send_x_x_receive_x", ProductEffect(x, x))
    # remstats' event(z) is the attribute of the event being explained, constant
    # over its risk set: a GlobalEffect stepping at the event times
    z = Float64.(g.values["event_z"])
    parity("event_z_x_inertia", Interaction(Inertia(), GlobalEffect(times, z)))

    # ---- event types: consider_type = "separate" is one statistic per past type --
    types = Symbol.(g.values["input_type"])
    typed_events = [Event(s, r, t; eventtype=ty)
                    for (s, r, t, ty) in zip(senders, receivers, times, types)]
    for ty in (:a, :b), (key, stat) in [
            "inertia" => Inertia(types=ty), "reciprocity" => Reciprocation(types=ty),
            "otp" => OTP(types=ty), "outdegreeSender" => OutdegreeSender(types=ty)]
        k = "typed_$(key)_$(ty)"
        push!(covered, k)
        @test NetworkCore.check_golden(g, k, via_history(stat, typed_events))
        # REM's event log carries no types, so the REM interface refuses it
        @test_throws ArgumentError via_state(stat, typed_events)
    end
    @test NetworkCore.check_golden(g, "typed_inertia_a",
                                via_history(split_by_type(Inertia, [:a, :b])[1], typed_events))

    # Every statistic array in the fixture is asserted above
    arrays = [k for (k, v) in g.values if v isa AbstractVector &&
              length(v) in (length(events) * n_dyads, length(events) * n_dyads ÷ 2)]
    @test length(arrays) == 141
    @test isempty(setdiff(arrays, covered))
    @test isempty(setdiff(covered, arrays))
end

# ----------------------------------------------------------------------------
# Regressions: tied events, case predicates, singular fits
# ----------------------------------------------------------------------------

@testset "ties=:efron with sampled controls is refused (REM's guard); :breslow is unbiased" begin
    # `event_design` used to force the other tied
    # cases into every stratum of an Efron block while sampling the controls,
    # which biased the estimate toward zero (0.654 for a truth of 1.0 at 10
    # controls). The refusal is REM.jl's `check_tie_sampling` — the same rule,
    # the same words — and Breslow, whose strata are nested-case-control
    # strata, is the correction for a sampled design.
    n = 20
    xa = randn(Xoshiro(5), n); za = randn(Xoshiro(6), n)
    st = [SendEffect(Covariate(xa)), ReceiveEffect(Covariate(za))]
    β = [1.0, -0.7]
    times = Float64[div(k - 1, 4) + 1 for k in 1:240]          # ticks of 4 events
    ev = simulate_events(st, β, n, 240; rng=Xoshiro(1), times=times, ties=:efron)
    for m in (5, 10, 100)
        err = try
            fit_revel(ev, st, n; ties=:efron, n_controls=m, rng=Xoshiro(2)); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("cannot be combined with sampled controls", err.msg)
        @test occursin("ties=:breslow", err.msg) && occursin("event_design", err.msg)
        @test_throws ArgumentError event_design(ev, st, n; ties=:efron, n_controls=m)
    end
    @test parentmodule(REM.check_tie_sampling) === REM && Base.ispublic(REM, :check_tie_sampling)
    # the full risk set (n_controls at least the eligible controls) is Efron's
    full_rows = event_design(ev[1:8], st, n; ties=:efron, n_controls=n * (n - 1))
    @test metadata(full_rows, "tie_method") == "efron"
    @test all(==(1.0), full_rows.sampling_prob)
    # simulation: d = 4 distinct dyads per tick on a frozen history (what
    # `simulate_events(...; ties=:efron)` draws), truth β; measured at 400
    # replicates: Breslow 0.999 / 0.997 at 5 / 10 controls, full Efron 1.001,
    # Wald coverage 0.93–0.96
    nrep = 150
    cfg = ((:breslow, 5), (:breslow, 10), (:efron, nothing))
    est = Dict(c => zeros(nrep, 2) for c in cfg)
    cover = Dict(c => zeros(Int, 2) for c in cfg)
    rng = Xoshiro(2026)
    for r in 1:nrep
        evr = simulate_events(st, β, n, 240; rng=rng, times=times, ties=:efron)
        for c in cfg
            f = fit_revel(evr, st, n; ties=c[1], n_controls=c[2], rng=rng)
            est[c][r, :] .= coef(f)
            ci = confint(f)
            cover[c] .+= [ci[j, 1] <= β[j] <= ci[j, 2] for j in 1:2]
        end
    end
    for c in cfg
        mu = vec(mean(est[c]; dims=1))
        mcse = vec(std(est[c]; dims=1)) ./ sqrt(nrep)
        @test all(abs.(mu .- β) .< 0.03 .+ 3 .* mcse)
        @test all(0.88 .<= cover[c] ./ nrep .<= 0.99)
    end
end

@testset "simulate_events(ties=:efron) draws a tie block without replacement" begin
    # stats LOW: the simulator drew dyads WITH replacement inside a tie block,
    # and `fit_revel(...; ties=:efron)` then refused the sequence it had made
    # ("a dyad acts twice") — 4 of 10 seeds on 8 actors in pairs.
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    times = Float64[div(k - 1, 2) + 1 for k in 1:60]
    for sd in 1:10
        evt = simulate_events(stats, [1.0, 0.5], 8, 60; rng=Xoshiro(sd), times=times,
                              ties=:efron)
        for t in unique(times)
            block = [(e.sender, e.receiver) for e in evt if e.time == t]
            @test allunique(block)
        end
        @test fit_revel(evt, stats, 8; ties=:efron) isa RevelFit
    end
    # Breslow's model is independent draws (with replacement): a repeated dyad
    # within a block stays possible, and the fitter accepts it
    many = simulate_events([Inertia()], [3.0], 3, 40; rng=Xoshiro(1),
                           times=Float64[div(k - 1, 8) + 1 for k in 1:40], ties=:breslow,
                           history=[Event(1, 2, 0.0)])
    @test any(t -> !allunique([(e.sender, e.receiver) for e in many if e.time == t]),
              unique(e.time for e in many))
    # a block larger than the risk set cannot be drawn without replacement
    @test_throws ArgumentError simulate_events([Inertia()], [1.0], 2, 3; rng=Xoshiro(1),
                                               times=[1.0, 1.0, 1.0], ties=:efron)
end

@testset "A dyad-dependent `cases` predicate restricts the risk set" begin
    # `fit_revel(...; cases = e -> grp[e.sender] == 2)` used to keep the full
    # risk set, so the sender covariate "explained" the
    # selection: 2.57 (2.67 over 8 replicates) against a truth of 0.5.
    grp = [1, 1, 1, 2, 2, 2, 2, 2]
    x = Covariate(Float64.(1:8) ./ 8)
    truth = [Inertia(transform=:log1p), SendEffect(x)]
    pred = e -> grp[e.sender] == 2
    ev = simulate_events(truth, [0.8, 0.5], 8, 600; rng=Xoshiro(301))
    fit = fit_revel(ev, truth, 8; cases=pred)
    # the conditional likelihood is fit_stratified's stratum, exactly
    strat = fit_stratified(ev, truth, 8; by = e -> grp[e.sender])
    @test coef(fit) ≈ coef(strat[2]) atol = 1e-10
    @test stderror(fit) ≈ stderror(strat[2]) atol = 1e-10
    # each case's risk set is the 5 group-2 senders × their 7 receivers
    d = event_design(ev, truth, 8; cases=pred)
    @test all(==(35), d.risk_set_size)
    @test all(grp[s] == 2 for s in d.sender)
    @test fit.riskset isa Revel._CaseRestricted
    @test occursin("`cases` predicate", sprint(show, MIME"text/plain"(), fit))
    # the diagnostics rebuild the same strata from the fit
    @test size(event_diagnostics(fit), 1) == count(pred, ev)
    # a predicate that does not depend on the dyad keeps the risk set
    late = fit_revel(ev, truth, 8; cases = e -> e.time > 300)
    @test late.riskset === :full
    @test coef(late) ≈ coef(fit_revel(ev, truth, 8; cases=findall(e -> e.time > 300,
                                                                     sort(ev; by=e -> e.time)))) atol = 1e-10
    # simulation: the sender effect is recovered (it was ≈ 2.6 before)
    est = [coef(fit_revel(simulate_events(truth, [0.8, 0.5], 8, 600; rng=Xoshiro(300 + r)),
                          truth, 8; cases=pred))[2] for r in 1:8]
    @test abs(mean(est) - 0.5) < 3 * std(est) / sqrt(8) + 0.05
    # Standardized over a fixed set cannot follow a per-case risk set
    @test_throws ArgumentError fit_revel(ev, [Standardized(Inertia(), 8)], 8; cases=pred)
end

# A copy of the struct `x` with the named fields replaced and every other field
# kept, so that a test does not depend on the field order or count of a type
# it does not own.
function _with_fields(x::T; changes...) where T
    unknown = setdiff(keys(changes), fieldnames(T))
    isempty(unknown) || error("$(T) has no field(s) $(collect(unknown))")
    return T((haskey(changes, f) ? changes[f] : getfield(x, f) for f in fieldnames(T))...)
end

@testset "Diagnostics refuse a fit with singular information" begin
    # A converged fit whose standard errors are not finite has coefficients
    # that are not determined along the null direction of the information:
    # every diagnostic refuses it, naming the statistics.
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    ev = simulate_events(stats, [0.8, 0.6], 6, 120; rng=Xoshiro(3))
    good = fit_revel(ev, stats, 6; engine=:design)
    r = good.fit
    # Built by field name, so the test survives REM adding a field to its result
    nan_se = _with_fields(r; std_errors=[NaN, r.std_errors[2]], converged=true,
                          var_cov=fill(NaN, 2, 2), singular=false,
                          singular_suspects=String[], separated=String[])
    sing = _with_fields(good; fit=nan_se)
    for f in (() -> event_diagnostics(sing), () -> prediction_summary(sing),
              () -> score_test(sing, [OTP()]), () -> score_process_test(sing; n_sim=5),
              () -> gof(sing; n_sim=2, rng=Xoshiro(1)))
        err = try f(); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("singular", err.msg) && occursin("log1p(inertia)", err.msg)
    end
    @test prediction_summary(good).n_events == 120          # the identified fit runs
end

@testset "RevelFit is concretely typed" begin
    stats = [Inertia(transform=:log1p)]
    ev = simulate_events(stats, [1.0], 5, 40; rng=Xoshiro(4))
    for fit in (fit_revel(ev, stats, 5), fit_revel(ev, stats, 5; riskset=:sender),
                fit_revel(ev, stats, 5; n_controls=3, rng=Xoshiro(1)))
        @test all(isconcretetype(fieldtype(typeof(fit), f))
                  for f in (:fit, :events, :riskset, :n_actors, :model))
    end
end

@testset "The score-process resampling p-value is calibrated under ties=:efron" begin
    # Tick-stamped data, three tied events per
    # tick on six actors (a tenth of the risk set tied at each tick), the
    # fitted model true. Measured over 800 data sets at n_sim = 199: the
    # resampling p-value rejects 6 % per statistic and 5.75 % globally at the
    # 5 % level. Here 200 data sets at n_sim = 99, checked in both directions as
    # in the untied calibration testset (measured: 18 rejections, D = 0.105).
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    times = Float64[div(k - 1, 3) + 1 for k in 1:240]
    pvals = Float64[]
    for seed in 1:200
        ev = simulate_events(stats, [1.0, 1.0], 6, 240; rng=Xoshiro(seed), times=times,
                             ties=:efron)
        fit = fit_revel(ev, stats, 6; ties=:efron)
        @test NetworkCore.tie_method(fit) === :efron
        push!(pvals, score_process_test(fit; n_sim=99, rng=Xoshiro(seed)).p_value[end])
    end
    @test _size_two_sided(pvals)          # neither liberal nor conservative
end

@testset "score_test has nominal size under tied event times (:breslow and :efron)" begin
    # Under heavy Efron ties a Rao test built from Efron's own score
    # and information was badly sized — its information understates the
    # variance of its score when the tied dyads carry much of the risk set's
    # weight (7.6 % on an endogenous model; 57 % in this design, where the
    # candidate is inertia and tied dyads cannot repeat within a tick).
    # `score_test` now takes each Efron tie block at its exact partial
    # likelihood (the average over the orderings), re-maximised under the null.
    # Design: 8 actors, strong sender/receiver covariates, ticks of four events,
    # candidate inertia, the fitted model true. Measured over 4,000 data sets:
    # untied 6.0 %, Breslow 5.7 %, Efron 5.7 % (120 events is a small sample).
    # Here 400 data sets per policy: a 5.7 % test gives 23 ± 4.6 rejections.
    n = 8
    xa = randn(Xoshiro(5), n); za = randn(Xoshiro(6), n)
    stats = [SendEffect(Covariate(xa)), ReceiveEffect(Covariate(za))]
    cand = [Inertia(transform=:log1p)]
    times = Float64[div(k - 1, 4) + 1 for k in 1:120]
    rejections = Dict(:breslow => 0, :efron => 0)
    lk = ReentrantLock()
    Threads.@threads for seed in 1:400
        for policy in (:breslow, :efron)
            ev = simulate_events(stats, [2.0, -2.0], n, 120; rng=Xoshiro(seed), times=times,
                                 ties=policy)
            fit = fit_revel(ev, stats, n; ties=policy)
            reject = score_test(fit, cand).p_value[1] < 0.05
            reject && lock(() -> rejections[policy] += 1, lk)
        end
    end
    @test 8 <= rejections[:breslow] <= 36
    @test 8 <= rejections[:efron] <= 36
    # the exact block likelihood: gradient and Hessian against finite differences
    function loglik(X, tied, θ)
        orderings(v) = length(v) <= 1 ? [copy(v)] :
            [vcat(v[i], q) for i in eachindex(v) for q in orderings(v[[1:i-1; i+1:end]])]
        w = exp.(X * θ); total = 0.0
        for π in orderings(tied)
            S = sum(w); l = 1.0
            for c in π; l *= w[c] / S; S -= w[c]; end
            total += l
        end
        return log(total)
    end
    X = randn(Xoshiro(1), 12, 3); tied = [2, 5, 9, 11]; θ = [0.4, -0.3, 0.8]
    score = zeros(3); info = zeros(3, 3)
    Revel._exact_block!(score, info, X, tied, θ)
    h = 1e-5; unit(k) = Float64.(1:3 .== k)
    @test score ≈ [(loglik(X, tied, θ + h * unit(k)) - loglik(X, tied, θ - h * unit(k))) / 2h
                   for k in 1:3] atol = 1e-8
    @test info ≈ [-(loglik(X, tied, θ + h * unit(k) + h * unit(l)) -
                    loglik(X, tied, θ + h * unit(k) - h * unit(l)) -
                    loglik(X, tied, θ - h * unit(k) + h * unit(l)) +
                    loglik(X, tied, θ - h * unit(k) - h * unit(l))) / 4h^2
                  for k in 1:3, l in 1:3] atol = 1e-4
    # with one tied event per block the exact form is the ordinary one: an
    # Efron fit of untied data gives the same test as the plain fit
    ev = simulate_events(stats, [2.0, -2.0], n, 120; rng=Xoshiro(1))
    @test score_test(fit_revel(ev, stats, n; ties=:efron), cand).chisq ≈
          score_test(fit_revel(ev, stats, n), cand).chisq
    # blocks too large to enumerate, and Efron fits on a subset of cases, are
    # refused with a pointer to the resampling test
    big = simulate_events(stats, [0.5, -0.5], n, 80; rng=Xoshiro(2),
                          times=Float64[div(k - 1, 8) + 1 for k in 1:80], ties=:efron)
    err = try score_test(fit_revel(big, stats, n; ties=:efron), cand); nothing catch e; e end
    @test err isa ArgumentError && occursin("score_process_test", err.msg)
    tied_ev = simulate_events(stats, [2.0, -2.0], n, 120; rng=Xoshiro(3), times=times,
                              ties=:efron)
    sub = fit_revel(tied_ev, stats, n; ties=:efron, cases=1:100)
    @test_throws ArgumentError score_test(sub, cand)
    @test score_test(fit_revel(tied_ev, stats, n; ties=:breslow, cases=1:100), cand).p_value[1] >= 0
end

@testset "gof(refit=true): refitted residuals are not conservative by construction" begin
    stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    ev = simulate_events(stats, [0.6, 0.4], 5, 80; rng=Xoshiro(1))
    fit = fit_revel(ev, stats, 5)
    g = gof(fit; n_sim=12, refit=true, n_inner=4, rng=Xoshiro(3))
    @test g isa NetworkCore.GOFResult && n_simulations(g) == 12
    @test 0 < g.p_overall <= 1
    # reproducible from `rng`, and independent of the thread count: the serial
    # loop gives the same bits as the threaded one (one seed per replicate)
    same = gof(fit; n_sim=12, refit=true, n_inner=4, rng=Xoshiro(3))
    serial = gof(fit; n_sim=12, refit=true, n_inner=4, rng=Xoshiro(3), threaded=false)
    for (a, b, c) in zip(g.statistics, same.statistics, serial.statistics)
        @test a.simulated == b.simulated == c.simulated
        @test a.observed == c.observed
    end
    @test g.p_overall == serial.p_overall
    # the observed auxiliaries are the plug-in check's; only the reference moves
    plug = gof(fit; n_sim=12, rng=Xoshiro(3))
    @test [s.observed for s in plug.statistics] == [s.observed for s in g.statistics]
    @test_throws ArgumentError gof(fit; n_sim=5, refit=true, n_inner=1)
    # it runs on the other routes: the design engine with sampled controls,
    # undirected events, and the timing model
    @test gof(fit_revel(ev, stats, 5; n_controls=6, rng=Xoshiro(1)); n_sim=6, refit=true,
              n_inner=3, rng=Xoshiro(2)) isa NetworkCore.GOFResult
    timed = simulate_events(stats, [0.6, 0.4], 5, 80; baseline=0.05, rng=Xoshiro(4))
    @test gof(fit_revel(timed, stats, 5; model=:timing); n_sim=6, refit=true, n_inner=3,
              rng=Xoshiro(2)) isa NetworkCore.GOFResult
    # --- size on true models. Measured (300 data sets, n_sim = 99, n_inner =
    # 20): per-statistic p-values reject 3.4 % at the 5 % level and p_overall
    # 3.0 %, against 1.3 % and 1.0 % for the plug-in check; at the settings
    # below (120 data sets) 5.2 % and 5.8 % against 2.3 % and 1.7 %.
    R = 120
    reject_refit = 0; reject_plug = 0
    rate_refit = Float64[]; rate_plug = Float64[]
    for r in 1:R
        evr = simulate_events(stats, [0.6, 0.4], 5, 80; rng=Xoshiro(r))
        f = fit_revel(evr, stats, 5)
        gr = gof(f; n_sim=39, refit=true, n_inner=8, rng=Xoshiro(10_000 + r))
        gp = gof(f; n_sim=39, rng=Xoshiro(10_000 + r))
        reject_refit += gr.p_overall <= 0.05
        reject_plug += gp.p_overall <= 0.05
        push!(rate_refit, mean(reduce(vcat, [s.p_values for s in gr.statistics]) .<= 0.05))
        push!(rate_plug, mean(reduce(vcat, [s.p_values for s in gp.statistics]) .<= 0.05))
    end
    @test 0.03 <= mean(rate_refit) <= 0.08          # ≈ 5 % per statistic
    @test mean(rate_plug) < mean(rate_refit) - 0.01 # the plug-in check is conservative
    @test 2 <= reject_refit <= 12                   # p_overall: 5 % of 120 is 6
    @test reject_plug <= reject_refit
end

# Mark every flag from `Threads.@threads :greedy` tasks, which interleave
# neighbouring indices across threads; return the number of lost writes.
function _mark_all_flags!(flags)
    Threads.@threads :greedy for b in eachindex(flags)
        flags[b] = true
    end
    return length(flags) - count(flags)
end

@testset "gof(refit=true): failure flags written from threads lose no write" begin
    # The refit loop records a failed replicate from `Threads.@threads` tasks.
    # In a BitVector neighbouring flags share a 64-bit word, so setting one is a
    # read-modify-write of the word: two tasks can race and a failed replicate
    # can be lost, leaving its NaN row in the reference distribution and an
    # undercounted warning. With 4 threads a BitVector lost 2 to 7 of these
    # 100,000 writes in each of 10 runs; a Vector{Bool} lost none.
    flags = Revel._failure_flags(10)
    @test flags isa Vector{Bool} && !any(flags)
    @test all(_mark_all_flags!(Revel._failure_flags(100_000)) == 0 for _ in 1:5)
    # ... and the refit loop itself takes its flags from that helper: the global
    # names its lowered code calls include `_failure_flags` and no `falses` or
    # `BitVector` (the two ways back to a packed vector)
    called = Set{Symbol}()
    function collect_globals!(x)
        x isa GlobalRef && push!(called, x.name)
        x isa Expr && foreach(collect_globals!, x.args)
        return nothing
    end
    for ci in code_lowered(Revel._gof_refit)
        foreach(collect_globals!, ci.code)
    end
    @test :_failure_flags in called
    @test isdisjoint(called, (:falses, :BitVector, :BitArray, :trues))
end

# ----------------------------------------------------------------------------
# The full-risk-set engine: Revel.fit_obpm and Revel.fit_timing
# ----------------------------------------------------------------------------

# A statistic that varies within an interval (its value is the clock), and one
# that declares itself constant between events (the sender's index − 1)
struct TimingUncertifiedStatistic <: AbstractStatistic end
struct TimingCertifiedStatistic <: AbstractStatistic end
Revel.name(::TimingUncertifiedStatistic) = "uncertified"
Revel.name(::TimingCertifiedStatistic) = "certified_sender"
Revel.compute(::TimingUncertifiedStatistic, ::InteractionHistory, s::Int, r::Int, t) = t
Revel.compute(::TimingCertifiedStatistic, ::InteractionHistory, s::Int, r::Int, t) = Float64(s - 1)
Revel.is_interval_constant(::TimingCertifiedStatistic) = true

# An exponentially fading recency of the dyad's last event: exp(−log 2 · Δ / h)
fading_recency(h) = TimeSince(:dyad; transform=Δ -> exp(-Δ * log(2) / h))

function engine_history_fixture()
    h = InteractionHistory{Float64}()
    update_history!(h, Event(1, 2, 1.0))
    update_history!(h, Event(1, 2, 2.0))
    update_history!(h, Event(2, 1, 3.0))
    update_history!(h, Event(1, 3, 4.0))
    return h
end

@testset "Engine: the interaction history" begin
    h = engine_history_fixture()
    @test length(h.events) == 4
    @test h.events[end] == Event(1, 3, 4.0)
    @test InteractionHistory() isa InteractionHistory{Float64}
    @test build_history(h.events).events == h.events
    Revel._reset_history!(h)
    @test isempty(h.events)
end

@testset "Engine: participation shifts (Gibson 2003)" begin
    @test length(pshift_types()) == 13
    @test name(PShift(:AB_BA)) == "PSAB-BA"
    @test name(PShift(:AB_BA; name="answer")) == "answer"
    @test PShift("PSAB-XY").shift == :AB_XY
    @test PShift("AB-B0").shift == :AB_B0
    @test_throws ArgumentError PShift(:AB_ZZ)

    # Hand-computed indicators. After the dyadic event 1→2 (A = 1, B = 2), each
    # candidate realizes exactly one shift (or none: repeating 1→2 is not one).
    h = InteractionHistory{Float64}()
    update_history!(h, Event(1, 2, 1.0))
    expected_dyadic = Dict(
        (2, 1) => :AB_BA,   # turn receiving: B answers A
        (2, 3) => :AB_BY,   # turn receiving: B addresses someone new
        (2, 0) => :AB_B0,   # turn receiving: B addresses the group
        (1, 3) => :AB_AY,   # turn continuing: A addresses someone new
        (1, 0) => :AB_A0,   # turn continuing: A addresses the group
        (3, 1) => :AB_XA,   # turn usurping: outsider addresses A
        (3, 2) => :AB_XB,   # turn usurping: outsider addresses B
        (3, 4) => :AB_XY,   # turn usurping: outsider addresses outsider
        (3, 0) => :AB_X0,   # turn usurping: outsider addresses the group
        (1, 2) => nothing,  # repetition of A→B: no shift
    )
    for ((i, j), shift) in expected_dyadic, ps in pshift_types()
        @test compute(PShift(ps), h, i, j, 2.0) == (ps === shift ? 1.0 : 0.0)
    end

    # After the group-directed event 1→0 (A = 1, null receiver), only the A0-*
    # shifts can fire
    h0 = InteractionHistory{Float64}()
    update_history!(h0, Event(1, 0, 1.0))
    expected_group = Dict(
        (2, 0) => :A0_X0,   # turn claiming: outsider addresses the group
        (2, 1) => :A0_XA,   # turn claiming: outsider answers A
        (2, 3) => :A0_XY,   # turn claiming: outsider addresses outsider
        (1, 3) => :A0_AY,   # turn continuing: A addresses someone
    )
    for ((i, j), shift) in expected_group, ps in pshift_types()
        @test compute(PShift(ps), h0, i, j, 2.0) == (ps === shift ? 1.0 : 0.0)
    end

    # No previous event → all shifts are 0
    @test all(compute(PShift(ps), InteractionHistory{Float64}(), 1, 2, 1.0) == 0.0
              for ps in pshift_types())

    # The REM.EventNetworkState interface agrees with the history one
    seq = EventSequence([Event(1, 2, 1.0)])
    state = REM.EventNetworkState(seq)
    REM.update!(state, seq[1])
    state.current_time = 2.0
    for ps in pshift_types(), (i, j) in keys(expected_dyadic)
        @test compute(PShift(ps), state, i, j) == compute(PShift(ps), h, i, j, 2.0)
    end
    @test compute(PShift(:AB_BA), REM.EventNetworkState{Float64}(), 1, 2) == 0.0

    # P-shifts fit in both estimators: a strongly turn-receiving stream
    rng = Random.Xoshiro(11)
    events = Event{Float64}[]
    t = 0.0
    prev = (1, 2)
    for m in 1:80
        t += rand(rng)
        if rand(rng) < 0.7
            s, r = prev[2], prev[1]
        else
            s, r = rand(rng, 1:5), rand(rng, 1:5)
            s == r && continue
        end
        push!(events, Event(s, r, t))
        prev = (s, r)
    end
    result = Revel.fit_obpm(events, [PShift(:AB_BA)], 5)
    @test result.converged
    @test result.coefficients[1] > 0   # answering is over-represented
    rem_fit = REM.fit_rem(EventSequence(events; actors=1:5), [Repetition(), PShift(:AB_BA)];
                          n_controls=10, rng=Xoshiro(3))
    @test all(isfinite, coef(rem_fit))
end

@testset "Engine: the two likelihoods recover known coefficients" begin
    # Ordinal: simulate from the multinomial model with a fading-recency effect,
    # then fit by full-risk-set maximum likelihood
    rng = Random.Xoshiro(2026)
    n = 6
    θ_true = 1.2
    stat = fading_recency(5.0)
    dyads = dyads_of(n)
    h = InteractionHistory{Float64}()
    events = Event{Float64}[]
    for m in 1:400
        t = Float64(m)
        η = [θ_true * compute(stat, h, s, r, t) for (s, r) in dyads]
        w = exp.(η .- maximum(η))
        w ./= sum(w)
        u = rand(rng)
        pick = something(findfirst(>=(u), cumsum(w)), length(dyads))
        ev = Event(dyads[pick][1], dyads[pick][2], t)
        push!(events, ev)
        update_history!(h, ev)
    end
    result = Revel.fit_obpm(events, [stat], n)
    @test result.converged
    @test isfinite(result.loglik) && result.loglik < 0
    @test result.std_errors[1] > 0
    @test result.coefficients[1] ≈ θ_true atol = 0.25
    @test coef(result) == result.coefficients
    @test stderror(result) == result.std_errors
    # a single event carries no information: the log-likelihood is log(1/30)
    @test Revel.fit_obpm(events[1:1], [stat], n).loglik ≈ log(1 / 30) atol = 1e-6

    # Sender effect: actor 3 sends at a much higher rate
    rng = Random.Xoshiro(5)
    z = [0.0, 0.0, 2.0, 0.0]
    skewed = Event{Float64}[]
    for m in 1:120
        s = rand(rng) < 0.75 ? 3 : rand(rng, [1, 2, 4])
        push!(skewed, Event(s, rand(rng, setdiff(1:4, s)), Float64(m)))
    end
    sender = Revel.fit_obpm(skewed, [SendEffect(z)], 4)
    @test sender.converged && sender.coefficients[1] > 0

    # Timing: waiting times exponential with rate λ₀·Σexp(θ'x), dyad ∝ exp(θ'x)
    rng = Random.Xoshiro(7)
    n = 5
    λ0_true = 0.4
    θ_true = 0.9
    stat = SendEffect(collect(range(-1.0, 1.0; length=n)))
    dyads = dyads_of(n)
    h = InteractionHistory{Float64}()
    events = Event{Float64}[]
    t = 0.0
    for m in 1:500
        x = [compute(stat, h, s, r, t) for (s, r) in dyads]
        w = exp.(θ_true .* x)
        t += rand(rng, Distributions.Exponential(1 / (λ0_true * sum(w))))
        probs = w ./ sum(w)
        u = rand(rng)
        pick = something(findfirst(>=(u), cumsum(probs)), length(dyads))
        ev = Event(dyads[pick][1], dyads[pick][2], t)
        push!(events, ev)
        update_history!(h, ev)
    end
    result = Revel.fit_timing(events, [stat], n)
    @test result.converged
    @test isfinite(result.loglik)
    @test result.baseline_params[1] ≈ λ0_true rtol = 0.2
    @test result.coefficients[1] ≈ θ_true atol = 0.3
    @test coef(result) == [result.log_baseline; result.coefficients]
    @test stderror(result) == [result.log_baseline_se; result.std_errors]
    # Only the exponential baseline is fitted; any other is refused, not mis-fit
    @test_throws ArgumentError Revel.fit_timing(events, [stat], n; baseline=:weibull)

    # t0 (observation onset): shifting the whole timeline by c and setting
    # t0 = c leaves every waiting time — hence the fit — unchanged
    c = 25.0
    shifted = [Event(e.sender, e.receiver, e.time + c) for e in events]
    shifted_fit = Revel.fit_timing(shifted, [stat], n; t0=c)
    @test shifted_fit.coefficients ≈ result.coefficients atol = 1e-8
    @test shifted_fit.baseline_params[1] ≈ result.baseline_params[1] atol = 1e-8
    @test shifted_fit.loglik ≈ result.loglik atol = 1e-6
    # Without t0 the first interval is overstated by c, biasing λ₀ down
    @test Revel.fit_timing(shifted, [stat], n).baseline_params[1] <
          shifted_fit.baseline_params[1]
    # The onset must precede the first event
    @test_throws ArgumentError Revel.fit_timing(shifted, [stat], n; t0=shifted[1].time + 1.0)
end

@testset "Engine: relevent::rem.dyad (golden, ordinal and timing)" begin
    # Frozen output of R relevent::rem.dyad (test/fixtures/r/relevent_rem_dyad.R).
    # Both likelihoods are EXACT — no Monte Carlo — so 1e-6 is the reference
    # optimizer's termination slack, nothing more: a failure means the
    # likelihood differs.
    g = NetworkCore.load_golden(joinpath(@__DIR__, "fixtures", "relevent_rem_dyad.toml"))
    report(key, actual) = begin
        ok = NetworkCore.check_golden(g, key, actual)
        ok || println(stderr, NetworkCore.golden_report(g, key, actual))
        ok
    end
    n = Int(g.values["n_actors"])
    times = Float64.(g.values["input_time"])
    senders = Int.(g.values["input_sender"])
    receivers = Int.(g.values["input_receiver"])
    z = Float64.(g.values["input_covariate"])
    t_end = Float64(g.values["t_end"])
    events = [Event(senders[i], receivers[i], times[i]) for i in eachindex(times)]

    # In R's coefficient order: CovSnd, CovRec (relevent's names for SendEffect
    # and ReceiveEffect), then the p-shifts
    stats = AbstractStatistic[SendEffect(z), ReceiveEffect(z),
                              PShift(:AB_BA), PShift(:AB_BY), PShift(:AB_XB), PShift(:AB_AY)]
    @test g.values["ordinal_names"] ==
          ["CovSnd.1", "CovRec.1", "PSAB-BA", "PSAB-BY", "PSAB-XB", "PSAB-AY"]

    ord = Revel.fit_obpm(events, stats, n; tol=1e-12)
    @test ord.converged
    @test report("ordinal_coefficients", ord.coefficients)
    @test report("ordinal_std_errors", ord.std_errors)
    @test report("ordinal_loglik", ord.loglik)

    # rem.dyad has no intercept, so R gets a constant CovSnd column whose
    # coefficient IS log(λ₀); its last edgelist row ends the observation window,
    # so the likelihood carries a right-censored final interval: `t_end`.
    tim = Revel.fit_timing(events, stats, n; t_end=t_end, tol=1e-12)
    @test tim.converged
    @test report("timing_log_baseline", log(tim.baseline_params[1]))
    @test report("timing_coefficients", tim.coefficients)
    @test report("timing_std_errors", tim.std_errors)
    @test report("timing_log_baseline_se", stderror(tim)[1])
    @test report("timing_loglik", tim.loglik)

    # The gap that `t_end` closes, pinned: drop the eventless tail and λ₀ is
    # biased UPWARD (the same events in a shorter window), here by about 0.047
    # in log λ₀ — 350 times the tolerance
    no_end = Revel.fit_timing(events, stats, n; tol=1e-12)
    @test log(no_end.baseline_params[1]) - Float64(g.values["timing_log_baseline"]) > 0.04
    @test !NetworkCore.check_golden(g, "timing_log_baseline", log(no_end.baseline_params[1]))

    # ... and through fit_revel, which routes both models to the same estimators
    @test report("ordinal_coefficients", coef(fit_revel(events, stats, n; tol=1e-12)))
    tfit = fit_revel(events, stats, n; model=:timing, t_end=t_end, tol=1e-12)
    @test report("timing_coefficients", coef(tfit)[2:end])

    # The SAME fixture under every cache policy: held to the golden numbers, and
    # bit-identical to each other
    for cache in (:all, :chunked, :none)
        o = Revel.fit_obpm(events, stats, n; tol=1e-12, cache=cache, chunk=8)
        t = Revel.fit_timing(events, stats, n; t_end=t_end, tol=1e-12, cache=cache, chunk=8)
        @test report("ordinal_coefficients", o.coefficients)
        @test report("timing_coefficients", t.coefficients)
        @test o.coefficients == ord.coefficients
        @test o.std_errors == ord.std_errors
        @test o.loglik === ord.loglik
        @test t.coefficients == tim.coefficients
        @test t.std_errors == tim.std_errors
        @test t.baseline_params == tim.baseline_params
        @test t.loglik === tim.loglik
    end
end

@testset "Engine: risk-set cache policies change memory, not the fit" begin
    # `cache=` bounds the O(E · n² · p) design matrices and nothing else: the
    # intervals are visited in the same order and the statistics read off the
    # same histories, so the fits are bit-identical — `==`, not `≈`.
    rng = MersenneTwister(31337)
    n = 9
    events = Event{Float64}[]
    t = 0.0
    s, r = 1, 2
    for k in 1:80
        t += 0.05 + rand(rng)
        u = rand(rng)
        if k > 1 && u < 0.3
            s, r = r, s                      # reciprocity, so PShift bites
        elseif k > 1 && u >= 0.5
            s = rand(rng, 1:n)
            r = rand(rng, filter(!=(s), 1:n))
        end
        push!(events, Event(s, r, t))
    end
    stats = AbstractStatistic[Inertia(memory=HalfLife(4.0)), PShift(:AB_BA),
                              OutdegreeSender(memory=HalfLife(3.0))]
    timing_stats = [Inertia(), PShift(:AB_BA), OutdegreeSender()]
    ref_o = Revel.fit_obpm(events, stats, n)
    ref_t = Revel.fit_timing(events, timing_stats, n; t_end=t + 5.0)
    @test ref_o.converged && ref_t.converged

    for cache in (:auto, :all, :chunked, :none), chunk in (nothing, 1, 3, 17, 500)
        cache === :chunked || chunk === nothing || continue   # chunk only bites there
        o = Revel.fit_obpm(events, stats, n; cache=cache, chunk=chunk)
        @test o.coefficients == ref_o.coefficients
        @test o.std_errors == ref_o.std_errors
        @test o.loglik === ref_o.loglik
        tm = Revel.fit_timing(events, timing_stats, n; t_end=t + 5.0, cache=cache, chunk=chunk)
        @test tm.coefficients == ref_t.coefficients
        @test tm.baseline_params == ref_t.baseline_params
        @test tm.std_errors == ref_t.std_errors
        @test tm.loglik === ref_t.loglik
    end

    # ... including under a tie correction, where `:all` SHARES one design matrix
    # across a frozen block and the streamed policies recompute it
    tied = copy(events)
    push!(tied, Event(4, 5, tied[10].time))     # a tie, distinct dyads
    ref_b = Revel.fit_obpm(tied, stats, n; ties=:breslow)
    ref_e = Revel.fit_obpm(tied, stats, n; ties=:efron)
    for cache in (:all, :chunked, :none)
        b = Revel.fit_obpm(tied, stats, n; ties=:breslow, cache=cache, chunk=2)
        e = Revel.fit_obpm(tied, stats, n; ties=:efron, cache=cache, chunk=2)
        @test b.coefficients == ref_b.coefficients
        @test e.coefficients == ref_e.coefficients   # the Efron weights too
        @test e.loglik === ref_e.loglik
    end
    @test ref_b.coefficients != ref_e.coefficients   # the corrections differ

    # The memory the policy buys, computed without allocating any of it
    plan = Revel._risk_set_plan(events, Tuple(stats), n)
    one_matrix = Revel._design_bytes(plan)
    @test one_matrix == n * (n - 1) * length(stats) * sizeof(Float64)
    @test Revel._resolve_cache(plan; cache=:all) == (:all, plan.n_int)
    @test Revel._resolve_cache(plan; cache=:none) == (:none, 1)
    @test Revel._resolve_cache(plan; cache=:chunked, chunk=5) == (:chunked, 5)
    # a chunk covering everything IS :all — same memory, no recomputation
    @test Revel._resolve_cache(plan; cache=:chunked, chunk=plan.n_int) == (:all, plan.n_int)
    # :auto caches everything under budget and chunks above it
    @test Revel._resolve_cache(plan; cache=:auto)[1] === :all
    @test Revel._resolve_cache(plan; cache=:auto, cache_bytes=4 * one_matrix) == (:chunked, 4)

    @test_throws ArgumentError Revel.fit_obpm(events, stats, n; cache=:some)
    @test_throws ArgumentError Revel.fit_obpm(events, stats, n; cache=:chunked, chunk=0)
    # fit_revel forwards the policy
    @test coef(fit_revel(events, stats, n; cache=:none)) == ref_o.coefficients
end

@testset "Engine: derivative evaluations allocate O(p²), not O(E · n² · p)" begin
    # The REAL closures the fitters hand to `newton_fit`, on preallocated
    # workspaces: what an evaluation allocates must not grow with the events or
    # the risk set. (The same bound is in benchmark/regression_tests.jl.)
    stats = AbstractStatistic[Inertia(memory=HalfLife(4.0)), PShift(:AB_BA)]
    p = length(stats)
    function derivative_allocs(n, E; cache=:all)
        rng = MersenneTwister(4)
        events = Event{Float64}[]
        t = 0.0
        for _ in 1:E
            t += 0.1 + rand(rng)
            s = rand(rng, 1:n)
            push!(events, Event(s, rand(rng, filter(!=(s), 1:n)), t))
        end
        # the ordinal likelihood has no censored tail; the timing one does
        plan_o = Revel._risk_set_plan(events, Tuple(stats), n)
        plan_t = Revel._risk_set_plan(events, Tuple(stats), n; t_end=t + 1.0)
        dO = Revel._obpm_derivatives(Revel._risk_sets(plan_o; cache=cache))
        dT = Revel._timing_derivatives(Revel._risk_sets(plan_t; cache=cache))
        θ = fill(0.05, p)
        β = fill(0.05, p + 1)
        dO(θ); dT(β)                       # warm up: a first call compiles
        return (@allocated dO(θ)), (@allocated dT(β))
    end
    # 30 dyads / 21 intervals, then 182 dyads / 201 intervals: 60x the work, and
    # only the p-sized gradient and p×p Hessian are allocated per evaluation
    o_small, t_small = derivative_allocs(6, 20)
    o_big, t_big = derivative_allocs(14, 200)
    @test o_small <= 512 && t_small <= 512
    @test o_big <= 512 && t_big <= 512
    @test o_big <= o_small + 64
    @test t_big <= t_small + 64
    # The streamed policies recompute the design matrices into bounded buffers;
    # replaying the history allocates O(E), an order of magnitude below the
    # design-matrix cache they avoid
    o_none, t_none = derivative_allocs(14, 200; cache=:none)
    plan = Revel._risk_set_plan([Event(1, 2, float(k)) for k in 1:200], Tuple(stats), 14)
    cached = Revel._design_bytes(plan) * plan.n_int
    @test o_none < cached ÷ 10
    @test t_none < cached ÷ 10
end

@testset "Engine: show and the result-metadata protocol" begin
    # On strictly ordered data nothing stands between the objective and the
    # exact likelihood: the full risk set is enumerated, nothing sampled
    events = [Event(1, 2, 1.0), Event(2, 3, 2.0), Event(3, 1, 3.0),
              Event(2, 1, 4.0), Event(1, 3, 5.0), Event(3, 2, 6.0),
              Event(1, 2, 7.0), Event(2, 3, 8.0)]
    n = 4
    ord = Revel.fit_obpm(events, [Inertia(memory=HalfLife(2.0))], n)
    md = NetworkCore.fit_metadata(ord)
    @test md.estimand == :relational_event
    @test md.objective == :likelihood
    @test md.is_exact
    @test md.se_method == :hessian
    @test md.missing_method == :none
    @test md.tie_method == :none
    @test isempty(md.approximations)

    tim = Revel.fit_timing(events, [Inertia()], n)
    mdt = NetworkCore.fit_metadata(tim)
    @test mdt.estimand == :relational_event_timing
    @test mdt.objective == :likelihood
    @test mdt.is_exact
    @test mdt.se_method == :hessian
    @test mdt.tie_method == :none
    @test isempty(mdt.approximations)

    # REM's default case-control fit of the same events: the same family of
    # objective, but a SAMPLED risk set, so not exact
    rem_fit = REM.fit_rem(EventSequence(events; actors=1:n), [REM.Repetition()];
                          n_controls=2, rng=Xoshiro(3))
    @test NetworkCore.objective(rem_fit) == :partial_likelihood
    @test !NetworkCore.is_exact(rem_fit)

    # show renders the shared coefficient table
    for (res, heading) in ((ord, "Ordinal relational event model"),
                           (tim, "Interval-timing relational event model"))
        out = sprint(show, res)
        @test occursin(heading, out)
        @test occursin(coefnames(res)[end], out)
        @test occursin("Pr(>|z|)", out)
        @test occursin("Signif. codes", out)
    end
    @test occursin("Revel.fit_obpm",
                   sprint(show, MIME"text/plain"(), fit_revel(events, [Inertia(memory=HalfLife(2.0))], n)))
end

@testset "Engine: tied event times — the two likelihoods take different policies" begin
    # ONE vocabulary (`NetworkCore.TIE_POLICIES`), ONE keyword (`ties=`), but the
    # two likelihoods claim DIFFERENT things and accept different subsets of it:
    #   fit_obpm   likelihood over the ORDER: :error :ordered :breslow :efron
    #   fit_timing exact-TIME likelihood:     :error :ordered :batch
    n = 4
    # PShift(:AB_BA) is the sharp statistic: under an arbitrary ordering the 2→1
    # event tied with 1→2 "responds" to it instantly
    stats = [PShift(:AB_BA), Inertia()]
    tied = [Event(1, 2, 1.0), Event(2, 1, 1.0),      # tie at t = 1
            Event(1, 3, 2.0),
            Event(3, 1, 3.0), Event(1, 3, 3.0),      # tie at t = 3
            Event(2, 3, 4.0)]
    untied = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0),
              Event(3, 1, 4.0), Event(1, 3, 5.0), Event(2, 3, 6.0)]

    # --- the default REFUSES, in BOTH likelihoods, and names the tie ---------
    e_o = try; Revel.fit_obpm(tied, stats, n); catch e; e; end
    e_t = try; Revel.fit_timing(tied, stats, n); catch e; e; end
    @test e_o isa ArgumentError && e_t isa ArgumentError
    m_o, m_t = sprint(showerror, e_o), sprint(showerror, e_t)
    for m in (m_o, m_t)
        @test occursin("tied timestamps", m)
        @test occursin("events 1–2", m)              # WHICH events
        @test occursin("t = 1.0", m)                 # at WHICH time
    end
    # ... for DIFFERENT reasons, and the messages say which
    @test occursin("likelihood over the ORDER", m_o)
    @test occursin(":breslow", m_o) && occursin(":efron", m_o)
    @test occursin("EXACT-TIME likelihood", m_t)
    @test occursin("probability ZERO", m_t)
    @test occursin(":batch", m_t)

    # --- each model refuses the policies that are not DEFINED for it ---------
    e_batch = try; Revel.fit_obpm(tied, stats, n; ties=:batch); catch e; e; end
    @test e_batch isa ArgumentError
    @test occursin("`:batch` is not defined", sprint(showerror, e_batch))
    @test occursin("IS the Breslow correction", sprint(showerror, e_batch))
    for bad in (:breslow, :efron)
        e_bad = try; Revel.fit_timing(tied, stats, n; ties=bad); catch e; e; end
        @test e_bad isa ArgumentError
        m = sprint(showerror, e_bad)
        @test occursin("`:$bad` is not defined", m)
        @test occursin("PARTIAL likelihood", m)     # why
        @test occursin("fit_obpm", m)                # where it DOES apply
    end
    # ... and both refuse a symbol outside the shared vocabulary
    for f in (Revel.fit_obpm, Revel.fit_timing)
        e_junk = try; f(tied, stats, n; ties=:jitter); catch e; e; end
        @test e_junk isa ArgumentError
        @test occursin("unknown tie policy", sprint(showerror, e_junk))
        @test occursin("NetworkCore.TIE_POLICIES", sprint(showerror, e_junk))
    end

    # --- on TIE-FREE data every policy is a no-op ----------------------------
    # (each of these fits warns of separation; the verdict is asserted below)
    u_ord = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        [Revel.fit_obpm(untied, stats, n; ties=t) for t in (:error, :ordered, :breslow, :efron)]
    end
    for f in u_ord[2:end]
        @test coef(f) == coef(u_ord[1])
        @test stderror(f) == stderror(u_ord[1])
        @test f.loglik == u_ord[1].loglik
    end
    u_tim = [Revel.fit_timing(untied, stats, n; ties=t, t_end=7.0)
             for t in (:error, :ordered, :batch)]
    for f in u_tim[2:end]
        @test coef(f) == coef(u_tim[1])
        @test f.baseline_params == u_tim[1].baseline_params
        @test f.loglik == u_tim[1].loglik
    end
    @test all(NetworkCore.tie_method(f) === :none for f in vcat(u_ord, u_tim))
    @test all(NetworkCore.is_exact(f) for f in vcat(u_ord, u_tim))
    @test all(isempty(NetworkCore.approximations(f)) for f in u_tim)
    # These six events have no finite ordinal MLE: the shared separation verdict
    # flags both statistics, under every policy, and that is the only caveat
    for f in u_ord
        @test !f.converged
        @test f.separated == ["PSAB-BA", "inertia"]
        @test NetworkCore.approximations(f) == [NetworkCore.separation_caveat(f.separated)]
    end

    # --- ordinal: the three policies genuinely differ on tied data -----------
    o = Revel.fit_obpm(tied, stats, n; ties=:ordered)
    b = Revel.fit_obpm(tied, stats, n; ties=:breslow)
    ef = Revel.fit_obpm(tied, stats, n; ties=:efron)
    @test coef(o) != coef(b) && coef(b) != coef(ef)
    @test NetworkCore.tie_method(o) === :ordered
    @test NetworkCore.tie_method(b) === :breslow
    @test NetworkCore.tie_method(ef) === :efron
    @test !NetworkCore.is_exact(o) && !NetworkCore.is_exact(b) && !NetworkCore.is_exact(ef)
    @test any(occursin("BRESLOW correction", a) for a in NetworkCore.approximations(b))
    @test any(occursin("EFRON correction", a) for a in NetworkCore.approximations(ef))
    @test any(occursin("ordered arbitrarily", a) for a in NetworkCore.approximations(o))
    @test occursin("Tied event times: breslow", sprint(show, b))
    @test !occursin("Tied event times", sprint(show, u_ord[1]))

    # --- what the corrections DO: freeze the tie block -----------------------
    _, ci_o, X_o, _, W_o = Revel._risk_set_stats(tied, Tuple(stats), n; ties=:ordered)
    _, ci_b, X_b, _, W_b = Revel._risk_set_stats(tied, Tuple(stats), n; ties=:breslow)
    @test X_o[2][ci_o[2], 1] == 1.0     # the invented instantaneous response
    @test X_b[2][ci_b[2], 1] == 0.0     # a tie cannot see itself
    @test W_o === nothing && W_b === nothing
    # the block is absorbed as a WHOLE (frozen ≠ dropped)
    @test X_b[3][ci_b[3], 2] == 0.0     # 1→3 has no prior 1→3 yet
    @test X_o[6][ci_o[6], 2] == 0.0     # 2→3 has no prior 2→3
    d12 = findfirst(==((1, 2)), dyads_of(n))
    @test X_b[3][d12, 2] > 0.0

    # --- Efron's weights -------------------------------------------------------
    _, _, _, _, W_e = Revel._risk_set_stats(tied, Tuple(stats), n; ties=:efron)
    @test W_e !== nothing
    @test all(==(1.0), W_e[1])                       # j = 1: weight 1 − 0/2
    @test count(==(0.5), W_e[2]) == 2                # j = 2: the two tied cases
    @test count(==(1.0), W_e[2]) == n * (n - 1) - 2  # ... and nobody else
    @test all(==(1.0), W_e[3])                       # untied interval, untouched
    # Efron needs the tied cases to be DISTINCT dyads
    dup = [Event(1, 2, 1.0), Event(1, 2, 1.0), Event(2, 3, 2.0)]
    e_dup = try; Revel.fit_obpm(dup, stats, n; ties=:efron); catch e; e; end
    @test e_dup isa ArgumentError
    @test occursin("distinct dyads", sprint(showerror, e_dup))
    # Breslow accepts duplicated dyads; this tiny two-effect example has no
    # finite MLE, so tie admissibility and estimability are kept apart
    f_breslow = @test_logs (:warn, r"does not exist \(separation\)") match_mode=:any Revel.fit_obpm(dup, stats, n; ties=:breslow)
    @test !f_breslow.converged && !isempty(f_breslow.separated)
    @test Revel.fit_obpm(dup, [SendEffect([0., 1., 2., 3.])], n; ties=:breslow).converged

    # --- timing: :ordered vs :batch -------------------------------------------
    # The tied intervals have Δt = 0, so the policies differ only through the
    # tied events' OWN statistics — exactly where the arbitrary order does damage
    to = Revel.fit_timing(tied, stats, n; ties=:ordered, t_end=5.0)
    tb = Revel.fit_timing(tied, stats, n; ties=:batch, t_end=5.0)
    @test coef(to) != coef(tb)
    @test NetworkCore.tie_method(to) === :ordered
    @test NetworkCore.tie_method(tb) === :batch
    @test !NetworkCore.is_exact(to) && !NetworkCore.is_exact(tb)
    @test any(occursin("ZERO-LENGTH waiting interval", a) for a in NetworkCore.approximations(to))
    @test any(occursin("simultaneous BATCH", a) for a in NetworkCore.approximations(tb))
    @test any(occursin("COARSENED observation process", a) for a in NetworkCore.approximations(tb))
    @test occursin("probability zero", sprint(show, tb))

    # --- fit_revel forwards the policy to whichever likelihood it picks ------
    @test NetworkCore.tie_method(fit_revel(tied, stats, n; ties=:efron)) === :efron
    @test NetworkCore.tie_method(fit_revel(tied, stats, n; model=:timing, ties=:batch)) === :batch
    @test_throws ArgumentError fit_revel(tied, stats, n)                    # ordinal
    @test_throws ArgumentError fit_revel(tied, stats, n; model=:timing)
    # a policy sent to the likelihood it is not defined for is refused
    @test_throws ArgumentError fit_revel(tied, stats, n; model=:timing, ties=:efron)
    @test_throws ArgumentError fit_revel(tied, stats, n; ties=:batch)

    # --- ONE vocabulary, shared with REM.jl -----------------------------------
    seq = EventSequence(tied; actors=1:n)
    rstats = [REM.Repetition(), REM.Reciprocity()]
    @test_throws ArgumentError REM.fit_rem(seq, rstats; n_controls=11)
    rem_ef = REM.fit_rem(seq, rstats; n_controls=11, ties=:efron)
    @test NetworkCore.tie_method(rem_ef) === :efron
    @test !NetworkCore.is_exact(rem_ef)
end

@testset "Engine: the shared optimizer and a coherent StatsAPI surface" begin
    @test Revel.newton_fit === NetworkCore.newton_fit
    @test !isdefined(Revel, :_newton)
    @test Base.ispublic(Revel, :fit_obpm) && !Base.isexported(Revel, :fit_obpm)
    @test Base.ispublic(Revel, :fit_timing) && Base.ispublic(Revel, :OrdinalBPMResult)
    @test Base.ispublic(Revel, :TimingModelResult) && Base.ispublic(Revel, :is_interval_constant)
    g = NetworkCore.load_golden(joinpath(@__DIR__, "fixtures", "relevent_rem_dyad.toml"))
    v = g.values
    events = [Event(Int(s), Int(r), Float64(t)) for (s, r, t) in
              zip(v["input_sender"], v["input_receiver"], v["input_time"])]
    z = Float64.(v["input_covariate"])
    stats = [SendEffect(z), ReceiveEffect(z)]
    n = Int(v["n_actors"])
    t_end = Float64(v["t_end"])
    for timing in (false, true)
        fit = timing ? Revel.fit_timing(events, stats, n; t_end=t_end) :
                       Revel.fit_obpm(events, stats, n)
        @test fit.converged
        @test fit.iterations > 0
        @test length(coef(fit)) == dof(fit) == length(coefnames(fit))
        @test nobs(fit) == length(events)
        @test vcov(fit) ≈ vcov(fit)'
        @test stderror(fit) ≈ sqrt.(diag(vcov(fit)))
        @test loglikelihood(fit) == fit.loglik
        @test aic(fit) == -2fit.loglik + 2dof(fit)
        @test bic(fit) == -2fit.loglik + log(nobs(fit)) * dof(fit)
        @test Revel.aicc(fit) > aic(fit)
        @test size(confint(fit)) == (dof(fit), 2)
        @test all(confint(fit)[:, 1] .<= coef(fit) .<= confint(fit)[:, 2])
        @test_throws ArgumentError confint(fit; level=1)
        @test_throws ArgumentError confint(fit; level=NaN)
        @test coeftable(fit).names == coefnames(fit)
        # coefnames is the StatsAPI binding, checked with the other verbs, and a
        # caller's mutation of the returned vector cannot reach the fit
        @test all(values(NetworkCore.check_statsapi(fit;
            required=(NetworkCore.STATSAPI_VERBS..., :coefnames), strict=true)))
        labels = coefnames(fit); labels[1] = "changed"
        @test coefnames(fit)[1] != "changed"
        @test coefnames(fit) == (timing ? ["log_baseline", "send.x", "receive.x"] :
                                          ["send.x", "receive.x"])
        @test coeftable(fit).p_values == NetworkCore.z_pvalues(coef(fit), stderror(fit)).p
        @test occursin(coefnames(fit)[1], sprint(show, fit))
        # the reported covariance is the inverse information at the optimum
        plan = Revel._risk_set_plan(events, stats, n; t_end=timing ? t_end : nothing)
        rs = Revel._risk_sets(plan)
        deriv = timing ? Revel._timing_derivatives(rs) : Revel._obpm_derivatives(rs)
        ll, grad, hess = deriv(coef(fit))
        @test ll == fit.loglik
        @test vcov(fit) ≈ inv(-hess)
        # a statistic on a scale of 1e8 changes the coefficient, not the fit
        scaled = timing ? Revel.fit_timing(events, [SendEffect(1e8 .* z), ReceiveEffect(z)], n;
                                           t_end=t_end) :
                          Revel.fit_obpm(events, [SendEffect(1e8 .* z), ReceiveEffect(z)], n)
        @test scaled.converged
        @test scaled.loglik ≈ fit.loglik atol = 1e-8
        @test scaled.coefficients .* [1e8, 1] ≈ fit.coefficients rtol = 1e-6
        @test_logs (:warn, r"did not converge") begin
            stalled = timing ? Revel.fit_timing(events, stats, n; maxiter=1) :
                               Revel.fit_obpm(events, stats, n; maxiter=1)
            @test !stalled.converged
        end
    end
    # a constant statistic has no information in the ordinal likelihood: NaN
    # uncertainty, not a pseudoinverse
    @test_logs (:warn, r"not negative definite") (:warn, r"did not converge") begin
        singular = Revel.fit_obpm(events, [SendEffect(ones(n))], n)
        @test !singular.converged
        @test all(isnan, stderror(singular))
        @test all(isnan, vcov(singular))
    end
    # argument validation
    ev = [Event(1, 2, 1.0), Event(2, 3, 2.0)]
    for fitfun in (Revel.fit_obpm, Revel.fit_timing)
        @test_throws ArgumentError fitfun(ev, [SendEffect([1., 2., 3.])], 3; maxiter=0)
        @test_throws ArgumentError fitfun(ev, [SendEffect([1., 2., 3.])], 3; tol=NaN)
        @test_throws ArgumentError fitfun(Event{Float64}[], [SendEffect([1., 2., 3.])], 3)
        @test_throws ArgumentError fitfun(ev, [SendEffect([1., 2., 3.])], 1)
        @test_throws ArgumentError fitfun(ev, AbstractStatistic[], 3)
    end
    @test_throws ArgumentError Revel.fit_timing(ev, [SendEffect([1., 2., 3.])], 3; t0=-Inf)
    @test_throws ArgumentError Revel.fit_timing(ev, [SendEffect([1., 2., 3.])], 3; t_end=Inf)
    @test_throws ArgumentError Revel.fit_timing(ev, [SendEffect([1., 2., 3.])], 3; t_end=1.5)
    @test_throws ArgumentError Revel.fit_obpm([Event(1, 4, 1.0)], [SendEffect([1., 2., 3.])], 3)
end

@testset "Engine: the timing likelihood's predictor and analytic derivatives" begin
    events = [Event(1, 2, 1.0), Event(2, 3, 2.0), Event(3, 1, 3.0)]
    plan = Revel._risk_set_plan(events, [SendEffect(ones(3))], 3)
    f = Revel._timing_derivatives(Revel._risk_sets(plan))
    @test f([0.0, 0.0])[1] == -18.0
    @test f([-1000.0, 1000.0]) == f([0.0, 0.0])
    @test f([1000.0, -1000.0]) == f([0.0, 0.0])
    # Zero exposure with an overflowing hazard is exactly zero, not Inf*0
    plan0 = Revel._risk_set_plan([Event(1, 2, 0.0)], [SendEffect(ones(3))], 3)
    f0 = Revel._timing_derivatives(Revel._risk_sets(plan0))
    @test f0([1000.0, 1000.0]) == (2000.0, [1.0, 1.0], zeros(2, 2))
    # Finite exposure even where the hazard alone would overflow
    tiny = [Event(1, 2, 1e-300)]
    ft = Revel._timing_derivatives(Revel._risk_sets(
         Revel._risk_set_plan(tiny, [SendEffect(ones(3))], 3)))
    @test all(isfinite, ft([710.0, 0.0])[2])
    rs = Revel._risk_sets(Revel._risk_set_plan(events,
         [SendEffect([-0.5, 0.1, 0.7]), ReceiveEffect([0.4, -0.2, 0.8])], 3; t_end=4.5))
    f = Revel._timing_derivatives(rs)
    b = [-1.2, 0.3, -0.4]
    ll, grad, hess = f(b)
    eps = 1e-5
    for k in eachindex(b)
        d = zeros(3); d[k] = eps
        @test grad[k] ≈ (f(b + d)[1] - f(b - d)[1]) / (2eps) atol = 1e-8
        @test hess[:, k] ≈ (f(b + d)[2] - f(b - d)[2]) / (2eps) atol = 1e-8
    end
end

@testset "Engine: exact timing requires interval-constant statistics" begin
    events = [Event(1, 2, 1.0), Event(2, 1, 3.0)]
    history = engine_history_fixture()
    ic = Revel.is_interval_constant
    for (decayed, cumulative) in (
            (Inertia(memory=HalfLife(1.0)), Inertia()),
            (Reciprocation(memory=HalfLife(1.0)), Reciprocation()),
            (DyadActivity(memory=HalfLife(1.0)), DyadActivity(memory=HalfLife(Inf))),
            (OutdegreeSender(memory=HalfLife(1.0)), OutdegreeSender()),
            (IndegreeReceiver(memory=HalfLife(1.0)), IndegreeReceiver()),
            (TimeSince(:dyad), TimeSince(:dyad; clock=:order)))
        @test !ic(decayed)
        @test ic(cumulative)
        @test compute(decayed, history, 1, 2, 5.0) != compute(decayed, history, 1, 2, 6.0)
        @test compute(cumulative, history, 1, 2, 5.0) == compute(cumulative, history, 1, 2, 6.0)
        for cache in (:all, :chunked, :none)
            err = try
                Revel.fit_timing(events, [decayed], 2; cache=cache)
            catch exception
                exception
            end
            @test err isa ArgumentError
            @test occursin("exposure integration", sprint(showerror, err))
        end
        @test_throws ArgumentError fit_revel(events, [decayed], 2; model=:timing)
    end
    for stat in (PShift(:AB_BA), PShiftABAB(), RecencyRank(:send), OTP(),
                 SendEffect([1., 2., 3.]), ReceiveEffect([1., 2., 3.]),
                 SumEffect([1., 2., 3.]), TieEffect(zeros(3, 3)),
                 OutdegreeSender(scaling=:prop, empty=0.5))
        @test ic(stat)
    end
    @test !ic(TimingUncertifiedStatistic())
    @test_throws ArgumentError Revel.fit_timing(events, [TimingUncertifiedStatistic()], 2)
    balanced = [Event(1, 2, 1.), Event(2, 3, 2.), Event(3, 1, 3.),
                Event(1, 3, 4.), Event(2, 1, 5.), Event(3, 2, 6.)]
    custom = Revel.fit_timing(balanced, [TimingCertifiedStatistic()], 3; t_end=7.)
    built_in = Revel.fit_timing(balanced, [SendEffect([0., 1., 2.])], 3; t_end=7.)
    @test custom.converged && built_in.converged
    @test coef(custom) == coef(built_in)
    @test vcov(custom) == vcov(built_in)
    @test NetworkCore.is_exact(custom)

    # With no decay the true hazard is constant on [1, 3]: the integrated
    # exposure is exactly 2·(e + 1), after the initial exposure 2
    rs = Revel._risk_sets(Revel._risk_set_plan(events, [Inertia()], 2))
    @test Revel._timing_derivatives(rs)([0., 1.])[1] ≈ -2 - 2 * (exp(1) + 1)
    # Finite decay stays available to the ordinal likelihood (repeated and new
    # dyads both occur, so a repetition effect is identified)
    overlapping_history = [balanced; Event(3, 2, 7.); Event(1, 2, 8.)]
    @test Revel.fit_obpm(overlapping_history, [Inertia(memory=HalfLife(1.0))], 3).converged
end

@testset "Engine: separation — the shared verdict, a warning and withheld inference" begin
    # Quasi separation: two dyads share the maximal sender statistic. Complete
    # separation: one observed dyad has the unique maximal dyad statistic.
    quasi = [Event(3, isodd(k) ? 1 : 2, Float64(k)) for k in 1:12]
    complete = [Event(3, 1, Float64(k)) for k in 1:12]
    dyad_x = zeros(3, 3); dyad_x[3, 1] = 1.0
    sep_warning = (:warn, r"does not exist \(separation\)")
    for (events, stat) in ((quasi, SendEffect([0., 1., 2.])), (complete, TieEffect(dyad_x)))
        for cache in (:all, :chunked, :none), timing in (false, true)
            fit = @test_logs sep_warning match_mode=:any begin
                timing ? Revel.fit_timing(events, [stat], 3; t_end=13.0, cache=cache, chunk=2) :
                         Revel.fit_obpm(events, [stat], 3; cache=cache, chunk=2)
            end
            # the policy: converged == false, the terms flagged, inference withheld
            @test !fit.converged
            @test fit.separation.separated && fit.separation.certified
            @test fit.separation.family === (timing ? :poisson : :clogit)
            @test name(stat) in fit.separated
            @test fit.separated == coefnames(fit)[fit.separation.terms]
            @test all(isfinite, coef(fit))          # kept for diagnosis
            table = coeftable(fit)
            @test all(isnan, table.z_values) && all(isnan, table.p_values)
            @test all(isnan, confint(fit))
            @test any(occursin("separation", a) for a in NetworkCore.approximations(fit))
            @test occursin("Warning: separation", sprint(show, fit))
            # every interval of these sequences is predicted perfectly
            @test fit.separation.units == 1:(timing ? 13 : 12)
        end
    end
    # The baseline runs away with the sender effect: log λ₀ → −∞ as θ → ∞
    tfit = @test_logs sep_warning match_mode=:any Revel.fit_timing(quasi, [SendEffect([0., 1., 2.])], 3; t_end=13.0)
    @test tfit.separated == ["log_baseline", "send.x"]
    # The verdict is a property of the data, not of where Newton stopped
    early = @test_logs sep_warning match_mode=:any Revel.fit_obpm(quasi, [SendEffect([0., 1., 2.])], 3; maxiter=1)
    full = @test_logs sep_warning match_mode=:any Revel.fit_obpm(quasi, [SendEffect([0., 1., 2.])], 3)
    @test early.separation.terms == full.separation.terms

    verdict(events, stats; timing=false, kwargs...) = Revel._separation_verdict(
        Revel._risk_sets(Revel._risk_set_plan(events, stats, 3; kwargs...)); timing)

    # Genuine overlap as small as one representable increment is never PROVED
    # separated. In the ordinal likelihood it is not separated at all; in the
    # timing likelihood the overlap survives only in a combination of rows, so
    # the shared verdict reports it as separated numerically but not certified.
    for scale in (1e-8, 1.0, 1e8), timing in (false, true)
        overlap = [0., nextfloat(1.), 1.] .* scale
        v = verdict(quasi, [SendEffect(overlap)]; timing, t_end=timing ? 13.0 : nothing)
        @test timing ? !v.certified : !v.separated
        wide = [0., 1.5, 1.] .* scale
        @test !verdict(quasi, [SendEffect(wide)]; timing, t_end=timing ? 13.0 : nothing).separated
    end
    # An exactly constant statistic gives a flat direction, not separation
    for timing in (false, true)
        @test !verdict(quasi, [SendEffect(ones(3))]; timing).separated
    end

    # Tied cases: Efron's denominator weights stay positive, so every dyad stays
    # in the choice set; a zero-length timing interval has an event row but no
    # exposure
    tied = [Event(3, 1, 1.0), Event(3, 2, 1.0), Event(3, 1, 2.0), Event(3, 2, 2.0)]
    for policy in (:ordered, :breslow, :efron)
        fit = @test_logs sep_warning match_mode=:any Revel.fit_obpm(tied, [SendEffect([0., 1., 2.])], 3; ties=policy)
        @test fit.separated == ["send.x"]
    end
    for policy in (:ordered, :batch)
        fit = @test_logs sep_warning match_mode=:any Revel.fit_timing(tied, [SendEffect([0., 1., 2.])], 3; ties=policy, t_end=3.)
        @test !fit.converged && "send.x" in fit.separated
    end
    # A single event at the onset has no exposure at all without a censored tail:
    # ℓ = log λ₀ + θ x grows without bound in log λ₀. With the tail the baseline
    # is identified, and the sender effect still separates.
    instant = [Event(3, 1, 0.0)]
    v_instant = verdict(instant, [SendEffect([0., 1., 2.])]; timing=true)
    @test v_instant.separated && v_instant.certified && length(v_instant.terms) == 1
    @test verdict(instant, [SendEffect([0., 1., 2.])]; timing=true, t_end=1.0).separated
    # A tail predictor above the case's rules the direction out, even though
    # every event interval alone admits it. Modify only the tail design.
    rs = Revel._risk_sets(Revel._risk_set_plan(quasi, [SendEffect([0., 1., 2.])], 3; t_end=13.0))
    @test Revel._separation_verdict(rs; timing=true).separated
    rs.X[end][1, 1] = 3.0
    @test !Revel._separation_verdict(rs; timing=true).separated

    # Against the shared family entry points on the flattened design: the
    # ordinal verdict is `clogit_separation` with one stratum per event, and the
    # timing verdict is `poisson_separation` on (1, x) rows of the intervals with
    # exposure. Random small integer designs, separated and not; deduplicating
    # the margin rows, or generating them under a tiny budget, changes nothing.
    rng = Xoshiro(20261006)
    agree_o = agree_t = n_sep = 0
    for trial in 1:60
        n = 3 + trial % 2
        E = rand(rng, 3:7)
        pairs = [(s, r) for s in 1:n for r in 1:n if s != r]
        evs = [Event(rand(rng, pairs)..., Float64(k)) for k in 1:E]
        stats = [SendEffect(Float64.(rand(rng, 0:2, n))), TieEffect(Float64.(rand(rng, 0:1, n, n)))]
        plan = Revel._risk_set_plan(evs, stats, n; t_end=E + 1.0)
        rs = Revel._risk_sets(plan)
        X, chosen, strata = Matrix{Float64}(undef, 0, 2), Bool[], Int[]
        Z, y = Matrix{Float64}(undef, 0, 3), Float64[]
        Revel._each_interval(rs) do m, Xm, _
            ci = plan.case_idx[m]
            if ci > 0
                X = vcat(X, Xm); append!(strata, fill(m, size(Xm, 1)))
                append!(chosen, (1:size(Xm, 1)) .== ci)
            end
            Z = vcat(Z, hcat(ones(size(Xm, 1)), Xm)); append!(y, (1:size(Xm, 1)) .== ci)
        end
        rso = Revel._risk_sets(Revel._risk_set_plan(evs, stats, n))
        vo = Revel._separation_verdict(rso)
        vt = Revel._separation_verdict(rs; timing=true)
        ro = NetworkCore.clogit_separation(X, chosen, strata)
        rt = NetworkCore.poisson_separation(Z, y)
        # a budget of a few rows forces the row-generation route
        bo = Revel._separation_verdict(rso; budget=1)
        bt = Revel._separation_verdict(rs; timing=true, budget=1)
        agree_o += vo.separated == ro.separated == bo.separated
        agree_t += vt.separated == rt.separated == bt.separated
        bo.separated && @test bo.certified
        n_sep += vo.separated
        vo.separated && @test vo.certified
    end
    @test agree_o == 60 && agree_t == 60
    @test 5 <= n_sep <= 55                  # both outcomes are exercised

    # Seeded overlapping data remain identified and cache-identical. Each sender
    # appears equally often, so the static-covariate MLE is θ = 0.
    rng = Xoshiro(20260914)
    dyads = shuffle(rng, dyads_of(3))
    balanced = [Event(s, r, Float64(k)) for (k, (s, r)) in enumerate(dyads)]
    for timing in (false, true)
        fits = [timing ? Revel.fit_timing(balanced, [SendEffect([0., 1., 2.])], 3; t_end=7., cache=cache, chunk=2) :
                         Revel.fit_obpm(balanced, [SendEffect([0., 1., 2.])], 3; cache=cache, chunk=2)
                for cache in (:all, :chunked, :none)]
        @test all(f.converged for f in fits)
        @test all(coef(f) == coef(fits[1]) for f in fits)
        @test all(vcov(f) == vcov(fits[1]) for f in fits)
        @test fits[1].coefficients[1] ≈ 0.0 atol = 1e-8
        @test fits[1].std_errors[1] ≈ 0.5 atol = 1e-8
    end
end

@testset "Engine: the docstring examples run without warnings" begin
    # A warning in a demonstration (a separation, a non-convergence) teaches the
    # wrong lesson; the claims in these blocks are checked by the docstring
    # testset, this one checks that they are silent
    for nm in (:fit_obpm, :fit_timing, :OrdinalBPMResult, :TimingModelResult,
               :is_interval_constant)
        b = Base.Docs.Binding(Revel, nm)
        texts = [join(string.(ds.text)) for ds in values(Base.Docs.meta(Revel)[b].docs)]
        blocks = [m.captures[1] for t in texts for m in eachmatch(r"```julia\n(.*?)```"s, t)]
        @test !isempty(blocks)
        for block in blocks
            @test_logs min_level=Base.CoreLogging.Warn begin
                redirect_stdout(devnull) do
                    Base.include_string(Module(), block)
                end
            end
        end
    end
end

@testset "Aqua: package hygiene" begin
    Aqua.test_all(Revel)
end
