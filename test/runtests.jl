using Test, Revel, REM, Relevent, Networks, Random, LinearAlgebra, Statistics, DataFrames

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

const MEMORIES = (FullMemory(), HalfLife(3.0), HalfLife(3.0; normalized=true),
                  Window(4.0), Interval(1.0, 6.0), PowerLaw(0.7; offset=0.5),
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
    @test kernel_weight(Interval(1.0, 7.0), 1.0) == 0.0
    @test kernel_weight(Interval(1.0, 7.0), 7.0) == 1.0
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
    @test_throws ArgumentError Interval(2.0, 2.0)
    @test_throws ArgumentError Interval(-1.0, 2.0)
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
    @test compute(Inertia(memory=Interval(1.0, 3.0)), h, 1, 2, 4.0) == 1.0   # only t = 1
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
    # Replaying a history in place (what Relevent's streamed risk sets do)
    # rebuilds the layer rather than double counting
    empty!(h.events); empty!(h.sender_history); empty!(h.receiver_history)
    empty!(h.pair_history); empty!(h.event_counts)
    for e in events[1:10]
        update_history!(h, e)
    end
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
    # 3 is 1's friend and 2's enemy; 4 is an enemy of both (weights 1 and 2)
    @test compute(BalanceEffect(:friend_of_enemy), h, 1, 2, 6.0) == 1.0
    @test compute(BalanceEffect(:enemy_of_friend), h, 2, 1, 6.0) == 1.0
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
# Agreement with Relevent.jl and REM.jl
# ----------------------------------------------------------------------------

@testset "Parity with Relevent's relevent catalogue, both interfaces" begin
    rng = Xoshiro(11)
    n = 7
    events = random_events(rng, n, 120)
    e0 = 1 / (n - 1)      # relevent's value before any event
    pairs = [
        (OTP(), OTPSnd(n)), (ITP(), ITPSnd(n)), (ISP(), ISPSnd(n)),
        (RecencyRank(:receive), RRecSnd(n)), (RecencyRank(:send), RSndSnd(n)),
        (IndegreeSender(scaling=:prop, empty=e0), NIDSnd(n)),
        (IndegreeReceiver(scaling=:prop, empty=e0), NIDRec(n)),
        (OutdegreeSender(scaling=:prop, empty=e0), NODSnd(n)),
        (OutdegreeReceiver(scaling=:prop, empty=e0), NODRec(n)),
        (TotaldegreeSender(scaling=:prop, empty=e0), NTDegSnd(n)),
        (TotaldegreeReceiver(scaling=:prop, empty=e0), NTDegRec(n)),
        (Inertia(memory=HalfLife(4.0)), PriorInteraction(4.0)),
        (Reciprocation(memory=HalfLife(4.0)), PriorInteraction(4.0; direction=:incoming)),
        (DyadActivity(memory=HalfLife(4.0)), PriorInteraction(4.0; direction=:both)),
        (OutdegreeSender(memory=HalfLife(4.0)), SendingCapacity(4.0)),
        (IndegreeReceiver(memory=HalfLife(4.0)), ReceivingCapacity(4.0)),
        (TimeSince(:dyad; transform=Δ -> exp(-Δ * log(2) / 4.0)), LocalInertia(4.0)),
        (SendEffect(collect(1.0:n)), CovSnd(collect(1.0:n))),
        (ReceiveEffect(collect(1.0:n)), CovRec(collect(1.0:n))),
        (SumEffect(collect(1.0:n)), CovInt(collect(1.0:n))),
        (TieEffect([i - 2j for i in 1:n, j in 1:n]), CovEvent([i - 2j for i in 1:n, j in 1:n])),
    ]
    history = InteractionHistory()
    state = REM.EventNetworkState{Float64}()
    worst = zeros(length(pairs))
    interface_gap = 0.0
    for e in events
        state.current_time = e.time
        for (s, r) in dyads_of(n), (k, (mine, theirs)) in enumerate(pairs)
            v = compute(mine, history, s, r, e.time)
            worst[k] = max(worst[k], abs(v - compute(theirs, history, s, r, e.time)))
            interface_gap = max(interface_gap, abs(v - compute(mine, state, s, r)))
        end
        update_history!(history, e)
        REM.update!(state, e)
    end
    # The cumulative statistics are counts and ranks: identical, not close
    @test all(==(0.0), worst[1:11])
    @test all(<(1e-12), worst[12:end])
    # Relevent's history interface and REM's state interface are one computation
    @test interface_gap == 0.0

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
    # any statistic of the ecosystem can be a part: Relevent's p-shift here
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
    ic = Relevent.is_interval_constant
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
    @test sizes(cases=e -> e.sender == 1) == [6, 6]
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
    @test exact.engine === :relevent && design.engine === :design
    @test exact.fit isa Relevent.OrdinalBPMResult && design.fit isa REM.REMResult
    # the same likelihood, maximised by the same shared Newton optimizer
    @test coef(exact) ≈ coef(design) atol = 1e-8
    @test stderror(exact) ≈ stderror(design) atol = 1e-8
    @test loglikelihood(exact) ≈ loglikelihood(design) atol = 1e-8
    @test vcov(exact) ≈ vcov(design) atol = 1e-8
    # ... and as REM.fit_rem on the full risk set, through REM's own interface
    seq = EventSequence(events; actors=ActorSet(1:n))
    rem_fit = fit_rem(seq, truth; n_controls=n * (n - 1) - 1)
    @test coef(rem_fit) ≈ coef(exact) atol = 1e-8
    # ... and as Relevent.fit_obpm called directly
    @test coef(fit_obpm(events, truth, n)) == coef(exact)

    # the StatsAPI surface and the result-metadata protocol
    for fit in (exact, design)
        @test all(values(Networks.check_statsapi(fit; strict=true)))
        @test coefnames(fit) == ["log1p(inertia)", "log1p(reciprocity)", "log1p(otp)"]
        @test nobs(fit) == 300 && dof(fit) == 3
        @test aic(fit) ≈ -2loglikelihood(fit) + 6
        @test size(confint(fit)) == (3, 2)
        @test confint(fit; level=0.5)[1, 2] < confint(fit)[1, 2]
        @test coeftable(fit) isa Networks.CoefficientTable
        meta = Networks.fit_metadata(fit)
        @test meta.estimand === :relational_event
        @test Networks.is_exact(fit)
        @test Networks.tie_method(fit) === :none
        @test isempty(Networks.approximations(fit))
        @test occursin("Revel relational event model", sprint(show, fit))
    end
    @test Networks.objective(exact) === :likelihood
    @test Networks.se_method(design) === :hessian
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
    @test tfit.fit isa Relevent.TimingModelResult
    @test coefnames(tfit) == ["log_baseline", "log1p(inertia)", "x"]
    @test all(abs.(coef(tfit) .- [log(0.02), 0.8, 0.7]) .< 4 .* stderror(tfit))
    @test all(values(Networks.check_statsapi(tfit; strict=true)))
    @test Networks.estimand(tfit) === :relational_event_timing
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
    sender_only = fit_receiver_choice(events, [Inertia(transform=:log1p),
                                               SendEffect(collect(1.0:n))], n)
    @test sender_only.fit.singular
    @test isnan(stderror(sender_only)[2])
    @test !Networks.is_exact(sender_only)

    # a subset of cases: the other events still build the history
    late = fit_revel(events, stats, n; cases=201:400)
    @test nobs(late) == 200 && count(late.cases) == 200
    @test late.engine === :design
    @test occursin("200 modelled as cases", sprint(show, late))
    @test_throws ArgumentError fit_revel(events, stats, n; cases=e -> false)

    # sampled controls are reproducible from rng, and close to the full fit
    full = fit_revel(events, stats, n)
    a = fit_revel(events, stats, n; n_controls=10, rng=Xoshiro(4))
    b = fit_revel(events, stats, n; n_controls=10, rng=Xoshiro(4))
    @test coef(a) == coef(b)
    @test all(abs.(coef(a) .- coef(full)) .< 3 .* stderror(full))
    @test !Networks.is_exact(a)
    @test all(a.fit.sampling_probs .≈ 10 / 29)

    sandwich = fit_revel(events, stats, n; se=:sandwich)
    @test sandwich.engine === :design
    @test Networks.se_method(sandwich) === :sandwich
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
    @test occursin("20 listed dyads", sprint(show, tfit))

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
    @test Networks.tie_method(br) === :breslow && Networks.tie_method(ef) === :efron
    @test !Networks.is_exact(br)
    @test coef(fit_revel(coarse, stats, n; ties=:breslow, engine=:design)) ≈ coef(br) atol = 1e-8
    @test coef(fit_revel(coarse, stats, n; ties=:efron)) ≈ coef(ef) atol = 1e-8

    @test_throws ArgumentError fit_revel(events, stats, n; model=:cox)
    @test_throws ArgumentError fit_revel(events, stats, n; engine=:glm)
    @test_throws ArgumentError fit_revel(Event{Float64}[], stats, n)
    @test_throws ArgumentError fit_revel(events, [Inertia(), Inertia()], n)
    @test_throws ArgumentError fit_revel(events, stats, n; engine=:relevent, riskset=:sender)
    @test_throws ArgumentError fit_revel(events, stats, n; engine=:relevent, se=:sandwich)
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
    # on its own it is constant across every risk set
    alone = fit_revel(events, [Inertia(transform=:log1p), late], n; engine=:design)
    @test alone.fit.singular
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
    @test names(profile) == ["value", "loglik", "aic", "bic", "converged", "best"]
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
                       "probability", "rank", "rank_fraction", "deviance_residual",
                       "null_residual", "surprise"]
    @test all(0 .< d.probability .<= 1)
    @test all(1 .<= d.rank .<= 30)
    # the deviance residuals add up to the fitted deviance
    @test sum(d.deviance_residual) ≈ -2loglikelihood(fit)
    @test all(d.null_residual .≈ 2log(30))
    # before any event every dyad is equally likely and shares the average rank
    @test d.probability[1] ≈ 1 / 30 && d.rank[1] == 15.5
    @test d.surprise ≈ -log2.(d.probability)

    s = prediction_summary(fit; ks=(1, 5, 0.2))
    @test s.n_events == 300 && s.ks == [1, 5, 0.2]
    @test issorted(s.recall) && s.recall[1] == count(<=(1), d.rank) / 300
    @test s.recall[3] == count(<=(6), d.rank) / 300          # 20 % of 30 dyads
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
    @test result isa Networks.GOFResult
    @test n_simulations(result) == 40
    @test [s.name for s in result.statistics] ==
          ["mechanism shares", "degree concentration", "closing times (events)"]
    @test result.statistics[1].observed ≈ collect(values(mechanism_shares(events)))
    @test all(0 .< s.p_values[k] <= 1 for s in result.statistics for k in eachindex(s.p_values))
    Random.seed!(1); r1 = gof(fit; n_sim=10, rng=Xoshiro(3))
    Random.seed!(2); r2 = gof(fit; n_sim=10, rng=Xoshiro(3))
    @test r1.statistics[1].simulated == r2.statistics[1].simulated
    @test occursin("Goodness-of-fit", sprint(show, result))

    # a model without reciprocity cannot reproduce data generated with strong
    # reciprocity
    strong = simulate_events(stats, [0.2, 2.5], n, 300; rng=Xoshiro(4))
    poor = gof(fit_revel(strong, stats[1:1], n); n_sim=200, rng=Xoshiro(5))
    good = gof(fit_revel(strong, stats, n); n_sim=200, rng=Xoshiro(5))
    # ... it concentrates activity in too few dyads and actors
    @test maximum(poor.statistics[2].p_values) < 0.05
    @test minimum(good.statistics[2].p_values) > 0.05

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
    @test lookup(:relevent, "FrPSndSnd") == "Inertia(scaling=:prop, empty=1/(n-1))"
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
    # Revel attaches to the shared Networks/StatsAPI generics (`compute`, `name`,
    # `gof`, `coef`, …) — must contain a fenced ```julia block, and every block
    # must run in a fresh module. Names re-exported unchanged from REM, Relevent
    # and Networks are documented where they are defined.
    reexported = (:Event, :InteractionHistory, :update_history!, :PShift, :pshift_types,
                  :n_simulations)
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
    failed = String[]
    for (nm, block) in blocks
        mod = Module()
        try
            Core.eval(mod, Meta.parseall(block))
        catch err
            @error "docstring example of $nm failed" exception = (err, catch_backtrace())
            push!(failed, nm)
        end
    end
    @test isempty(failed)
    @test length(blocks) >= 90
end

@testset "Namespace" begin
    # no export shadows Base
    @test isempty([s for s in names(Revel) if isdefined(Base, s) && Base.isexported(Base, s) &&
                                              getfield(Base, s) !== getfield(Revel, s)])
    # a name Revel shares with REM, Relevent or Networks is the SAME binding, so
    # co-loading leaves every name usable unqualified
    for pkg in (REM, Relevent, Networks), sym in names(Revel)
        sym === :Revel && continue
        if sym in names(pkg)
            @test getfield(pkg, sym) === getfield(Revel, sym)
        end
    end
    @test Revel.compute === REM.compute === Networks.compute
    @test Revel.gof === Networks.gof
    @test Revel.coeftable === Networks.coeftable
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
        using ERGM, SNA, Siena, REM, Relevent, Revel
        bad = Symbol[]
        for nm in names(Revel)
            nm === :Revel && continue
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
    @test compute(SubsetRepetition(2; memory=Interval(1.0, 4.0)), h, [1, 2], none, 6.0) == 1.0
    @test compute(SubsetRepetition(2; memory=Interval(1.0, 4.0)), h, [2, 3], none, 6.0) == 1.0
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
    design = hyper_design(events, stats, 5; n_controls=6, rng=Xoshiro(21))
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
    full = hyper_design(events, stats, 5; n_controls=29)
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
    @test Networks.tie_method(fit) === :breslow
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
    fit = fit_rhem(events, stats, 5; n_controls=30, rng=Xoshiro(6))
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
    # hyperevent design is the full dyadic risk set
    dstats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
    hstats = [DirectedSubsetRepetition(1, 1; transform=:log1p),
              HyperReciprocation(transform=:log1p)]
    dfit = fit_revel(dyadic, dstats, 5)
    hfit = fit_rhem(hyper, hstats, 5; n_controls=19)
    @test coef(hfit) ≈ coef(dfit) atol = 1e-6
    @test stderror(hfit) ≈ stderror(dfit) atol = 1e-6
    @test loglikelihood(hfit) ≈ loglikelihood(dfit) atol = 1e-6
    @test nobs(hfit) == nobs(dfit) == 80

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
    @test coeftable(fit) isa Networks.CoefficientTable
    # the StatsAPI surface and the result-metadata protocol, as for every model
    @test all(values(Networks.check_statsapi(fit; strict=true)))
    @test Networks.estimand(fit) == Networks.estimand(fit.fit)
    @test Networks.objective(fit) == Networks.objective(fit.fit)
    @test Networks.is_exact(fit) == Networks.is_exact(fit.fit)
    @test Networks.se_method(fit) === :hessian
    @test Networks.tie_method(fit) === :none
    @test Networks.missing_method(fit) == Networks.missing_method(fit.fit)
    @test Networks.approximations(fit) == Networks.approximations(fit.fit)
    @test Networks.fit_metadata(fit) isa Networks.ResultMetadata
    @test occursin("relational hyperevent model", sprint(show, fit))

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
    @test Networks.se_method(robust) === :sandwich && coef(robust) == coef(fit)

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
    g = Networks.load_golden(joinpath(pkgdir(Revel), "test", "fixtures",
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
    interval = Interval(g.values["interval_lo"], g.values["interval_hi"])

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
            @test Networks.check_golden(g, key, actual)
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
        @test !Networks.check_golden(g, key, plain)     # the population form is NOT remstats' "std"
    end

    # ---- memory = "window" and "interval" ---------------------------------------
    # remstats keeps an event of age a when a <= width ("window") and when
    # lo < a <= hi ("interval"): exactly `Window` and `Interval`. The time grid
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
        "rrankSend" => RecencyRank(:send; memory=memory),
        "recencyContinue" => TimeSince(:dyad; memory=memory),
        "psABBA" => PShift(:AB_BA)]
    for (key, stat) in memory_effects(window)
        parity("window_" * key, stat)
    end
    for (key, stat) in memory_effects(interval)
        key == "outdegreeSender_prop" && continue
        parity("interval_" * key, stat)
    end
    # The boundary conventions are really exercised by the fixture:
    @test !Networks.check_golden(g, "window_inertia", via_history(
        Inertia(memory=KernelMemory(a -> a < window.width ? 1.0 : 0.0)), events))
    @test !Networks.check_golden(g, "interval_inertia", via_history(
        Inertia(memory=KernelMemory(a -> interval.lo <= a <= interval.hi ? 1.0 : 0.0)), events))
    @test !Networks.check_golden(g, "interval_inertia", via_history(
        Inertia(memory=KernelMemory(a -> interval.lo < a < interval.hi ? 1.0 : 0.0)), events))

    let key = "interval_outdegreeSender_prop"
        push!(covered, key)
        actual = via_history(OutdegreeSender(memory=interval, scaling=:prop, empty=p_node), events)
        # A deliberate difference. remstats returns 1/n at the FIRST time point
        # only and 0 at a later one whose memory holds no event (0/0 → NaN → 0);
        # Revel returns `empty` whenever there is nothing to take a share of.
        @test !Networks.check_golden(g, key, actual)
        # … and that is the whole difference: zero the later empty-memory rows
        in_memory = reshape(via_history(Inertia(memory=interval), events), n_dyads, :)
        patched = reshape(copy(actual), n_dyads, :)
        for k in 2:length(events)
            iszero(sum(view(in_memory, :, k))) && (patched[:, k] .= 0.0)
        end
        @test Networks.check_golden(g, key, vec(patched))
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
            @test Networks.check_golden(g, key, lagged)
            # A deliberate difference: Revel evaluates the decay at the time of
            # the event being explained, as remstats documents. The two then
            # differ by exactly the decay over the last waiting time.
            own = via_history(stat, evs)
            @test !Networks.check_golden(g, key, own)
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
        @test Networks.check_golden(g, key, via_history(stat, events; directed=false))
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
    @test !Networks.check_golden(g, "undirected_psABAB",
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

    # Every statistic array in the fixture is asserted above
    arrays = [k for (k, v) in g.values if v isa AbstractVector &&
              length(v) in (length(events) * n_dyads, length(events) * n_dyads ÷ 2)]
    @test length(arrays) == 131
    @test isempty(setdiff(arrays, covered))
    @test isempty(setdiff(covered, arrays))
end
