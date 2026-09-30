# Bounded performance gate: no tuning and no machine-dependent wall-clock
# thresholds. Two properties are pinned:
#   * `compute` allocates nothing on a warmed history, for every statistic family;
#   * absorbing a history costs O(events): twice the events, at most about twice
#     the bytes, for the accumulating kernels.
using Test, Random, Revel, REM, Relevent

function random_sequence(rng, n, m)
    events = Event{Float64}[]
    for i in 1:m
        s = rand(rng, 1:n)
        r = rand(rng, 1:(n - 1))
        push!(events, Event(s, r >= s ? r + 1 : r, Float64(i)))
    end
    return events
end

compute_bytes(stat, history, t) = @allocated compute(stat, history, 3, 7, t)

@testset "compute allocates nothing on a warmed history" begin
    events = random_sequence(Xoshiro(20260930), 30, 2000)
    history = build_history(events)
    t = events[end].time + 1
    x = collect(1.0:30)
    statistics = (Inertia(), Inertia(scaling=:prop), Reciprocation(memory=HalfLife(50.0)),
                  OutdegreeSender(), IndegreeReceiver(measure=:partners),
                  TotaldegreeDyad(), DegreeAssortativity(), OTP(), ITP(combine=:product),
                  OSP(combine=:count), ISP(memory=Window(200.0)), FourCycleEffect(),
                  RecencyRank(:send), RecencyRank(:receive), TimeSince(:dyad),
                  PShiftABAB(), NodeTransitivity(), StructuralSimilarity(),
                  SendEffect(x), MatchEffect(x), DiffEffect(x),
                  TieEffect(ones(30, 30)), TertiusEffect(x), MatchedDegree(x),
                  Interaction(Inertia(), ReceiveEffect(x)), Transformed(OTP(), log1p))
    bytes = Int[]
    for stat in statistics
        compute(stat, history, 3, 7, t); compute_bytes(stat, history, t)     # warm up
        push!(bytes, compute_bytes(stat, history, t))
    end
    @test all(==(0), bytes)
    println("compute bytes per statistic: ", bytes)
end

function absorb_bytes(stat, n, m)
    events = random_sequence(Xoshiro(1), n, m)
    history = build_history(events)
    t = events[end].time + 1
    return @allocated compute(stat, history, 1, 2, t)
end

@testset "absorbing a history is O(events)" begin
    for make in (() -> Inertia(), () -> OTP(memory=HalfLife(100.0)))
        absorb_bytes(make(), 20, 100)                                        # compile
        small = absorb_bytes(make(), 20, 2000)
        large = absorb_bytes(make(), 20, 4000)
        # dense storage depends on the actors, not on the events: the accumulating
        # kernels retain no per-event state
        @test large <= small + 4096
        println("absorb bytes (2000, 4000 events): ", (small, large))
    end
end
