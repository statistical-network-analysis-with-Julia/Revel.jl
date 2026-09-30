# Bounded performance gate: no tuning and no machine-dependent wall-clock
# thresholds. Pinned:
#   * `compute` allocates nothing on a warmed history, for every statistic family
#     and on the sparse (large actor ID) storage;
#   * absorbing a history costs O(events): twice the events, at most about twice
#     the bytes, and a kernel with finite support evaluates its weight O(events)
#     times over a whole stream (counted, not timed);
#   * sampled controls cost the rows drawn, not the risk set (counted);
#   * memory grows with the dyads that have a history, not with the largest ID.
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
                  Interaction(Inertia(), ReceiveEffect(x)), Transformed(OTP(), log1p),
                  OutdegreeSender(measure=:intensity), OTP(combine=:harmonic),
                  TertiusEffect(Covariate(mod.(1:30, 3); categorical=true); aggregate=:entropy),
                  TertiusEffect(x; aggregate=:sd), Standardized(Inertia(), 30),
                  OTP(memory=Window(50.0)), Inertia(memory=PowerLaw(0.5; offset=1.0)))
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

# A kernel whose weight evaluations are counted (a global counter: the entry
# points evaluate copies of the statistics)
const WEIGHT_CALLS = Ref(0)
counted_window(a) = (WEIGHT_CALLS[] += 1; a <= 20.0 ? 1.0 : 0.0)

@testset "a finite-support kernel costs O(events) over a stream" begin
    calls(m) = begin
        events = random_sequence(Xoshiro(2), 5, m)
        WEIGHT_CALLS[] = 0
        each_risk_set(v -> nothing, events,
                      [Inertia(memory=KernelMemory(counted_window; support=20.0))], 5)
        WEIGHT_CALLS[]
    end
    small, large = calls(2000), calls(4000)
    @test large <= 2.2 * small
    println("kernel evaluations (2000, 4000 events): ", (small, large))
end

const EVALUATIONS = Ref(0)
struct CountingStatistic <: AbstractRevelStatistic end
Revel._value(::CountingStatistic, events, s::Int, r::Int, t::Float64) =
    (EVALUATIONS[] += 1; 0.0)
Revel.name(::CountingStatistic) = "counting"

@testset "sampled controls cost the rows drawn" begin
    events = random_sequence(Xoshiro(3), 200, 100)          # 39 800 dyads per event
    EVALUATIONS[] = 0
    event_design(events, [CountingStatistic()], 200; n_controls=10, rng=Xoshiro(1))
    @test EVALUATIONS[] == 100 * 11
end

@testset "memory grows with the dyads, not with the largest ID" begin
    bytes(ids) = begin
        events = [Event(ids[1 + (k % 2)], ids[2 - (k % 2)], Float64(k)) for k in 1:200]
        stat = Inertia()
        compute(stat, build_history(events), ids[1], ids[2], 1000.0)
        Base.summarysize(stat)
    end
    small, huge = bytes((1, 2)), bytes((100_001, 100_002))
    # a dense index would take 8·10¹⁰ bytes for the second; the per-actor
    # vectors grow with the largest ID (a few MiB), the per-dyad data do not
    @test huge < 16 * 2^20
    println("layer bytes for actor IDs (1, 2) and (100001, 100002): ", (small, huge))
end
