# Endogenous Effects API Reference

```@meta
CurrentModule = Revel
```

## The statistic protocol

Revel's statistics are methods of the ecosystem's shared `compute` and `name`
generics (Networks.jl), in both signatures its relational event packages use.

```@docs
AbstractRevelStatistic
compute(::AbstractRevelStatistic, ::REM.EventNetworkState, ::Int, ::Int)
name(::AbstractRevelStatistic)
build_history
```

## The five configurations

```@docs
DyadEffect
DegreeEffect
DyadDegreeEffect
TwoPathEffect
FourCycleEffect
```

## Dyadic effects

```@docs
Inertia
Reciprocation
DyadActivity
```

## Degree effects

```@docs
OutdegreeSender
IndegreeSender
TotaldegreeSender
OutdegreeReceiver
IndegreeReceiver
TotaldegreeReceiver
TotaldegreeDyad
DegreeMin
DegreeMax
DegreeDiff
DegreeAssortativity
```

## Triadic effects

```@docs
OTP
ITP
OSP
ISP
SharedPartners
BalanceEffect
matching_third
```

## Order and recency

```@docs
RecencyRank
TimeSince
inverse_gap
PShiftABAB
UndirectedPShift
```

## Neighbourhood statistics

```@docs
NodeTransitivity
StructuralSimilarity
```
