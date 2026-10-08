# Hyperevents API Reference

```@meta
CurrentModule = Revel
```

## Hyperevents and their history

```@docs
HyperEvent
HyperHistory
update_hyper_history!
build_hyper_history
is_directed(::HyperEvent)
participants
```

## Statistics

```@docs
AbstractHyperStatistic
HyperedgeSize
HyperCovariate
```

### Undirected hyperevents

```@docs
SubsetRepetition
ExactRepetition
SharedPriorEvents
PriorSuccess
```

### Directed hyperevents

```@docs
DirectedSubsetRepetition
UnorderedRepetition
ReceiverSetRepetition
SenderReceiverSetRepetition
HyperSenderActivity
HyperReceiverPopularity
HyperReciprocation
OutInPopularity
InteractionAmongReceivers
```

### Closure

```@docs
HyperClosure
```

## Design, fitting and simulation

```@docs
hyper_design
HyperFit
fit_rhem
rhem
simulate_hyperevents
coefnames(::HyperFit)
```
