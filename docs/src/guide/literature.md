# Literature

Revel.jl implements the effect catalogue of a scoping review of the relational
event model literature from 2008 to 2026. This page says what that review
covered, what it concluded, and where each part of the package comes from.

## The review

The corpus was assembled from OpenAlex (eleven queries: phrase searches and
forward citations of five seed papers), screened to **210 works** — 95
empirical applications, 58 methodological papers, 25 methods papers with an
empirical illustration, 22 reviews or tutorials and 10 software records. The
fitted effect list was read in full text for 55 of them; software catalogues
were read from package sources and help pages.

It is a scoping review by a single coder, not a systematic review in the PRISMA
sense: one database, many paywalled full texts, papers inspected by
accessibility. Its conclusions about what the literature *contains* are better
supported than its counts of how often each effect is *used*.

Its main conclusions, each of which shaped the package:

| Conclusion | Consequence here |
|---|---|
| The literature holds about five structural configurations crossed with a few measurement choices. | Five parametric statistics on an [`EventLayer`](@ref); named constructors on top. |
| The founding papers after Butts (2008) changed the measurement, not the configuration: memory became a modelling dimension of its own. | Memory kernels are orthogonal to every effect; [`profile_memory`](@ref) and [`interval_partition`](@ref) estimate the memory. |
| The same configuration has a different name and default in each package. | [`effect_catalogue`](@ref); the `combine`, `empty`, `scaling` and `normalized` keywords. |
| Exogenous effects are inherited forms; the literature's own contribution is an identification rule. | [`GlobalEffect`](@ref) exists to be interacted; [`statistic_collinearity`](@ref) reports unidentified terms. |
| Interactions are rare as product terms and common as filters, type splits and separate fits — three constructions that are not equivalent. | Each construction has its own name: see [Interactions](interactions.md). |
| Closure and popularity, the second and third most reported effects, are fragile against actor heterogeneity and repetition. | [`score_test`](@ref) asks whether a term still has something to explain. |
| Goodness of fit has no consensus; three families of proposals exist. | All three: see [Goodness of fit](gof.md). |
| No review offers a collinearity diagnostic, or advice on centring before forming a product. | [`statistic_collinearity`](@ref), [`Transformed`](@ref). |
| Hyperedge statistics are the one genuinely new structural family since 2016. | [Relational hyperevents](hyperevents.md). |

## Sources by part of the package

Keys refer to `docs/references.bib`. The text of this documentation cites the
year a work first appeared online, as the review does; the bibliography records
the year of the journal issue, so a few keys carry a later year than the
citation.

| Part | Sources | Keys |
|---|---|---|
| The framework; persistence, preferential attachment, recency ranks, the four triads, participation shifts | Butts 2008; Butts & Marcum 2017; Gibson 2003 | `butts2008relational`, `butts2017relational`, `gibson2003participation` |
| Inertia, reciprocity, signed events, balance, half-life decay | Brandes, Lerner & Snijders 2009; Lerner, Bussmann, Snijders & Brandes 2013 | `brandes2009networks`, `lerner2013modeling` |
| Count statistics, product-form triads, time-varying coefficients | Vu, Hunter, Smyth & Asuncion 2011; Vu, Asuncion, Hunter & Smyth 2011 | `vu2011continuous`, `vu2011dynamic` |
| Interval partition of the past; sender stratification and identification | Perry & Wolfe 2013 | `perry2013point` |
| Windows; short- and long-term frames | de Nooy 2011; Quintane, Pattison, Robins & Mol 2013 | `denooy2011networks`, `quintane2013shortb` |
| Recency rank of sending; multilevel sequences | DuBois, Butts, McFarland & Smyth 2013 | `dubois2013hierarchical` |
| Proportion-scaled inertia and reciprocation | Kitts, Lomi, Mascia, Pallotti & Quintane 2017 | `kitts2017investigating` |
| Degree versus intensity; power-law memory; assortativity; stratified fits | Vu, Lomi, Mascia & Pallotti 2017; Bianchi & Lomi 2023; Zappa & Vu 2021 | `vu2017relational`, `bianchi2023ties`, `zappa2021markets` |
| Estimated memory; linear decay; time-ordered transitivity; typed memory | Arena, Mulder & Leenders 2022, 2023, 2025 | `arena2024bayesian`, `arena2023fast`, `arena2026weighting` |
| Actor-oriented models; windowed and cross-network effects; tertius | Stadtfeld & Block 2017; Stadtfeld, Hollway & Block 2017; Haunss & Hollway 2023 | `stadtfeld2017interactions`, `stadtfeld2017dynamic`, `haunss2023multimodal` |
| Two-mode effects, four-cycles, sampled controls | Lerner & Lomi 2020 | `lerner2020reliability` |
| Attribute-filtered statistics; contagion among similar actors | Brandenberger 2018; Malang, Brandenberger & Leifeld 2019 | `brandenberger2018trading`, `brandenberger2018rem`, `malang2019networks` |
| Stratified estimation by context | Amati, Lomi & Mascia 2019 | `amati2019some` |
| Moving-window estimation | Mulder & Leenders 2019; Meijerink-Bosman, Leenders & Mulder 2022 | `mulder2019modeling`, `meijerinkbosman2022dynamic` |
| Product terms and their interpretation; the remstats vocabulary | Meijerink-Bosman, Back, Geukes, Leenders & Mulder 2023 | `meijerinkbosman2023discovering` |
| Global covariates as moderators | Lembo, Juozaitienė, Vinciotti & Wit 2025 | `lembo2026relational` |
| Ghost triadic effects; temporal closure | Juozaitienė & Wit 2022, 2024, 2025 | `juozaitiene2022non`, `juozaitiene2024nodal`, `juozaitiene2025time` |
| Prediction-based fit | Butts 2008; Brandenberger 2019 | `butts2008relational`, `brandenberger2019predicting` |
| Residual-based fit; score processes | Boschi & Wit 2025; Lin, Wei & Ying 1993 | `boschi2026goodness`, `lin1993checking` |
| Simulation-based fit; closing times | Amati, Lomi & Snijders 2024 | `amati2024goodness` |
| Hyperevents | Lerner, Tranmer, Mowbray & Hâncean 2019; Lerner, Lomi, Mowbray, Rollings & Tranmer 2021; Lerner & Lomi 2023; Lerner & Hâncean 2023; Lerner, Hâncean & Perc 2025 | `lerner2019rem`, `lerner2021dynamica`, `lerner2023relational`, `lerner2023micro`, `lerner2025modeling` |
| Reviews and tutorials | Bianchi, Filippi-Mazzola, Lomi & Wit 2024; Butts, Lomi, Snijders & Stadtfeld 2023; Boschi & Wit 2026 | `bianchi2024relational`, `butts2023relational`, `boschi2026introduction` |

## What the review could not establish

- The concordance with rem, goldfish and eventnet rests on reading those
  packages, not on running them.
- Four frequently cited papers were not read in their journal versions
  (Quintane et al. 2013 and 2014, Stadtfeld & Geyer-Schulz 2011, Hoffman et al.
  2020).
- Usage frequencies (inertia in 76 % of inspected models, closure in 67 %,
  reciprocity in 71 % of directed one-mode models, participation shifts in
  13 %) come from 55 papers chosen by accessibility.
