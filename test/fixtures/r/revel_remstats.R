if (nzchar(Sys.getenv("R_LIBS_USER"))) .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
# Golden fixture: the tie-oriented statistics of R `remstats` on a small fixed
# event sequence, for Revel.jl's "remstats design parity (golden)" testset.
#
# Regenerate from the Revel.jl package root:
#
#   Rscript test/fixtures/r/revel_remstats.R > test/fixtures/revel_remstats.toml
#
# Needs R with `remify` and `remstats` (generated with remstats 4.1.0 / remify
# 4.1.0). If they live in a private library, name it in R_LIBS_USER; the first
# line of this script prepends that directory to .libPaths():
#
#   R_LIBS_USER=/path/to/Rlib Rscript test/fixtures/r/revel_remstats.R > test/fixtures/revel_remstats.toml
#
# Deterministic: no optimizer and no RNG. Every statistic is the full
# remstats array (first = 1, so the row of the FIRST event, whose history is
# empty, is included) flattened event-major, then dyads in the order
# sender 1..n, receiver 1..n, sender != receiver (directed runs) or (i, j),
# i < j, i-major (undirected runs). remstats' own column order is read off
# attr(stats, "riskset") and mapped onto that order, never assumed.
#
# The data (written out literally below):
# - 6 actors, 36 directed events, strictly increasing times on a 0.25 grid
#   (so event ages are exact binary fractions and hit the window width 3.0 and
#   both interval bounds 1.5 and 5.0 EXACTLY: the boundary conventions are
#   pinned, not dodged);
# - repeated dyads (1->2 five times), reciprocated dyads (1<->2, 2<->3, 3<->4,
#   1<->4, 2<->4), closed triads among actors 1-4;
# - actor 6 never sends; actor 5 is silent between t = 9.75 and t = 23.0;
# - a numeric actor covariate `x`, a categorical one `g`, a dyadic matrix `d`,
#   and (for the weighted run only) an event weight;
# - (for the typed_* keys only) an event type, "b" for every third event and
#   "a" otherwise, and (for the event_* keys) an event attribute `z`.
suppressMessages({library(remify); library(remstats)})

n <- 6L
ev <- matrix(c(
   0.50, 1, 2,   1.25, 2, 1,   2.00, 1, 2,   2.75, 1, 3,   3.50, 3, 2,   4.00, 2, 3,
   4.75, 3, 1,   5.50, 1, 2,   6.25, 4, 1,   7.00, 1, 4,   7.50, 4, 2,   8.25, 2, 4,
   9.00, 5, 1,   9.75, 1, 5,  10.50, 3, 4,  11.00, 4, 3,  11.75, 2, 6,  12.50, 1, 6,
  13.25, 2, 1,  14.00, 3, 6,  14.50, 1, 3,  15.25, 3, 2,  16.00, 2, 3,  16.75, 4, 6,
  17.50, 1, 2,  18.00, 2, 4,  18.75, 4, 1,  19.50, 3, 1,  20.25, 1, 4,  21.00, 2, 1,
  21.50, 3, 4,  22.25, 4, 3,  23.00, 5, 2,  23.75, 2, 5,  24.50, 1, 2,  25.00, 4, 2),
  ncol = 3, byrow = TRUE)
m <- nrow(ev)
weight <- c(1.5, 0.5, 2, 1, 0.25, 3, 1, 0.75, 2.5, 1, 0.5, 1.25, 2, 1, 0.5, 1.75,
            1, 2.25, 0.5, 1, 3, 0.25, 1.5, 1, 2, 0.75, 1, 1.25, 0.5, 2, 1, 1.5,
            0.25, 1, 2.5, 0.5)
type <- ifelse(seq_len(m) %% 3 == 0, "b", "a")   # event type (typed run only)
z <- round(sin(seq_len(m)), 3)                    # event attribute (event() only)
x <- c(1.5, -0.5, 2, 0, 3.25, 1)                  # numeric actor covariate
g <- c("a", "b", "a", "c", "b", "a")              # categorical actor covariate
d <- outer(seq_len(n), seq_len(n), function(i, j) (3 * i - j) / 8)   # dyadic
diag(d) <- 0

window_width   <- 3.0          # memory = "window"
decay_halflife <- 2.0          # memory = "decay"
interval_lo    <- 1.5          # memory = "interval"
interval_hi    <- 5.0

el  <- data.frame(time = ev[, 1], actor1 = as.integer(ev[, 2]), actor2 = as.integer(ev[, 3]))
elw <- cbind(el, weight = weight)
elt <- cbind(el, type = type)
info <- data.frame(name = seq_len(n), time = 0, x = x, g = g)
dimnames(d) <- list(seq_len(n), seq_len(n))

reh  <- remify(el,  model = "tie", directed = TRUE,  ordinal = FALSE, actors = seq_len(n))
rehu <- remify(el,  model = "tie", directed = FALSE, ordinal = FALSE, actors = seq_len(n))
rehw <- remify(elw, model = "tie", directed = TRUE,  ordinal = FALSE, actors = seq_len(n))
# event types on a dyadic risk set (extend_riskset_by_type = FALSE, the default)
reht <- remify(elt, model = "tie", directed = TRUE,  ordinal = FALSE, actors = seq_len(n))
# ordinal = TRUE: remify replaces the event times by the event index 1..M
reho <- remify(el,  model = "tie", directed = TRUE,  ordinal = TRUE,  actors = seq_len(n))

num <- function(v) paste(sprintf("%.17g", v), collapse = ", ")
num_array <- function(v) paste(sprintf("%.17g", v), collapse = ",")   # compact

# The fixture's dyad order -> remstats' column index, from the riskset attribute
column_map <- function(stats, directed) {
  rs <- attr(stats, "riskset")
  a <- as.integer(as.character(rs[[1]])); b <- as.integer(as.character(rs[[2]]))
  pairs <- if (directed) {
    do.call(rbind, lapply(seq_len(n), function(s) cbind(s, setdiff(seq_len(n), s))))
  } else {
    do.call(rbind, lapply(seq_len(n - 1L), function(i) cbind(i, (i + 1L):n)))
  }
  cols <- vapply(seq_len(nrow(pairs)), function(k) {
    hit <- which(a == pairs[k, 1] & b == pairs[k, 2])
    stopifnot(length(hit) == 1L)
    hit
  }, integer(1))
  stopifnot(length(cols) == nrow(rs))
  cols
}

# One remstats() call; `keys` names the fixture key of each statistic slice in
# the order of the formula's terms (the leading baseline slice is dropped).
emit <- function(keys, effects, reh, directed = TRUE, ...) {
  stats <- remstats(reh = reh, tie_effects = effects, first = 1, ...)
  stopifnot(dim(stats)[1] == m)
  labels <- dimnames(stats)[[3]]
  slices <- which(labels != "baseline")     # by position: labels can repeat
  stopifnot(length(slices) == length(keys))
  cols <- column_map(stats, directed)
  for (k in seq_along(keys)) {
    values <- as.vector(t(stats[, cols, slices[k]])) + 0   # event-major; -0 -> 0
    stopifnot(all(is.finite(values)))
    cat(sprintf("# remstats: %s\n%s = [%s]\n", labels[slices[k]], keys[k], num_array(values)))
  }
}

cat('name = "revel_remstats"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\nremstats_version = "%s"\nremify_version = "%s"\n',
            getRversion(), packageVersion("remstats"), packageVersion("remify")))
cat('seed = 0 # deterministic; no random draws\n')
cat('script = "test/fixtures/r/revel_remstats.R"\n')
cat('date = "2026-09-30"\n')
cat('dataset = "36 fixed directed events on a 0.25 time grid, 6 actors; actor 6 never sends, actor 5 silent for t in (9.75, 23); numeric and categorical actor covariates, a dyadic matrix, event weights for the weighted_* keys, event types for the typed_* keys, an event attribute z for the event_* keys"\n')
cat('method = "remify::remify(model = \'tie\', ordinal = FALSE, riskset = \'full\') + remstats::remstats(first = 1): the full statistic array, event-major, dyads sender-major (directed) or i < j (undirected_* keys)"\n\n')
cat('[tolerance]\n')
cat('# Deterministic on both sides. Counts, ranks, minima, participation shifts and\n')
cat('# covariate forms are exact in Float64. The rest differs only by roundoff: one\n')
cat('# division (prop, recency), a z-score over the 30 dyads of a risk set (std),\n')
cat('# and for decay a sum of at most 35 exp() weights that remstats adds event by\n')
cat('# event while Revel carries a running decayed total. Every value is O(1)-O(40);\n')
cat('# the largest difference observed when the fixture was generated was 2.7e-15.\n')
cat('# 1e-10 is orders of magnitude above that roundoff and nine below the smallest\n')
cat('# effect of a convention difference (one boundary event, one unit of degree,\n')
cat('# sample vs population sd), so it cannot hide one. Do not loosen it.\n')
cat('default = 1e-10\n\n[values]\n')
cat(sprintf('n_actors = %d\ninput_time = [%s]\ninput_sender = [%s]\ninput_receiver = [%s]\ninput_weight = [%s]\n',
            n, num(ev[, 1]), num(ev[, 2]), num(ev[, 3]), num(weight)))
cat(sprintf('input_type = [%s]\nevent_z = [%s]\n', paste(sprintf('"%s"', type), collapse = ", "),
            num(z)))
cat(sprintf('covariate_x = [%s]\ncovariate_g = [%s]\n', num(x),
            paste(sprintf('"%s"', g), collapse = ", ")))
cat(sprintf('# row-major: covariate_d[(s - 1) * n + r] is the value of dyad s -> r\ncovariate_d = [%s]\n',
            num(as.vector(t(d)))))
cat(sprintf('window_width = %s\ndecay_halflife = %s\ninterval_lo = %s\ninterval_hi = %s\n',
            num(window_width), num(decay_halflife), num(interval_lo), num(interval_hi)))

# ---- full memory, directed ---------------------------------------------------
emit(c("full_inertia", "full_reciprocity", "full_indegreeSender", "full_indegreeReceiver",
       "full_outdegreeSender", "full_outdegreeReceiver", "full_totaldegreeSender",
       "full_totaldegreeReceiver", "full_totaldegreeDyad"),
     ~ inertia() + reciprocity() + indegreeSender() + indegreeReceiver() +
       outdegreeSender() + outdegreeReceiver() + totaldegreeSender() +
       totaldegreeReceiver() + totaldegreeDyad(), reh)
emit(c("full_otp", "full_itp", "full_osp", "full_isp"),
     ~ otp() + itp() + osp() + isp(), reh)
emit(c("full_otp_unique", "full_itp_unique", "full_osp_unique", "full_isp_unique"),
     ~ otp(unique = TRUE) + itp(unique = TRUE) + osp(unique = TRUE) + isp(unique = TRUE), reh)
emit(c("full_psABBA", "full_psABBY", "full_psABXA", "full_psABXB", "full_psABXY",
       "full_psABAY", "full_psABAB"),
     ~ psABBA() + psABBY() + psABXA() + psABXB() + psABXY() + psABAY() + psABAB(), reh)
emit(c("full_rrankSend", "full_rrankReceive", "full_recencyContinue",
       "full_recencySendSender", "full_recencySendReceiver",
       "full_recencyReceiveSender", "full_recencyReceiveReceiver"),
     ~ rrankSend() + rrankReceive() + recencyContinue() + recencySendSender() +
       recencySendReceiver() + recencyReceiveSender() + recencyReceiveReceiver(), reh)

# ---- scaling -----------------------------------------------------------------
emit(c("full_inertia_prop", "full_reciprocity_prop", "full_indegreeSender_prop",
       "full_indegreeReceiver_prop", "full_outdegreeSender_prop",
       "full_outdegreeReceiver_prop", "full_totaldegreeSender_prop",
       "full_totaldegreeReceiver_prop", "full_totaldegreeDyad_prop"),
     ~ inertia(scaling = "prop") + reciprocity(scaling = "prop") +
       indegreeSender(scaling = "prop") + indegreeReceiver(scaling = "prop") +
       outdegreeSender(scaling = "prop") + outdegreeReceiver(scaling = "prop") +
       totaldegreeSender(scaling = "prop") + totaldegreeReceiver(scaling = "prop") +
       totaldegreeDyad(scaling = "prop"), reh)
emit(c("full_inertia_std", "full_indegreeReceiver_std", "full_otp_std", "full_send_x_std"),
     ~ inertia(scaling = "std") + indegreeReceiver(scaling = "std") + otp(scaling = "std") +
       send("x", attr_actors = info, scaling = "std"), reh)

# ---- exogenous ---------------------------------------------------------------
emit(c("exo_send_x", "exo_receive_x", "exo_same_g", "exo_difference_x_abs",
       "exo_difference_x", "exo_average_x", "exo_minimum_x", "exo_maximum_x", "exo_tie_d"),
     ~ send("x", attr_actors = info) + receive("x", attr_actors = info) +
       same("g", attr_actors = info) + difference("x", attr_actors = info, absolute = TRUE) +
       difference("x", attr_actors = info, absolute = FALSE) + average("x", attr_actors = info) +
       minimum("x", attr_actors = info) + maximum("x", attr_actors = info) +
       tie("d", attr_dyads = d), reh)

# ---- interactions (the main-effect slices are emitted above) -------------------
local({
  stats <- remstats(reh = reh, first = 1,
                    tie_effects = ~ inertia():send("x", attr_actors = info) +
                      outdegreeSender():indegreeReceiver() +
                      send("x", attr_actors = info):receive("x", attr_actors = info) +
                      event("z", event_attr = data.frame(z = z)):inertia())
  cols <- column_map(stats, TRUE)
  for (p in list(c("full_inertia_x_send_x", "inertia:send_x"),
                 c("full_outdegreeSender_x_indegreeReceiver", "outdegreeSender:indegreeReceiver"),
                 c("full_send_x_x_receive_x", "send_x:receive_x"),
                 c("event_z_x_inertia", "inertia:event_z"))) {
    stopifnot(p[2] %in% dimnames(stats)[[3]])
    cat(sprintf("# remstats: %s\n%s = [%s]\n", p[2], p[1], num_array(as.vector(t(stats[, cols, p[2]])) + 0)))
  }
})

# ---- memory variants ---------------------------------------------------------
memory_effects <- ~ inertia() + reciprocity() + outdegreeSender() + indegreeReceiver() +
  totaldegreeDyad() + otp() + itp() + osp() + isp() + otp(unique = TRUE) +
  inertia(scaling = "prop") + outdegreeSender(scaling = "prop") +
  rrankSend() + recencyContinue() + psABBA()
memory_keys <- c("inertia", "reciprocity", "outdegreeSender", "indegreeReceiver",
                 "totaldegreeDyad", "otp", "itp", "osp", "isp", "otp_unique",
                 "inertia_prop", "outdegreeSender_prop", "rrankSend", "recencyContinue",
                 "psABBA")
emit(paste0("window_", memory_keys), memory_effects, reh,
     memory = "window", memory_value = window_width)
emit(paste0("decay_", memory_keys), memory_effects, reh,
     memory = "decay", memory_value = decay_halflife)
emit(paste0("interval_", memory_keys), memory_effects, reh,
     memory = "interval", memory_value = c(interval_lo, interval_hi))

# ---- undirected --------------------------------------------------------------
emit(c("undirected_inertia", "undirected_totaldegreeDyad", "undirected_degreeMin",
       "undirected_degreeMax", "undirected_degreeDiff", "undirected_sp",
       "undirected_sp_unique", "undirected_recencyContinue", "undirected_psABAB",
       "undirected_psABAY", "undirected_totaldegreeDyad_prop", "undirected_degreeMin_prop",
       "undirected_degreeMax_prop"),
     ~ inertia() + totaldegreeDyad() + degreeMin() + degreeMax() + degreeDiff() + sp() +
       sp(unique = TRUE) + recencyContinue() + psABAB() + psABAY() +
       totaldegreeDyad(scaling = "prop") + degreeMin(scaling = "prop") +
       degreeMax(scaling = "prop"), rehu, directed = FALSE)
emit(c("undirected_window_inertia", "undirected_window_degreeMin", "undirected_window_sp"),
     ~ inertia() + degreeMin() + sp(), rehu, directed = FALSE,
     memory = "window", memory_value = window_width)

# ---- event weights (a `weight` column in the edgelist) ---------------------------
emit(c("weighted_inertia", "weighted_outdegreeSender", "weighted_indegreeReceiver",
       "weighted_reciprocity", "weighted_otp", "weighted_otp_unique",
       "weighted_inertia_prop", "weighted_outdegreeSender_prop", "weighted_recencyContinue"),
     ~ inertia() + outdegreeSender() + indegreeReceiver() + reciprocity() + otp() +
       otp(unique = TRUE) + inertia(scaling = "prop") + outdegreeSender(scaling = "prop") +
       recencyContinue(), rehw)
emit(c("weighted_decay_inertia", "weighted_decay_outdegreeSender"),
     ~ inertia() + outdegreeSender(), rehw, memory = "decay", memory_value = decay_halflife)

# ---- event types: consider_type = "separate" (one statistic per past type) -------
emit(c("typed_inertia_a", "typed_inertia_b", "typed_reciprocity_a", "typed_reciprocity_b",
       "typed_otp_a", "typed_otp_b", "typed_outdegreeSender_a", "typed_outdegreeSender_b"),
     ~ inertia(consider_type = "separate") + reciprocity(consider_type = "separate") +
       otp(consider_type = "separate") + outdegreeSender(consider_type = "separate"), reht)

# ---- ordinal = TRUE (the clock is the event index) -------------------------------
emit(c("ordinal_window_inertia", "ordinal_window_otp", "ordinal_recencyContinue",
       "ordinal_recencyReceiveSender"),
     ~ inertia() + otp() + recencyContinue() + recencyReceiveSender(), reho,
     memory = "window", memory_value = window_width)
