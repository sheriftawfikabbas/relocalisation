#!/usr/bin/env bash
# Fair per-parameter evaluation budget sweep for the re-localization study.
#
# Addresses the confound noted in the companion manuscript (Sec. 2.1): the production
# sweep (run_relocalization.sh) fixes a single total evaluation budget (1000
# evals, 16 restarts) across all four drive ansatze, even though they have
# different free-parameter counts (single-tone=2, chirp=3, two-tone=5,
# bang-bang=8). That standardizes the comparison but leaves the
# higher-dimensional families under-sampled per parameter, so their weaker
# scores may reflect optimizer under-coverage rather than genuine ansatz
# inferiority.
#
# This script instead fixes evaluations PER FREE PARAMETER (default: 500,
# matching single-tone's implied rate under the original fixed budget:
# 1000 evals / 2 params) and scales max_evals per ansatz accordingly:
#   single-tone (dim=2): max_evals = 500 * 2 = 1000  (unchanged)
#   chirp       (dim=3): max_evals = 500 * 3 = 1500
#   two-tone    (dim=5): max_evals = 500 * 5 = 2500
#   bang-bang   (dim=8): max_evals = 500 * 8 = 4000
# n_restarts is held fixed (default 16) across all families: restart count
# guards against a single local search converging to a poor local minimum,
# independent of dimensionality; only the per-restart search budget is scaled
# here, so this run isolates the effect of evaluation budget alone.
#
# Each ansatz is run as its own full production sweep (both scenarios A and
# B, all 52 beta values), parallelised across the (beta x scenario) grid via
# JOBS. The four ansatz runs are executed SEQUENTIALLY within one detached
# tmux session (each already saturates JOBS workers internally, so running
# ansatze concurrently would oversubscribe the CPU rather than speed anything
# up). After all four finish, the per-ansatz results are merged into combined
# sweep_A.csv / sweep_B.csv / beta_star.csv / figures, directly comparable to
# results/relocalization/.
#
# Cost warning: the equal-budget production sweep (all 4 ansatze at
# max_evals=1000 each, scenario A only) took ~8.8 hours wall time on the
# hardware it was last run on (see results/relocalization/
# manifest.json, wall_seconds=31602.61). This run does ~2.25x the total
# evaluations (1000+1500+2500+4000 vs 4x1000) AND covers both scenarios, so
# expect a substantially longer wall time. Reduce EVALS_PER_PARAM, or restrict
# to one scenario by editing the --scenario flag below, if that is
# prohibitive; progress can be monitored via the per-ansatz log files.
#
# Usage: ./run_fair_budget_sweep.sh [OUTPUT_DIR] [JOBS] [EVALS_PER_PARAM] [N_RESTARTS]
set -euo pipefail

OUT="${1:-results_fair_budget}"
JOBS="${2:-$(nproc)}"
EVALS_PER_PARAM="${3:-500}"
N_RESTARTS="${4:-16}"
mkdir -p "$OUT/logs"

PY="python -m relocalization.cli"
MERGE_PY="python -m relocalization.merge_fair_budget_sweep"

# ansatz:free-parameter-count, ordered cheapest first for early feedback
ANSATZE=(single_sinusoid:2 chirp:3 two_tone:5 bang_bang:8)

CMD=""
for entry in "${ANSATZE[@]}"; do
  ansatz="${entry%%:*}"
  dim="${entry##*:}"
  max_evals=$(( EVALS_PER_PARAM * dim ))
  adir="$OUT/by_ansatz/$ansatz"
  mkdir -p "$adir"
  log="$OUT/logs/$ansatz.log"
  step="{ $PY sweep --scenario both --profile production --ansatze $ansatz --max-evals $max_evals --n-restarts $N_RESTARTS --output $adir --jobs $JOBS; } 2>&1 | tee $log; ec=\${PIPESTATUS[0]}; echo EXIT_CODE_${ansatz}=\$ec | tee -a $log; test \$ec -eq 0"
  if [ -z "$CMD" ]; then
    CMD="$step"
  else
    CMD="$CMD && $step"
  fi
  echo "Queued $ansatz: dim=$dim max_evals=$max_evals n_restarts=$N_RESTARTS -> $adir"
done

merge_log="$OUT/logs/merge.log"
CMD="$CMD && { $MERGE_PY $OUT --evals-per-param $EVALS_PER_PARAM; } 2>&1 | tee $merge_log"

echo "Launching fair-budget sweep (4 ansatze, sequential, evals/param=$EVALS_PER_PARAM, n_restarts=$N_RESTARTS) into $OUT (jobs=$JOBS)"

tmux new-session -d -s fair-budget-sweep "bash -c '$CMD'"

echo "Started tmux session: fair-budget-sweep"
echo "Attach with:  tmux attach -t fair-budget-sweep"
echo "List with:    tmux ls"
echo "Per-ansatz logs: $OUT/logs/<ansatz>.log   Merge log: $merge_log"
