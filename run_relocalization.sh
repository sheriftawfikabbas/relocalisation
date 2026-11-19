#!/usr/bin/env bash
# Launch the production re-localization sweep in a detached tmux session.
#
# Runs BOTH scenarios in a single process (--scenario both) so the aggregate
# beta_star.csv and manifest.json are written exactly once. Two separate
# sessions writing to the same output dir would clobber those aggregate files.
# Within each scenario the (beta x ansatz) grid is parallelised across JOBS.
#
# Usage: ./run_relocalization.sh [OUTPUT_DIR] [JOBS]
set -euo pipefail

OUT="${1:-results/relocalization}"
JOBS="${2:-$(nproc)}"
mkdir -p "$OUT/logs"

PY="python -m relocalization.cli"

echo "Launching production sweep (scenarios A+B) into $OUT (jobs=$JOBS)"

tmux new-session -d -s reloc-prod \
  "$PY sweep --scenario both --profile production --output $OUT --jobs $JOBS 2>&1 | tee $OUT/logs/reloc-prod.log; echo EXIT_CODE=\${PIPESTATUS[0]} | tee -a $OUT/logs/reloc-prod.log"

echo "Started tmux session: reloc-prod"
echo "Attach with:  tmux attach -t reloc-prod"
echo "List with:    tmux ls"
