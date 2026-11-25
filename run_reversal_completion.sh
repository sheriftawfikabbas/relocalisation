#!/usr/bin/env bash
# Re-run the reversal-paper completion studies in a detached tmux session (CPU by default).
# Usage: ./run_reversal_completion.sh [STUDY ...]    (default: all studies)
# Output: results/relocalization/reversal_completion[_cpu]/ ; log: logs/reversal-completion.log
set -euo pipefail
STUDIES="${*:-bw_lam scaling120 ens_beta site_cross fewbody states lie starts modelfree modelfree_long}"
mkdir -p logs
tmux new-session -d -s reversal-completion \
  "REV_DEVICE=\${REV_DEVICE:-cpu} python -m relocalization.reversal_completion $STUDIES > logs/reversal-completion.log 2>&1; echo EXIT_CODE=\$? >> logs/reversal-completion.log"
echo "Started tmux session reversal-completion (studies: $STUDIES); tail -f logs/reversal-completion.log"
