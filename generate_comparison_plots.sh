#!/usr/bin/env bash
# Generate comparison plot sets for V1Paper, HYPERSCALE, and BERTvsOL configs.
#
# Prerequisites (run once to populate the caches):
#   python -m src.eval.collect_eval_data \
#       --flag <tag> \
#       --classification-out plots/tmp_results/classification.pkl \
#       --regression-out plots/tmp_results/regression.pkl
#   python -m src.eval.plot_classification_light   # writes plots/tmp_results/classification_light.pkl
#   python -m src.eval.plot_steps                  # writes plots/tmp_results/steps.pkl
#   python -m src.eval.plot_regression             # writes plots/tmp_results/regression.pkl (already done above)
#
# After that, re-running this script is fast for any config change.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

run_all() {
    local config="$1"
    local plots_dir="$2"
    echo
    echo "=== $(basename "$config" .json) -> $plots_dir ==="

    python -m src.eval.plot_steps \
        --plots-only --config "$config" --plots-dir "$plots_dir"

    python -m src.eval.plot_classification_light \
        --plots-only --config "$config" --plots-dir "$plots_dir"

    python -m src.eval.plot_regression \
        --plots-only --config "$config" --plots-dir "$plots_dir"

    python -m src.eval.plot_classification_q3 \
        --plots-only --config "$config" --plots-dir "$plots_dir"

    python -m src.eval.plot_classification_W \
        --plots-only --config "$config" --plots-dir "$plots_dir"

    python -m src.eval.plot_classification_Pions \
        --plots-only --config "$config" --plots-dir "$plots_dir"

    echo "Done -> $plots_dir"
}

run_all plots/configs/V1Paper.json    plots/V1Paper
run_all plots/configs/hyperscale.json plots/HYPERSCALE
run_all plots/configs/bert_vs_ol.json plots/BERTvsOL
