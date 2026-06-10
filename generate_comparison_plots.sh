#!/usr/bin/env bash
# Generate comparison plot sets for all configs in plot_configs/.
#
# Prerequisites (run once to populate the caches):
#
#   python -m src.eval.collect_eval_data \
#       --flag <tag> \
#       --classification-out plots/tmp_results/classification_<tag>.pkl \
#       --regression-out    plots/tmp_results/regression.pkl
#
#   python -m src.eval.plot_classification_light   # writes plots/tmp_results/classification_light.pkl
#   python -m src.eval.plot_steps                  # writes plots/tmp_results/steps.pkl
#   python -m src.eval.build_classification_cache  # writes plots/tmp_results/classification_metrics.pkl
#
# After that, re-running this script is fast for any config change.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CACHE_FILE="plots/tmp_results/classification_metrics.pkl"
if [ ! -f "$CACHE_FILE" ]; then
    echo "=== Building classification metrics cache (one-time, ~5–10 min) ==="
    python -m src.eval.build_classification_cache
fi

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

run_all plot_configs/V1Paper.json            plots/V1Paper
run_all plot_configs/hyperscale.json         plots/HYPERSCALE
run_all plot_configs/bert_vs_ol.json         plots/BERTvsOL
run_all plot_configs/20260606_Comparison.json plots/Comparison
run_all plot_configs/default.json            plots/default
