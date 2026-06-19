#!/usr/bin/env bash
# Wrapper — see scripts/generate_comparison_plots.sh for details.
exec bash "$(dirname "${BASH_SOURCE[0]}")/scripts/generate_comparison_plots.sh" "$@"
