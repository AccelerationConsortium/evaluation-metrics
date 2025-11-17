#!/usr/bin/env python3
"""
Standalone script to combine parallel Branin campaign results and generate plots.

This script can be used to re-generate plots from partial results without re-running
the entire workflow.

Usage:
    python combine_branin_parallel_results.py <partial_results_dir>
"""

import argparse
import datetime
import json
from pathlib import Path
import sys

# Add the parent directory to path to import from branin_repeat_campaigns
sys.path.insert(0, str(Path(__file__).parent))

from branin_repeat_campaigns import combine_parallel_results


def main():
    parser = argparse.ArgumentParser(
        description="Combine partial Branin campaign results and generate plots"
    )
    parser.add_argument(
        "partial_results_dir",
        type=str,
        help="Path to directory containing partial results from parallel jobs",
    )
    
    args = parser.parse_args()
    
    # Run the combine function
    combine_parallel_results(args.partial_results_dir)


if __name__ == "__main__":
    main()
