#!/usr/bin/env python3
# ============================================================
# DOC AI Data Pipeline – Run Script
# ============================================================

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow.pipeline import data_pipeline


def main():
    """Run the data pipeline with the given configuration."""
    # Get config path from environment variable or command line
    config_path = os.environ.get("DOC_CONFIG_PATH")
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not config_path:
        print("ERROR: No configuration file specified.", file=sys.stderr)
        print("Usage: python scripts/run_pipeline.py <config_path>", file=sys.stderr)
        print("Or set DOC_CONFIG_PATH environment variable.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting data pipeline with config: {config_path}")
    data_pipeline(config_path)


if __name__ == "__main__":
    main()