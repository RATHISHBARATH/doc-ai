#!/usr/bin/env python3
# ============================================================
# DOC AI Data Pipeline – Intermediate Cleanup Script
# ============================================================

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.minio_client import MinIOClient
from src.common.config import get_config


def main():
    """Delete intermediate data from MinIO."""
    config_path = os.environ.get("DOC_CONFIG_PATH")
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not config_path:
        print("ERROR: No configuration file specified.", file=sys.stderr)
        sys.exit(1)

    config = get_config(config_path)
    client = MinIOClient(config)

    prefixes = ["cleaned/", "deduped/", "filtered/", "tokenized/"]
    for prefix in prefixes:
        try:
            count = client.delete_prefix(prefix)
            print(f"Deleted {count} objects under {prefix}")
        except Exception as e:
            print(f"Error deleting prefix {prefix}: {e}", file=sys.stderr)

    print("Cleanup completed.")


if __name__ == "__main__":
    main()