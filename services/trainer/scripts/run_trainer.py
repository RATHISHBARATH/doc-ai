# ============================================================
# DOC AI Trainer – Entrypoint Script
# ============================================================

"""
This script is a simple entrypoint for the trainer service.
It imports and calls the main function from src.trainer.

Usage:
    python run_trainer.py
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trainer import main

if __name__ == "__main__":
    main()