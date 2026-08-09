#!/usr/bin/env python3
# ============================================================
# DOC AI DOCA Service – Coordinator Runner Script
# ============================================================

"""
Simple entrypoint script to run the DOCA Coordinator.

Usage:
    python run_coordinator.py
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.coordinator.main import start_coordinator

if __name__ == "__main__":
    import asyncio
    asyncio.run(start_coordinator())