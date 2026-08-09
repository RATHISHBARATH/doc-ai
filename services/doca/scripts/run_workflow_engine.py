#!/usr/bin/env python3
# ============================================================
# DOC AI DOCA Service – Workflow Engine Runner Script
# ============================================================

"""
Simple entrypoint script to run the Workflow Engine.

Usage:
    python run_workflow_engine.py
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.workflow_engine.scheduler import WorkflowScheduler

async def main():
    scheduler = WorkflowScheduler()
    await scheduler.task_queue.connect()
    print("Workflow Engine started. Waiting for workflows...")
    # Keep running
    import asyncio
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())