# ============================================================
# DOC AI DOCA Service – Task Queue (NATS JetStream)
# ============================================================

import asyncio
import json
import logging
from typing import Optional, Callable, Awaitable, Dict, Any

import nats
from nats.aio.client import Client as NATSClient
from nats.aio.msg import Msg

from src.common.models import Task, TaskStatus
from src.common.config import DOCAConfig, get_config

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Task queue abstraction using NATS JetStream.

    Provides durable, reliable task queues for the Workflow Engine.
    Tasks are serialized as JSON and published to a JetStream stream.
    Workers (agents) can subscribe to the stream and process tasks.
    """

    def __init__(self, config: Optional[DOCAConfig] = None):
        self.config = config or get_config()
        self.nc: Optional[NATSClient] = None
        self.js = None
        self.stream_name = "DOCA_TASKS"
        self.subject = "doca.tasks"
        self._is_connected = False

    async def connect(self) -> None:
        """Establish connection to NATS and set up JetStream."""
        if self._is_connected:
            return

        nats_url = self.config.nats_url if hasattr(self.config, 'nats_url') else "nats://nats:4222"
        try:
            self.nc = await nats.connect(nats_url)
            self.js = self.nc.jetstream()
            # Ensure the stream exists
            await self.js.add_stream(
                name=self.stream_name,
                subjects=[self.subject],
                max_age=24 * 60 * 60,  # 24 hours retention
                storage=nats.constants.STORAGE_FILE,
            )
            self._is_connected = True
            logger.info(f"Connected to NATS JetStream at {nats_url}")
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            self._is_connected = False
            raise

    async def enqueue(self, task: Task) -> bool:
        """
        Publish a task to the queue.

        Args:
            task: The Task object to enqueue.

        Returns:
            True if the task was published successfully.
        """
        if not self._is_connected:
            await self.connect()

        try:
            # Convert task to JSON-serializable dict
            task_data = {
                "task_id": task.task_id,
                "workflow_id": task.workflow_id,
                "agent_type": task.agent_type.value,
                "input_data": task.input_data,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "retries": task.retries,
                "max_retries": task.max_retries,
                "metadata": task.metadata,
            }
            payload = json.dumps(task_data).encode("utf-8")
            await self.js.publish(self.subject, payload)
            logger.debug(f"Task {task.task_id} enqueued")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False

    async def dequeue(self) -> Optional[Task]:
        """
        Pull a task from the queue (non-blocking).

        Returns:
            A Task object if one is available, else None.
        """
        if not self._is_connected:
            await self.connect()

        try:
            # Pull next message with a short timeout (or use fetch)
            msg: Msg = await self.js.pull_subscribe(
                subject=self.subject,
                durable="doca_worker",
            ).fetch(1, timeout=1.0)
            if msg:
                task_data = json.loads(msg.data.decode("utf-8"))
                # Reconstruct Task object
                task = Task(
                    task_id=task_data["task_id"],
                    workflow_id=task_data["workflow_id"],
                    agent_type=task_data["agent_type"],
                    input_data=task_data["input_data"],
                    status=TaskStatus(task_data["status"]),
                    retries=task_data["retries"],
                    max_retries=task_data["max_retries"],
                    metadata=task_data["metadata"],
                )
                # Acknowledge message (remove from queue)
                await msg.ack()
                logger.debug(f"Task {task.task_id} dequeued")
                return task
            else:
                return None
        except nats.errors.NATSNoRespondersError:
            # No messages available
            return None
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None

    async def subscribe(self, callback: Callable[[Task], Awaitable[None]]) -> None:
        """
        Subscribe to the task queue and process tasks with a callback.

        This is a continuous pull-based consumer. The callback will be invoked
        for each task received.

        Args:
            callback: An async function that takes a Task and processes it.
        """
        if not self._is_connected:
            await self.connect()

        sub = await self.js.pull_subscribe(
            subject=self.subject,
            durable="doca_worker",
        )

        logger.info("Subscribed to task queue. Waiting for tasks...")

        while True:
            try:
                msg = await sub.fetch(1, timeout=5.0)
                if msg:
                    task_data = json.loads(msg.data.decode("utf-8"))
                    task = Task(
                        task_id=task_data["task_id"],
                        workflow_id=task_data["workflow_id"],
                        agent_type=task_data["agent_type"],
                        input_data=task_data["input_data"],
                        status=TaskStatus(task_data["status"]),
                        retries=task_data["retries"],
                        max_retries=task_data["max_retries"],
                        metadata=task_data["metadata"],
                    )
                    await callback(task)
                    await msg.ack()
            except nats.errors.NATSNoRespondersError:
                # No messages, wait a bit and continue
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in task subscription: {e}")
                await asyncio.sleep(1)

    async def close(self) -> None:
        """Close the NATS connection."""
        if self.nc:
            await self.nc.close()
            self._is_connected = False
            logger.info("NATS connection closed")