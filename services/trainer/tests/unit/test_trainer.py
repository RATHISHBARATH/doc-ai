# ============================================================
# DOC AI Trainer – Unit Tests for Trainer Core (Final)
# ============================================================

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.trainer import Trainer
from src.config import TrainerConfig


class TestTrainer(unittest.TestCase):
    """Test the Trainer class."""

    def setUp(self):
        # Create a minimal config
        self.config = TrainerConfig()
        self.config.training.num_epochs = 1
        self.config.training.learning_rate = 5e-4
        self.config.training.per_device_train_batch_size = 2
        self.config.training.logging_steps = 1   # Log every step for testing

        # Create a real parameter to avoid optimizer empty parameter list
        self.param = torch.nn.Parameter(torch.tensor([1.0]))
        self.param.grad = None

        # Mock model that returns the real parameter
        self.mock_model = MagicMock(spec=PreTrainedModel)
        self.mock_model.parameters.return_value = [self.param]

        self.mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

        # Create mock DataLoader
        self.mock_dataloader = MagicMock(spec=DataLoader)
        self.mock_dataloader.__len__.return_value = 5
        self.mock_dataloader.__iter__.return_value = [
            {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
            for _ in range(5)
        ]

        # Mock Accelerator
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.prepare.side_effect = lambda *args: args
        self.mock_accelerator.accumulate.return_value = MagicMock()
        self.mock_accelerator.sync_gradients = True
        self.mock_accelerator.unwrap_model.return_value = self.mock_model

        # Create Trainer instance
        self.trainer = Trainer(
            config=self.config,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            train_dataloader=self.mock_dataloader,
            accelerator=self.mock_accelerator,
        )

    def test_initialization(self):
        """Test that the trainer initializes correctly."""
        self.assertEqual(self.trainer.config, self.config)
        self.assertEqual(self.trainer.model, self.mock_model)
        self.assertEqual(self.trainer.tokenizer, self.mock_tokenizer)
        self.assertEqual(self.trainer.train_dataloader, self.mock_dataloader)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.lr_scheduler)

    @patch("src.trainer.tqdm")
    def test_train_loop(self, mock_tqdm):
        """Test that the training loop executes and returns metrics."""
        # Configure mock model to return a loss
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5)
        self.mock_model.return_value = mock_output

        # Patch both optimizer and scheduler step methods
        with patch.object(self.trainer.optimizer, 'step') as mock_opt_step, \
             patch.object(self.trainer.lr_scheduler, 'step') as mock_sched_step:

            # Run training
            metrics = self.trainer.train()

            # Verify both steps were called
            mock_opt_step.assert_called()
            mock_sched_step.assert_called()

        # Verify loss computation and model forward pass
        self.mock_model.assert_called()

        # Verify metrics
        self.assertIn("loss", metrics)
        self.assertIn("steps", metrics)
        self.assertGreater(metrics["steps"], 0)

    def test_accelerator_integration(self):
        """Test that accelerator is used correctly."""
        # Verify prepare was called with model, optimizer, dataloader, scheduler
        self.mock_accelerator.prepare.assert_called_with(
            self.mock_model,
            self.trainer.optimizer,
            self.mock_dataloader,
            self.trainer.lr_scheduler,
        )

    @patch("src.trainer.logger")
    def test_logging(self, mock_logger):
        """Test that logging occurs during training."""
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5)
        self.mock_model.return_value = mock_output

        # Run training (will trigger logging at step intervals)
        self.trainer.train()

        # Check that the log message contains the step number and loss
        # With logging_steps=1, the first log should be at step 1.
        expected_substrings = ["Step 1", "Loss: 0.5000"]
        found = False
        for call in mock_logger.info.call_args_list:
            if all(sub in call[0][0] for sub in expected_substrings):
                found = True
                break
        self.assertTrue(found, "Log message with expected content not found")


if __name__ == "__main__":
    unittest.main()