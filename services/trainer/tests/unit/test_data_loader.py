# ============================================================
# DOC AI Trainer – Unit Tests for Data Loader (Corrected)
# ============================================================

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from transformers import PreTrainedTokenizerBase

from src.config import TrainerConfig
from src.data_loader import DOCDataset, create_dataloader, load_tokenizer_from_minio


class TestDOCDataset(unittest.TestCase):
    """Test the DOCDataset class."""

    @patch("src.data_loader.load_dataset")
    @patch("src.data_loader.tempfile.NamedTemporaryFile")
    @patch("src.data_loader.MinIOClient")
    def test_dataset_iteration(self, mock_minio_client, mock_tempfile, mock_load_dataset):
        # Mock the temporary file
        mock_tempfile.return_value.name = "/tmp/test_dataset.parquet"

        # Mock the MinIO client
        mock_client = MagicMock()
        mock_minio_client.return_value = mock_client

        # Mock the loaded dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 3
        mock_dataset.__iter__.return_value = [
            {"cleaned_text": "Test document one."},
            {"cleaned_text": "Test document two."},
            {"cleaned_text": "Test document three."},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Create a mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }

        # Create a minimal config
        config = TrainerConfig()
        config.data.dataset_path = "final/test_dataset.parquet"

        # Instantiate DOCDataset
        dataset = DOCDataset(config, mock_tokenizer, mock_client)

        # Verify download was called
        mock_client.download_file.assert_called_once_with(
            "final/test_dataset.parquet",
            Path("/tmp/test_dataset.parquet")
        )


class TestDataLoader(unittest.TestCase):
    """Test the create_dataloader function."""

    @patch("src.data_loader.DOCDataset")
    def test_dataloader_creation(self, mock_doc_dataset):
        # Mock dataset
        mock_dataset = MagicMock()
        mock_doc_dataset.return_value = mock_dataset

        config = TrainerConfig()
        config.training.per_device_train_batch_size = 4

        # Create a mock tokenizer and client
        tokenizer = MagicMock()
        client = MagicMock()

        dataloader = create_dataloader(config, tokenizer, client)

        # Verify DataLoader was created with correct batch size
        self.assertIsNotNone(dataloader)
        self.assertEqual(dataloader.batch_size, 4)


class TestTokenizerLoading(unittest.TestCase):
    """Test the load_tokenizer_from_minio function."""

    @patch("transformers.PreTrainedTokenizerFast.from_pretrained")
    @patch("src.data_loader.tempfile.TemporaryDirectory")
    @patch("src.data_loader.MinIOClient")
    def test_tokenizer_loading(self, mock_minio_client, mock_tempdir, mock_from_pretrained):
        # Mock temporary directory
        mock_tempdir.return_value.__enter__.return_value = "/tmp/tokenizer_dir"

        # Mock MinIO client
        mock_client = MagicMock()
        mock_minio_client.return_value = mock_client

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer

        config = TrainerConfig()
        config.data.tokenizer_path = "tokenizer/v1/tokenizer.json"

        tokenizer = load_tokenizer_from_minio(config, mock_client)

        # Verify download was called
        mock_client.download_file.assert_called_once_with(
            "tokenizer/v1/tokenizer.json",
            Path("/tmp/tokenizer_dir/tokenizer.json")
        )

        # Verify from_pretrained was called with the local dir
        mock_from_pretrained.assert_called_once_with(
            "/tmp/tokenizer_dir",
            local_files_only=True,
            trust_remote_code=False,
        )

        # Verify tokenizer is returned
        self.assertEqual(tokenizer, mock_tokenizer)


if __name__ == "__main__":
    unittest.main()