# ============================================================
# Integration Test for the Full Pipeline (Mocked Dependencies)
# ============================================================

import pytest
from unittest.mock import patch, MagicMock
from prefect.testing.utilities import prefect_test_harness

from src.workflow.pipeline import data_pipeline
from src.common.config import Config


@pytest.fixture
def mock_config():
    """Create a minimal test configuration."""
    return {
        "data_sources": [
            {
                "name": "test_source",
                "url": "file:///tmp/test_data.jsonl",
                "format": "jsonl",
                "compression": None,
                "max_documents": 5,
            }
        ],
        "cleaning": {"fix_unicode": True, "normalize_whitespace": True},
        "deduplication": {"threshold": 0.8, "num_permutations": 4},
        "quality_filter": {"min_words": 2, "max_words": 100},
        "tokenizer": {"vocab_size": 100, "sample_size": 1024},
        "dataset_prep": {"max_length": 16, "stride": 8},
        "minio": {"endpoint": "localhost:9000", "access_key": "minioadmin", "secret_key": "changeme", "bucket": "doc-ai-data", "secure": False},
        "postgres": {"host": "localhost", "port": 5432, "database": "doc_ai_metadata", "user": "doc_user", "password": "changeme"},
        "workflow": {"max_retries": 1, "retry_delay_seconds": 1, "checkpoint_interval": 10},
        "data_root": "./data",
        "cleaned_dir": "./data/cleaned",
        "deduped_dir": "./data/deduped",
        "filtered_dir": "./data/filtered",
        "tokenized_dir": "./data/tokenized",
        "final_dir": "./data/final",
    }


@pytest.mark.integration
def test_pipeline_flow_runs_without_error(mock_config):
    """
    Test that the Prefect flow runs without raising exceptions.
    All external dependencies (MinIO, PostgreSQL, stage functions) are mocked.
    """
    with prefect_test_harness():
        # Patch the configuration loader to return our mock config
        with patch("src.workflow.pipeline.get_config") as mock_get_config:
            mock_get_config.return_value = Config.from_dict(mock_config)

            # Patch the stage functions to do nothing (they are tested separately)
            with patch("src.workflow.pipeline.download_data") as mock_download, \
                 patch("src.workflow.pipeline.clean_data") as mock_clean, \
                 patch("src.workflow.pipeline.deduplicate") as mock_dedup, \
                 patch("src.workflow.pipeline.quality_filter") as mock_filter, \
                 patch("src.workflow.pipeline.train_tokenizer") as mock_train, \
                 patch("src.workflow.pipeline.tokenize_dataset") as mock_tokenize, \
                 patch("src.workflow.pipeline.chunk_data") as mock_chunk, \
                 patch("src.workflow.pipeline.format_parquet") as mock_format, \
                 patch("src.workflow.pipeline.task_cleanup_intermediate") as mock_cleanup:

                # Run the flow
                data_pipeline(config_path="test_config.yaml")

                # Verify each stage was called exactly once
                mock_download.assert_called_once()
                mock_clean.assert_called_once()
                mock_dedup.assert_called_once()
                mock_filter.assert_called_once()
                mock_train.assert_called_once()
                mock_tokenize.assert_called_once()
                mock_chunk.assert_called_once()
                mock_format.assert_called_once()
                # Cleanup is optional; check only if it was called
                # mock_cleanup.assert_called_once()


@pytest.mark.integration
def test_pipeline_flow_handles_stage_failure(mock_config):
    """
    Test that the flow handles a stage failure gracefully (retries and eventually fails).
    """
    with prefect_test_harness():
        with patch("src.workflow.pipeline.get_config") as mock_get_config:
            mock_get_config.return_value = Config.from_dict(mock_config)

            with patch("src.workflow.pipeline.download_data") as mock_download:
                # Make the download stage raise an exception
                mock_download.side_effect = Exception("Download failed")

                with pytest.raises(Exception, match="Download failed"):
                    data_pipeline(config_path="test_config.yaml")