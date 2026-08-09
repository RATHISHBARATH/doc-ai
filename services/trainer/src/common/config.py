# ============================================================
# DOC AI Data Pipeline – Configuration Module (Corrected)
# ============================================================

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""
    name: str
    url: str
    format: str
    compression: Optional[str] = None
    max_documents: Optional[int] = None


@dataclass
class CleaningConfig:
    fix_unicode: bool = True
    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    filter_language: Optional[str] = 'en'
    scrub_pii: bool = True


@dataclass
class DeduplicationConfig:
    threshold: float = 0.8
    num_permutations: int = 128
    batch_size: int = 10000


@dataclass
class QualityFilterConfig:
    min_words: int = 50
    max_words: int = 10000
    max_punctuation_ratio: float = 0.3
    min_stop_word_ratio: float = 0.05
    scorer_model_path: Optional[str] = None


@dataclass
class TokenizerConfig:
    vocab_size: int = 100_000
    special_tokens: List[str] = field(default_factory=list)
    min_frequency: int = 2
    sample_size: int = 5_000_000_000


@dataclass
class DatasetPrepConfig:
    max_length: int = 512
    stride: int = 256
    output_format: str = 'parquet'


@dataclass
class MinIOConfig:
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "changeme"
    bucket: str = "doc-ai-data"
    secure: bool = True


@dataclass
class PostgreSQLConfig:
    host: str
    database: str
    user: str
    password: str
    port: int = 5432


@dataclass
class WorkflowConfig:
    max_retries: int = 3
    retry_delay_seconds: int = 5
    checkpoint_interval: int = 1000


@dataclass
class Config:
    data_sources: List[DataSourceConfig]
    cleaning: CleaningConfig
    deduplication: DeduplicationConfig
    quality_filter: QualityFilterConfig
    tokenizer: TokenizerConfig
    dataset_prep: DatasetPrepConfig
    minio: MinIOConfig
    postgres: PostgreSQLConfig
    workflow: WorkflowConfig
    data_root: Path = Path("./data")
    cleaned_dir: Path = Path("./data/cleaned")
    deduped_dir: Path = Path("./data/deduped")
    filtered_dir: Path = Path("./data/filtered")
    tokenized_dir: Path = Path("./data/tokenized")
    final_dir: Path = Path("./data/final")

    def __post_init__(self):
        if not self.postgres.host:
            raise ValueError("postgres.host must be provided")
        if not self.postgres.database:
            raise ValueError("postgres.database must be provided")
        if not self.postgres.user:
            raise ValueError("postgres.user must be provided")
        if not self.postgres.password:
            raise ValueError("postgres.password must be provided")
        for field_name in ['data_root', 'cleaned_dir', 'deduped_dir',
                           'filtered_dir', 'tokenized_dir', 'final_dir']:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        data = cls._apply_env_overrides(data)
        config = cls(
            data_sources=[DataSourceConfig(**src) for src in data.get('data_sources', [])],
            cleaning=CleaningConfig(**data.get('cleaning', {})),
            deduplication=DeduplicationConfig(**data.get('deduplication', {})),
            quality_filter=QualityFilterConfig(**data.get('quality_filter', {})),
            tokenizer=TokenizerConfig(**data.get('tokenizer', {})),
            dataset_prep=DatasetPrepConfig(**data.get('dataset_prep', {})),
            minio=MinIOConfig(**data.get('minio', {})),
            postgres=PostgreSQLConfig(**data.get('postgres', {})),
            workflow=WorkflowConfig(**data.get('workflow', {})),
            data_root=data.get('data_root', './data'),
            cleaned_dir=data.get('cleaned_dir', './data/cleaned'),
            deduped_dir=data.get('deduped_dir', './data/deduped'),
            filtered_dir=data.get('filtered_dir', './data/filtered'),
            tokenized_dir=data.get('tokenized_dir', './data/tokenized'),
            final_dir=data.get('final_dir', './data/final'),
        )
        return config

    @staticmethod
    def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override configuration with environment variables (DOC_*).
        Use double underscore (__) to denote nesting.
        Example: DOC_MINIO__SECRET_KEY sets minio.secret_key.
        """
        def _convert(value: str, default: Any) -> Any:
            if default is None:
                return value
            if isinstance(default, bool):
                return value.lower() in ('true', '1', 'yes', 'on')
            if isinstance(default, int):
                try:
                    return int(value)
                except ValueError:
                    return default
            if isinstance(default, float):
                try:
                    return float(value)
                except ValueError:
                    return default
            return value

        env_vars = {k: v for k, v in os.environ.items() if k.startswith("DOC_")}
        for key, value in env_vars.items():
            # Remove prefix and split by '__' for nesting
            parts = key[4:].lower().split('__')
            # If no '__', treat the whole key as a single field (but we'll still use it)
            current = data
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Leaf: set the value
                    current[part] = _convert(value, current.get(part))
                else:
                    # Ensure the nested dict exists
                    if part not in current or not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]
        return data

    def to_yaml(self, path: Union[str, Path]) -> None:
        data = {
            'data_sources': [vars(src) for src in self.data_sources],
            'cleaning': vars(self.cleaning),
            'deduplication': vars(self.deduplication),
            'quality_filter': vars(self.quality_filter),
            'tokenizer': vars(self.tokenizer),
            'dataset_prep': vars(self.dataset_prep),
            'minio': vars(self.minio),
            'postgres': vars(self.postgres),
            'workflow': vars(self.workflow),
            'data_root': str(self.data_root),
            'cleaned_dir': str(self.cleaned_dir),
            'deduped_dir': str(self.deduped_dir),
            'filtered_dir': str(self.filtered_dir),
            'tokenized_dir': str(self.tokenized_dir),
            'final_dir': str(self.final_dir),
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


# Singleton
_config: Optional[Config] = None
_config_path: Optional[str] = None

def load_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        config_path = os.environ.get("DOC_CONFIG_PATH", "configs/development.yaml")
    return Config.from_yaml(config_path)

def get_config(config_path: Optional[str] = None) -> Config:
    global _config, _config_path
    if _config is None or (config_path is not None and config_path != _config_path):
        _config = load_config(config_path)
        _config_path = config_path
    return _config

def reset_config():
    global _config, _config_path
    _config = None
    _config_path = None