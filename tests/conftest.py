"""
Pytest configuration and shared fixtures for EntityLinker tests.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any


# ============================================================
# TEST DATA FIXTURES
# ============================================================

@pytest.fixture
def sample_entities() -> List[Dict[str, Any]]:
    """Sample entities for testing database operations."""
    return [
        {
            "id": "MESH:D000069",
            "entity_id": "MESH:D000069",
            "name": "TP53",
            "label": "TP53",
            "description": "Tumor protein p53, a tumor suppressor gene",
            "entity_type": "gene",
            "popularity": 1000000,
            "aliases": ["p53", "tumor protein 53", "TRP53"]
        },
        {
            "id": "MESH:D000070",
            "entity_id": "MESH:D000070",
            "name": "BRCA1",
            "label": "BRCA1",
            "description": "Breast cancer type 1 susceptibility protein",
            "entity_type": "gene",
            "popularity": 800000,
            "aliases": ["BRCA-1", "breast cancer 1"]
        },
        {
            "id": "MESH:D000071",
            "entity_id": "MESH:D000071",
            "name": "EGFR",
            "label": "EGFR",
            "description": "Epidermal growth factor receptor",
            "entity_type": "gene",
            "popularity": 600000,
            "aliases": ["ErbB-1", "HER1"]
        },
        {
            "id": "MESH:D001943",
            "entity_id": "MESH:D001943",
            "name": "Breast Cancer",
            "label": "Breast Cancer",
            "description": "Malignant neoplasm of the breast",
            "entity_type": "disease",
            "popularity": 900000,
            "aliases": ["breast carcinoma", "mammary cancer"]
        },
        {
            "id": "MESH:D002289",
            "entity_id": "MESH:D002289",
            "name": "Lung Cancer",
            "label": "Lung Cancer",
            "description": "Malignant neoplasm of the lung",
            "entity_type": "disease",
            "popularity": 850000,
            "aliases": ["lung carcinoma", "pulmonary cancer"]
        },
    ]


@pytest.fixture
def sample_texts() -> List[str]:
    """Sample texts for testing NER and linking."""
    return [
        "TP53 mutations are associated with breast cancer.",
        "BRCA1 is a tumor suppressor gene linked to hereditary breast cancer.",
        "EGFR inhibitors are used in lung cancer treatment.",
    ]


@pytest.fixture
def single_text() -> str:
    """Single text for simple tests."""
    return "TP53 mutations cause breast cancer."


@pytest.fixture
def entities_jsonl_file(sample_entities, tmp_path) -> str:
    """Create temporary JSONL file with entities."""
    filepath = tmp_path / "entities.jsonl"
    with open(filepath, 'w') as f:
        for entity in sample_entities:
            f.write(json.dumps(entity) + '\n')
    return str(filepath)


@pytest.fixture
def schema_template() -> str:
    """Standard label template."""
    return "{label}: {description}"


# ============================================================
# L1 FIXTURES
# ============================================================

@pytest.fixture
def l1_config_dict() -> Dict[str, Any]:
    """L1 processor configuration dictionary."""
    return {
        "model": "en_core_sci_sm",
        "device": "cpu",
        "batch_size": 1,
        "max_left_context": 50,
        "max_right_context": 50,
        "min_entity_length": 2,
        "include_noun_chunks": True
    }


@pytest.fixture
def l1_config(l1_config_dict):
    """L1Config instance."""
    from glinker.l1.models import L1Config
    return L1Config(**l1_config_dict)


@pytest.fixture
def l1_component(l1_config):
    """L1Component instance."""
    from glinker.l1.component import L1Component
    return L1Component(l1_config)


# ============================================================
# L2 FIXTURES
# ============================================================

@pytest.fixture
def l2_layer_config_dict() -> Dict[str, Any]:
    """Single Dict layer configuration."""
    return {
        "type": "dict",
        "priority": 0,
        "write": True,
        "search_mode": ["exact", "fuzzy"],
        "ttl": 0,
        "cache_policy": "always",
        "field_mapping": {
            "entity_id": "entity_id",
            "label": "label",
            "aliases": "aliases",
            "description": "description",
            "entity_type": "entity_type",
            "popularity": "popularity"
        },
        "fuzzy": {
            "max_distance": 64,
            "min_similarity": 0.6,
            "n_gram_size": 3,
            "prefix_length": 1
        }
    }


@pytest.fixture
def l2_config_dict(l2_layer_config_dict) -> Dict[str, Any]:
    """L2 processor configuration dictionary."""
    return {
        "max_candidates": 5,
        "min_popularity": 0,
        "embeddings": {
            "enabled": True,
            "model_name": "BioMike/gliner-deberta-base-v1-post",
            "dim": 768
        },
        "layers": [l2_layer_config_dict]
    }


@pytest.fixture
def l2_config(l2_config_dict):
    """L2Config instance."""
    from glinker.l2.models import L2Config
    return L2Config(**l2_config_dict)


@pytest.fixture
def l2_component(l2_config):
    """L2 DatabaseChainComponent instance."""
    from glinker.l2.component import DatabaseChainComponent
    return DatabaseChainComponent(l2_config)


@pytest.fixture
def dict_layer(l2_layer_config_dict):
    """DictLayer instance."""
    from glinker.l2.component import DictLayer
    from glinker.l2.models import LayerConfig
    config = LayerConfig(**l2_layer_config_dict)
    return DictLayer(config)


@pytest.fixture
def loaded_dict_layer(dict_layer, sample_entities):
    """DictLayer with loaded entities."""
    from glinker.l2.models import DatabaseRecord
    records = [
        DatabaseRecord(
            entity_id=e["id"],
            label=e["name"],
            description=e["description"],
            entity_type=e["entity_type"],
            popularity=e["popularity"],
            aliases=e["aliases"]
        )
        for e in sample_entities
    ]
    dict_layer.load_bulk(records)
    return dict_layer


# ============================================================
# L3 FIXTURES
# ============================================================

@pytest.fixture
def l3_config_dict() -> Dict[str, Any]:
    """L3 processor configuration dictionary."""
    return {
        "model_name": "BioMike/gliner-deberta-base-v1-post",
        "huggingface_token": "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",
        "device": "cpu",
        "threshold": 0.3,
        "flat_ner": True,
        "multi_label": False,
        "batch_size": 1,
        "use_precomputed_embeddings": False,
        "cache_embeddings": False,
        "max_length": 512
    }


@pytest.fixture
def l3_config(l3_config_dict):
    """L3Config instance."""
    from glinker.l3.models import L3Config
    return L3Config(**l3_config_dict)


# L3 component is expensive - session scoped
@pytest.fixture(scope="session")
def l3_component():
    """L3Component instance (session-scoped for efficiency)."""
    from glinker.l3.component import L3Component
    from glinker.l3.models import L3Config
    config = L3Config(
        model_name="BioMike/gliner-deberta-base-v1-post",
        token="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",
        device="cpu",
        threshold=0.3,
        flat_ner=True,
        multi_label=False,
        max_length=512
    )
    return L3Component(config)


# ============================================================
# L0 FIXTURES
# ============================================================

@pytest.fixture
def l0_config_dict() -> Dict[str, Any]:
    """L0 processor configuration dictionary."""
    return {
        "min_confidence": 0.0,
        "include_unlinked": True,
        "return_all_candidates": True,
        "strict_matching": True,
        "position_tolerance": 2
    }


@pytest.fixture
def l0_config(l0_config_dict):
    """L0Config instance."""
    from glinker.l0.models import L0Config
    return L0Config(**l0_config_dict)


@pytest.fixture
def l0_component(l0_config):
    """L0Component instance."""
    from glinker.l0.component import L0Component
    return L0Component(l0_config)


# ============================================================
# PIPELINE FIXTURES
# ============================================================

@pytest.fixture
def pipeline_config_dict() -> Dict[str, Any]:
    """Full pipeline configuration dictionary."""
    return {
        "name": "test_pipeline",
        "description": "Test pipeline for unit tests",
        "nodes": [
            {
                "id": "l1",
                "processor": "l1_batch",
                "inputs": {"texts": {"source": "$input", "fields": "texts"}},
                "output": {"key": "l1_result"},
                "config": {
                    "model": "en_core_sci_sm",
                    "device": "cpu",
                    "batch_size": 1,
                    "min_entity_length": 2
                }
            },
            {
                "id": "l2",
                "processor": "l2_chain",
                "inputs": {"mentions": {"source": "l1_result", "fields": "entities"}},
                "output": {"key": "l2_result"},
                "schema": {"template": "{label}: {description}"},
                "config": {
                    "max_candidates": 5,
                    "layers": [{
                        "type": "dict",
                        "priority": 0,
                        "search_mode": ["exact", "fuzzy"],
                        "write": True,
                        "field_mapping": {
                            "entity_id": "entity_id",
                            "label": "label",
                            "aliases": "aliases",
                            "description": "description",
                            "entity_type": "entity_type",
                            "popularity": "popularity"
                        },
                        "fuzzy": {"min_similarity": 0.6}
                    }]
                }
            },
            {
                "id": "l3",
                "processor": "l3_batch",
                "inputs": {
                    "texts": {"source": "$input", "fields": "texts"},
                    "candidates": {"source": "l2_result", "fields": "candidates"}
                },
                "output": {"key": "l3_result"},
                "schema": {"template": "{label}: {description}"},
                "config": {
                    "model_name": "BioMike/gliner-deberta-base-v1-post",
                    "huggingface_token": "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",
                    "device": "cpu",
                    "threshold": 0.3,
                    "max_length": 512
                }
            },
            {
                "id": "l0",
                "processor": "l0_aggregator",
                "requires": ["l1", "l2", "l3"],
                "inputs": {
                    "l1_entities": {"source": "l1_result", "fields": "entities"},
                    "l2_candidates": {"source": "l2_result", "fields": "candidates"},
                    "l3_entities": {"source": "l3_result", "fields": "entities"}
                },
                "output": {"key": "l0_result"},
                "schema": {"template": "{label}: {description}"},
                "config": {
                    "min_confidence": 0.0,
                    "include_unlinked": True,
                    "strict_matching": True,
                    "position_tolerance": 2
                }
            }
        ]
    }


@pytest.fixture(scope="session")
def executor():
    """DAGExecutor instance (session-scoped)."""
    from glinker.core.factory import ProcessorFactory

    config = {
        "name": "test_pipeline",
        "nodes": [
            {
                "id": "l1",
                "processor": "l1_batch",
                "inputs": {"texts": {"source": "$input", "fields": "texts"}},
                "output": {"key": "l1_result"},
                "config": {"model": "en_core_sci_sm", "device": "cpu", "batch_size": 1}
            },
            {
                "id": "l2",
                "processor": "l2_chain",
                "inputs": {"mentions": {"source": "l1_result", "fields": "entities"}},
                "output": {"key": "l2_result"},
                "schema": {"template": "{label}: {description}"},
                "config": {
                    "max_candidates": 5,
                    "layers": [{
                        "type": "dict", "priority": 0, "search_mode": ["exact", "fuzzy"],
                        "write": True,
                        "field_mapping": {
                            "entity_id": "entity_id", "label": "label",
                            "aliases": "aliases", "description": "description",
                            "entity_type": "entity_type", "popularity": "popularity"
                        },
                        "fuzzy": {"min_similarity": 0.6}
                    }]
                }
            },
            {
                "id": "l3",
                "processor": "l3_batch",
                "inputs": {
                    "texts": {"source": "$input", "fields": "texts"},
                    "candidates": {"source": "l2_result", "fields": "candidates"}
                },
                "output": {"key": "l3_result"},
                "schema": {"template": "{label}: {description}"},
                "config": {
                    "model_name": "BioMike/gliner-deberta-base-v1-post",
                    "huggingface_token": "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",
                    "device": "cpu", "threshold": 0.3, "max_length": 512
                }
            },
            {
                "id": "l0",
                "processor": "l0_aggregator",
                "requires": ["l1", "l2", "l3"],
                "inputs": {
                    "l1_entities": {"source": "l1_result", "fields": "entities"},
                    "l2_candidates": {"source": "l2_result", "fields": "candidates"},
                    "l3_entities": {"source": "l3_result", "fields": "entities"}
                },
                "output": {"key": "l0_result"},
                "schema": {"template": "{label}: {description}"},
                "config": {"min_confidence": 0.0, "include_unlinked": True}
            }
        ]
    }

    return ProcessorFactory.create_from_dict(config, verbose=False)


@pytest.fixture
def loaded_executor(executor, entities_jsonl_file):
    """Executor with loaded entities."""
    executor.load_entities(entities_jsonl_file, target_layers=["dict"])
    return executor
