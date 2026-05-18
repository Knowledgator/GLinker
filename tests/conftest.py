"""
Pytest configuration and shared fixtures for GLinker tests.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch
import torch


# ============================================================
# GLOBAL MOCKS FOR CI/CD
# ============================================================

@pytest.fixture(scope="session", autouse=True)
def mock_spacy_models():
    """Mock spacy.load globally to avoid downloading models in CI."""
    with patch("spacy.load") as mock_load:
        mock_nlp = MagicMock()

        def mock_call(text):
            mock_doc = MagicMock()
            entities = []
            # Simple mock: find common keywords
            keywords = {
                "TP53": ("GENE", 0, 4),
                "BRCA1": ("GENE", 0, 5),
                "cancer": ("DISEASE", 0, 6),
            }
            for keyword, (label, _, _) in keywords.items():
                if keyword in text:
                    start = text.find(keyword)
                    mock_ent = MagicMock()
                    mock_ent.text = keyword
                    mock_ent.label_ = label
                    mock_ent.start_char = start
                    mock_ent.end_char = start + len(keyword)
                    entities.append(mock_ent)

            mock_doc.ents = entities
            mock_doc.noun_chunks = []
            mock_doc.text = text
            return mock_doc

        mock_nlp.side_effect = mock_call
        mock_load.return_value = mock_nlp
        yield mock_load


@pytest.fixture(scope="session", autouse=True)
def mock_gliner_models():
    """Mock GLiNER.from_pretrained globally to avoid downloading models in CI."""
    with patch("gliner.GLiNER.from_pretrained") as mock_from_pretrained:
        mock_model = MagicMock()

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 512
        mock_data_processor = MagicMock()
        mock_data_processor.labels_tokenizer = mock_tokenizer
        mock_model.data_processor = mock_data_processor

        # Mock config for BiEncoder
        mock_config = MagicMock()
        mock_config.labels_encoder = MagicMock()
        mock_model.config = mock_config

        # Mock predict_entities
        def mock_predict(text, labels, **kwargs):
            results = []
            keywords = {
                "gene": ["TP53", "BRCA1", "EGFR"],
                "disease": ["cancer", "carcinoma"],
                "protein": ["protein", "p53"],
            }
            for label in labels:
                for keyword in keywords.get(label, []):
                    if keyword.lower() in text.lower():
                        start = text.lower().find(keyword.lower())
                        results.append({
                            "text": text[start : start + len(keyword)],
                            "label": label,
                            "start": start,
                            "end": start + len(keyword),
                            "score": 0.9,
                            "class_probs": {label: 0.9},
                        })
            return results

        mock_model.predict_entities.side_effect = mock_predict
        mock_model.to.return_value = mock_model

        # Mock encode_labels
        def mock_encode_labels(labels, batch_size=32):
            return torch.randn(len(labels), 768)

        mock_model.encode_labels.side_effect = mock_encode_labels

        mock_from_pretrained.return_value = mock_model
        yield mock_from_pretrained


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
    """L1SpacyComponent instance (uses global spacy mock)."""
    from glinker.l1.component import L1SpacyComponent

    return L1SpacyComponent(l1_config)


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
            "model_name": "knowledgator/gliner-linker-large-v1.0",
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
        "model_name": "knowledgator/gliner-linker-large-v1.0",
        "huggingface_token": "hf_",
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


# L3 component (uses global GLiNER mock)
@pytest.fixture
def l3_component(l3_config):
    """L3Component instance (uses global GLiNER mock)."""
    from glinker.l3.component import L3Component

    return L3Component(l3_config)


# ============================================================
# L4 FIXTURES
# ============================================================

@pytest.fixture
def l4_config_dict() -> Dict[str, Any]:
    """L4 processor configuration dictionary."""
    return {
        "model_name": "knowledgator/gliner-linker-large-v1.0",
        "token": "hf_",
        "device": "cpu",
        "threshold": 0.5,
        "flat_ner": True,
        "multi_label": False,
        "max_labels": 20,
        "max_length": 512
    }


@pytest.fixture
def l4_config(l4_config_dict):
    """L4Config instance."""
    from glinker.l4.models import L4Config
    return L4Config(**l4_config_dict)


# L4 component (uses global GLiNER mock)
@pytest.fixture
def l4_component(l4_config):
    """L4Component instance (uses global GLiNER mock)."""
    from glinker.l4.component import L4Component

    return L4Component(l4_config)


@pytest.fixture
def l4_processor(l4_component, l4_config):
    """L4Processor instance."""
    from glinker.l4.processor import L4Processor
    return L4Processor(l4_config, l4_component)


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
                "processor": "l1_spacy",
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
                    "model_name": "knowledgator/gliner-linker-large-v1.0",
                    "huggingface_token": "hf_",
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
                "processor": "l1_spacy",
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
                    "model_name": "knowledgator/gliner-linker-large-v1.0",
                    "huggingface_token": "hf_",
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
