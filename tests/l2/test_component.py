"""
Tests for src/l2/component.py - L2 database components.
"""

import pytest


class TestDictLayer:
    """Tests for DictLayer (in-memory database layer)."""

    def test_import(self):
        from src.l2.component import DictLayer
        assert DictLayer is not None

    def test_creation(self, dict_layer):
        assert dict_layer is not None

    def test_is_available(self, dict_layer):
        assert dict_layer.is_available() is True

    def test_load_bulk(self, dict_layer, sample_entities):
        from src.l2.models import DatabaseRecord

        records = [
            DatabaseRecord(
                entity_id=e["entity_id"],
                label=e["label"],
                description=e["description"],
                entity_type=e["entity_type"],
                popularity=e["popularity"],
                aliases=e["aliases"]
            )
            for e in sample_entities
        ]

        dict_layer.load_bulk(records)
        assert len(dict_layer._storage) == len(sample_entities)

    def test_search_exact(self, loaded_dict_layer):
        results = loaded_dict_layer.search("TP53")
        assert len(results) >= 1
        assert any(r.label == "TP53" for r in results)

    def test_search_by_alias(self, loaded_dict_layer):
        results = loaded_dict_layer.search("p53")
        assert len(results) >= 1
        # Should find TP53 via alias
        assert any(r.label == "TP53" for r in results)

    def test_search_case_insensitive(self, loaded_dict_layer):
        results_lower = loaded_dict_layer.search("tp53")
        results_upper = loaded_dict_layer.search("TP53")
        assert len(results_lower) == len(results_upper)

    def test_search_not_found(self, loaded_dict_layer):
        results = loaded_dict_layer.search("NONEXISTENT_XYZ")
        assert len(results) == 0

    def test_search_fuzzy(self, loaded_dict_layer):
        # Search with typo
        results = loaded_dict_layer.search_fuzzy("TP5")
        assert len(results) >= 1

    def test_search_fuzzy_partial(self, loaded_dict_layer):
        results = loaded_dict_layer.search_fuzzy("breast")
        assert len(results) >= 1

    def test_get_all_entities(self, loaded_dict_layer, sample_entities):
        all_entities = loaded_dict_layer.get_all_entities()
        assert len(all_entities) == len(sample_entities)

    def test_update_embeddings(self, loaded_dict_layer, sample_entities):
        entity_ids = [sample_entities[0]["entity_id"]]
        embeddings = [[0.1] * 768]
        model_id = "test-model"

        count = loaded_dict_layer.update_embeddings(entity_ids, embeddings, model_id)
        assert count == 1

        # Verify embedding stored - DatabaseRecord uses attribute access not dict
        record = loaded_dict_layer._storage[sample_entities[0]["entity_id"]]
        assert record.embedding is not None
        assert record.embedding_model_id == model_id

    def test_count(self, loaded_dict_layer, sample_entities):
        assert loaded_dict_layer.count() == len(sample_entities)

    def test_count_empty(self, dict_layer):
        assert dict_layer.count() == 0


class TestDatabaseChainComponent:
    """Tests for DatabaseChainComponent."""

    def test_import(self):
        from src.l2.component import DatabaseChainComponent
        assert DatabaseChainComponent is not None

    def test_creation(self, l2_component):
        assert l2_component is not None

    def test_has_layers(self, l2_component):
        assert hasattr(l2_component, 'layers')
        assert len(l2_component.layers) >= 1

    def test_get_available_methods(self, l2_component):
        methods = l2_component.get_available_methods()
        assert isinstance(methods, list)
        assert "search" in methods
        assert "filter_by_popularity" in methods

    def test_search(self, l2_component, sample_entities):
        from src.l2.models import DatabaseRecord

        # Load entities first
        records = [
            DatabaseRecord(
                entity_id=e["entity_id"],
                label=e["label"],
                description=e["description"],
                entity_type=e["entity_type"],
                popularity=e["popularity"],
                aliases=e["aliases"]
            )
            for e in sample_entities
        ]
        l2_component.layers[0].load_bulk(records)

        results = l2_component.search("TP53")
        assert len(results) >= 1

    def test_search_multiple(self, l2_component, sample_entities):
        from src.l2.models import DatabaseRecord

        records = [
            DatabaseRecord(
                entity_id=e["entity_id"],
                label=e["label"],
                description=e["description"],
                entity_type=e["entity_type"],
                popularity=e["popularity"],
                aliases=e["aliases"]
            )
            for e in sample_entities
        ]
        l2_component.layers[0].load_bulk(records)

        # Test multiple individual searches
        queries = ["TP53", "BRCA1", "NONEXISTENT"]
        results = [l2_component.search(q) for q in queries]

        assert len(results) == 3
        assert len(results[0]) >= 1  # TP53 found
        assert len(results[1]) >= 1  # BRCA1 found
        assert len(results[2]) == 0  # Not found


class TestDatabaseChainComponentPrecompute:
    """Tests for precompute_embeddings in DatabaseChainComponent."""

    def test_precompute_embeddings_method_exists(self, l2_component):
        assert hasattr(l2_component, 'precompute_embeddings')

    def test_precompute_embeddings(self, l2_component, sample_entities):
        from src.l2.models import DatabaseRecord
        import torch

        # Load entities
        records = [
            DatabaseRecord(
                entity_id=e["entity_id"],
                label=e["label"],
                description=e["description"],
                entity_type=e["entity_type"],
                popularity=e["popularity"],
                aliases=e["aliases"]
            )
            for e in sample_entities
        ]
        l2_component.layers[0].load_bulk(records)

        # Mock encoder function
        def mock_encoder(labels):
            return torch.randn(len(labels), 768)

        stats = l2_component.precompute_embeddings(
            encoder_fn=mock_encoder,
            template="{label}: {description}",
            model_id="test-model",
            target_layers=["dict"],
            batch_size=2
        )

        assert "dict" in stats
        assert stats["dict"] == len(sample_entities)

        # Verify embeddings stored - DatabaseRecord uses attribute access
        layer = l2_component.layers[0]
        sample_id = sample_entities[0]["entity_id"]
        assert layer._storage[sample_id].embedding is not None
