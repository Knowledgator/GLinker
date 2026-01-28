"""
Tests for src/core/builders.py - ConfigBuilder.
"""

import pytest
import yaml
from pathlib import Path


class TestConfigBuilderInit:
    """Tests for ConfigBuilder initialization."""

    def test_builder_import(self):
        from glinker.core.builders import ConfigBuilder
        assert ConfigBuilder is not None

    def test_builder_init_with_name(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test_pipeline")
        assert builder.name == "test_pipeline"

    def test_builder_init_with_description(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test", description="Custom description")
        assert builder.description == "Custom description"

    def test_builder_init_auto_description(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        assert "test" in builder.description
        assert "auto-generated" in builder.description

    def test_builder_has_sub_builders(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        assert hasattr(builder, "l1")
        assert hasattr(builder, "l2")
        assert hasattr(builder, "l3")
        assert hasattr(builder, "l0")

    def test_builder_default_schema_template(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        assert builder._schema_template == "{label}: {description}"


class TestL1Builder:
    """Tests for L1 configuration builder."""

    def test_l1_spacy_default_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        result = builder.l1.spacy()

        assert builder._l1_type == "l1_spacy"
        assert builder._l1_config["model"] == "en_core_sci_sm"
        assert builder._l1_config["device"] == "cpu"
        assert builder._l1_config["batch_size"] == 32
        assert result is builder  # Check chaining

    def test_l1_spacy_custom_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy(
            model="en_core_web_lg",
            device="cuda",
            batch_size=64,
            min_entity_length=3
        )

        assert builder._l1_config["model"] == "en_core_web_lg"
        assert builder._l1_config["device"] == "cuda"
        assert builder._l1_config["batch_size"] == 64
        assert builder._l1_config["min_entity_length"] == 3

    def test_l1_gliner_minimal_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.gliner(
            model="urchade/gliner_small-v2.1",
            labels=["gene", "protein"]
        )

        assert builder._l1_type == "l1_gliner"
        assert builder._l1_config["model"] == "urchade/gliner_small-v2.1"
        assert builder._l1_config["labels"] == ["gene", "protein"]

    def test_l1_gliner_full_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.gliner(
            model="urchade/gliner_small-v2.1",
            labels=["gene", "disease"],
            token="hf_token",
            device="cuda",
            threshold=0.5,
            batch_size=32,
            use_precomputed_embeddings=True,
            max_length=1024
        )

        assert builder._l1_config["token"] == "hf_token"
        assert builder._l1_config["threshold"] == 0.5
        assert builder._l1_config["use_precomputed_embeddings"] is True
        assert builder._l1_config["max_length"] == 1024

    def test_l1_builder_returns_parent(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        result = builder.l1.spacy()
        assert result is builder


class TestL2Builder:
    """Tests for L2 configuration builder."""

    def test_l2_add_dict_layer_defaults(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("dict")

        assert len(builder._l2_layers) == 1
        layer = builder._l2_layers[0]
        assert layer["type"] == "dict"
        assert layer["priority"] == 0
        assert layer["write"] is True
        assert layer["search_mode"] == ["exact", "fuzzy"]
        assert layer["ttl"] == 0
        assert layer["fuzzy"]["min_similarity"] == 0.6

    def test_l2_add_redis_layer_defaults(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("redis")

        layer = builder._l2_layers[0]
        assert layer["type"] == "redis"
        assert layer["search_mode"] == ["exact"]  # Redis only exact
        assert layer["ttl"] == 3600
        assert layer["config"]["host"] == "localhost"
        assert layer["config"]["port"] == 6379

    def test_l2_add_elasticsearch_layer_defaults(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("elasticsearch")

        layer = builder._l2_layers[0]
        assert layer["type"] == "elasticsearch"
        assert layer["ttl"] == 86400
        assert layer["cache_policy"] == "miss"
        assert layer["fuzzy"]["min_similarity"] == 0.3
        assert layer["config"]["hosts"] == ["http://localhost:9200"]

    def test_l2_add_postgres_layer_defaults(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("postgres")

        layer = builder._l2_layers[0]
        assert layer["type"] == "postgres"
        assert layer["write"] is False  # Postgres no write by default
        assert layer["fuzzy"]["min_similarity"] == 0.3
        assert layer["config"]["database"] == "entities_db"

    def test_l2_add_custom_priority(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("redis", priority=2)
        builder.l2.add("postgres", priority=0)

        assert builder._l2_layers[0]["priority"] == 2
        assert builder._l2_layers[1]["priority"] == 0

    def test_l2_add_custom_db_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add(
            "postgres",
            host="custom-host",
            port=5433,
            database="custom_db",
            user="admin",
            password="secret"
        )

        config = builder._l2_layers[0]["config"]
        assert config["host"] == "custom-host"
        assert config["port"] == 5433
        assert config["database"] == "custom_db"
        assert config["user"] == "admin"
        assert config["password"] == "secret"

    def test_l2_add_multiple_layers(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("redis", priority=2)
        builder.l2.add("elasticsearch", priority=1)
        builder.l2.add("postgres", priority=0)

        assert len(builder._l2_layers) == 3
        assert builder._l2_layers[0]["type"] == "redis"
        assert builder._l2_layers[1]["type"] == "elasticsearch"
        assert builder._l2_layers[2]["type"] == "postgres"

    def test_l2_embeddings_defaults(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("postgres")
        builder.l2.embeddings()

        assert builder._l2_embeddings["enabled"] is True
        assert builder._l2_embeddings["model_name"] == "BioMike/gliner-deberta-base-v1-post"
        assert builder._l2_embeddings["dim"] == 768

    def test_l2_embeddings_custom_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("postgres")
        builder.l2.embeddings(
            enabled=True,
            model_name="custom/model",
            dim=1024,
            precompute_on_load=True
        )

        assert builder._l2_embeddings["model_name"] == "custom/model"
        assert builder._l2_embeddings["dim"] == 1024
        assert builder._l2_embeddings["precompute_on_load"] is True

    def test_l2_embeddings_adds_fields_to_layers(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l2.add("postgres")
        builder.l2.embeddings()

        layer = builder._l2_layers[0]
        assert "embedding" in layer["field_mapping"]
        assert "embedding_model_id" in layer["field_mapping"]

    def test_l2_builder_returns_parent(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        result = builder.l2.add("dict")
        assert result is builder


class TestL3Builder:
    """Tests for L3 configuration builder."""

    def test_l3_configure_defaults(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l3.configure()

        assert builder._l3_config["model_name"] == "BioMike/gliner-deberta-base-v1-post"
        assert builder._l3_config["device"] == "cpu"
        assert builder._l3_config["threshold"] == 0.5
        assert builder._l3_config["use_precomputed_embeddings"] is False

    def test_l3_configure_custom_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l3.configure(
            model="custom/model",
            token="hf_token",
            device="cuda",
            threshold=0.7,
            batch_size=8,
            use_precomputed_embeddings=True,
            cache_embeddings=True,
            max_length=1024
        )

        assert builder._l3_config["model_name"] == "custom/model"
        assert builder._l3_config["huggingface_token"] == "hf_token"
        assert builder._l3_config["device"] == "cuda"
        assert builder._l3_config["threshold"] == 0.7
        assert builder._l3_config["batch_size"] == 8
        assert builder._l3_config["use_precomputed_embeddings"] is True
        assert builder._l3_config["cache_embeddings"] is True
        assert builder._l3_config["max_length"] == 1024

    def test_l3_builder_returns_parent(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        result = builder.l3.configure()
        assert result is builder


class TestL0Builder:
    """Tests for L0 configuration builder."""

    def test_l0_configure_defaults(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        # L0 has default config
        assert builder._l0_config["min_confidence"] == 0.0
        assert builder._l0_config["include_unlinked"] is True
        assert builder._l0_config["strict_matching"] is True

    def test_l0_configure_custom_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l0.configure(
            min_confidence=0.5,
            include_unlinked=False,
            return_all_candidates=True,
            strict_matching=False,
            position_tolerance=5
        )

        assert builder._l0_config["min_confidence"] == 0.5
        assert builder._l0_config["include_unlinked"] is False
        assert builder._l0_config["return_all_candidates"] is True
        assert builder._l0_config["strict_matching"] is False
        assert builder._l0_config["position_tolerance"] == 5

    def test_l0_builder_returns_parent(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        result = builder.l0.configure()
        assert result is builder


class TestConfigBuilding:
    """Tests for config building and validation."""

    def test_build_requires_l1(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l3.configure()

        with pytest.raises(ValueError, match="L1 configuration is required"):
            builder.build()

    def test_build_requires_l3(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()

        with pytest.raises(ValueError, match="L3 configuration is required"):
            builder.build()

    def test_build_minimal_config(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()

        config = builder.build()
        assert config["name"] == "test"
        assert len(config["nodes"]) == 4  # L1, L2, L3, L0

    def test_build_auto_adds_dict_layer(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()

        config = builder.build()
        l2_node = [n for n in config["nodes"] if n["id"] == "l2"][0]
        assert len(l2_node["config"]["layers"]) == 1
        assert l2_node["config"]["layers"][0]["type"] == "dict"

    def test_build_uses_custom_l2_layers(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l2.add("redis", priority=2)
        builder.l2.add("postgres", priority=0)
        builder.l3.configure()

        config = builder.build()
        l2_node = [n for n in config["nodes"] if n["id"] == "l2"][0]
        assert len(l2_node["config"]["layers"]) == 2
        assert l2_node["config"]["layers"][0]["type"] == "redis"
        assert l2_node["config"]["layers"][1]["type"] == "postgres"

    def test_build_includes_embeddings(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l2.add("postgres")
        builder.l2.embeddings(model_name="custom/model")
        builder.l3.configure()

        config = builder.build()
        l2_node = [n for n in config["nodes"] if n["id"] == "l2"][0]
        assert "embeddings" in l2_node["config"]
        assert l2_node["config"]["embeddings"]["model_name"] == "custom/model"

    def test_build_sets_max_candidates_with_embeddings(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l2.add("postgres")
        builder.l2.embeddings()
        builder.l3.configure()

        config = builder.build()
        l2_node = [n for n in config["nodes"] if n["id"] == "l2"][0]
        assert l2_node["config"]["max_candidates"] == 10

    def test_build_sets_max_candidates_without_embeddings(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()

        config = builder.build()
        l2_node = [n for n in config["nodes"] if n["id"] == "l2"][0]
        assert l2_node["config"]["max_candidates"] == 5

    def test_build_includes_schema_template(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()
        builder.set_schema_template("{label} ({entity_type})")

        config = builder.build()
        l2_node = [n for n in config["nodes"] if n["id"] == "l2"][0]
        l3_node = [n for n in config["nodes"] if n["id"] == "l3"][0]
        l0_node = [n for n in config["nodes"] if n["id"] == "l0"][0]

        assert l2_node["schema"]["template"] == "{label} ({entity_type})"
        assert l3_node["schema"]["template"] == "{label} ({entity_type})"
        assert l0_node["schema"]["template"] == "{label} ({entity_type})"

    def test_get_config_same_as_build(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()

        config1 = builder.build()
        config2 = builder.get_config()
        assert config1 == config2


class TestSchemaTemplate:
    """Tests for schema template configuration."""

    def test_set_schema_template(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.set_schema_template("{label}")
        assert builder._schema_template == "{label}"

    def test_set_schema_template_returns_builder(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        result = builder.set_schema_template("{label}")
        assert result is builder


class TestSaveConfig:
    """Tests for saving configuration to file."""

    def test_save_creates_file(self, tmp_path):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()

        filepath = tmp_path / "config.yaml"
        builder.save(str(filepath))

        assert filepath.exists()

    def test_save_creates_directory(self, tmp_path):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()

        filepath = tmp_path / "nested" / "dir" / "config.yaml"
        builder.save(str(filepath))

        assert filepath.exists()

    def test_save_valid_yaml(self, tmp_path):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()

        filepath = tmp_path / "config.yaml"
        builder.save(str(filepath))

        with open(filepath) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config["name"] == "test"
        assert len(loaded_config["nodes"]) == 4

    def test_save_matches_build(self, tmp_path):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="test")
        builder.l1.spacy()
        builder.l3.configure()

        config = builder.get_config()

        filepath = tmp_path / "config.yaml"
        builder.save(str(filepath))

        with open(filepath) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config == config


class TestCompleteExamples:
    """Tests for complete pipeline configurations."""

    def test_simple_spacy_pipeline(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="simple_spacy")
        builder.l1.spacy(model="en_core_sci_sm")
        builder.l3.configure()

        config = builder.get_config()
        assert config["name"] == "simple_spacy"
        assert len(config["nodes"]) == 4

    def test_gliner_pipeline_with_custom_labels(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="gliner_bio")
        builder.l1.gliner(
            model="urchade/gliner_small-v2.1",
            labels=["gene", "protein", "disease"]
        )
        builder.l3.configure()

        config = builder.get_config()
        l1_node = [n for n in config["nodes"] if n["id"] == "l1"][0]
        assert l1_node["config"]["labels"] == ["gene", "protein", "disease"]

    def test_production_pipeline_multi_layer(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="production")
        builder.l1.gliner(
            model="urchade/gliner_small-v2.1",
            labels=["gene", "protein"]
        )
        builder.l2.add("redis", priority=2, ttl=3600)
        builder.l2.add("elasticsearch", priority=1)
        builder.l2.add("postgres", priority=0)
        builder.l3.configure()

        config = builder.get_config()
        l2_node = [n for n in config["nodes"] if n["id"] == "l2"][0]
        assert len(l2_node["config"]["layers"]) == 3

    def test_pipeline_with_embeddings(self):
        from glinker.core.builders import ConfigBuilder
        builder = ConfigBuilder(name="with_embeddings")
        builder.l1.gliner(
            model="urchade/gliner_small-v2.1",
            labels=["gene"],
            use_precomputed_embeddings=True
        )
        builder.l2.add("postgres")
        builder.l2.embeddings(enabled=True)
        builder.l3.configure(use_precomputed_embeddings=True)
        builder.l0.configure(min_confidence=0.3)

        config = builder.get_config()
        l2_node = [n for n in config["nodes"] if n["id"] == "l2"][0]
        l3_node = [n for n in config["nodes"] if n["id"] == "l3"][0]

        assert "embeddings" in l2_node["config"]
        assert l3_node["config"]["use_precomputed_embeddings"] is True

    def test_chained_configuration(self):
        from glinker.core.builders import ConfigBuilder

        config = (
            ConfigBuilder(name="chained")
            .l1.spacy()
            .l2.add("redis", priority=1)
            .l3.configure()
            .l0.configure(min_confidence=0.5)
            .set_schema_template("{label}")
            .get_config()
        )

        assert config["name"] == "chained"
        assert len(config["nodes"]) == 4
