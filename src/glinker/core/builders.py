"""
Pipeline configuration builder for easy setup.

ConfigBuilder: Unified builder with automatic defaults and full customization support.
"""

from typing import List, Optional, Dict, Any, Literal
import yaml
from pathlib import Path


class ConfigBuilder:
    """
    Unified configuration builder for pipeline setup.

    Automatically creates simple configs with dict-based L2 by default.
    Supports full customization when needed.

    Simple usage (auto dict layer):
        builder = ConfigBuilder(name="demo")
        builder.l1.gliner(model="...", labels=[...])
        builder.l3.configure(model="...")
        config = builder.get_config()
        builder.save("config.yaml")

    Advanced usage (custom layers):
        builder = ConfigBuilder(name="production")
        builder.l1.gliner(model="...", labels=[...])
        builder.l2.add("redis", priority=2, ttl=3600)
        builder.l2.add("postgres", priority=0)
        builder.l2.embeddings(enabled=True)
        builder.l3.configure(model="...")
        builder.l0.configure(strict_matching=True)
        builder.save("config.yaml")
    """

    class L1Builder:
        """L1 configuration builder"""

        def __init__(self, parent):
            self.parent = parent

        def spacy(
            self,
            model: str = "en_core_sci_sm",
            device: str = "cpu",
            batch_size: int = 32,
            max_right_context: int = 50,
            max_left_context: int = 50,
            min_entity_length: int = 2,
            include_noun_chunks: bool = False
        ) -> "ConfigBuilder":
            """Configure L1 with spaCy NER"""
            self.parent._l1_type = "l1_spacy"
            self.parent._l1_config = {
                "model": model,
                "device": device,
                "batch_size": batch_size,
                "max_right_context": max_right_context,
                "max_left_context": max_left_context,
                "min_entity_length": min_entity_length,
                "include_noun_chunks": include_noun_chunks
            }
            return self.parent

        def gliner(
            self,
            model: str,
            labels: List[str],
            token: Optional[str] = None,
            device: str = "cpu",
            threshold: float = 0.3,
            flat_ner: bool = True,
            multi_label: bool = False,
            batch_size: int = 16,
            max_right_context: int = 50,
            max_left_context: int = 50,
            min_entity_length: int = 2,
            use_precomputed_embeddings: bool = False,
            max_length: Optional[int] = 512
        ) -> "ConfigBuilder":
            """Configure L1 with GLiNER"""
            self.parent._l1_type = "l1_gliner"
            self.parent._l1_config = {
                "model": model,
                "labels": labels,
                "token": token,
                "device": device,
                "threshold": threshold,
                "flat_ner": flat_ner,
                "multi_label": multi_label,
                "batch_size": batch_size,
                "max_right_context": max_right_context,
                "max_left_context": max_left_context,
                "min_entity_length": min_entity_length,
                "use_precomputed_embeddings": use_precomputed_embeddings,
                "max_length": max_length
            }
            return self.parent

    class L2Builder:
        """L2 configuration builder"""

        def __init__(self, parent):
            self.parent = parent

        def add(
            self,
            layer_type: Literal["dict", "redis", "elasticsearch", "postgres"],
            priority: int = 0,
            write: bool = None,
            search_mode: List[str] = None,
            ttl: int = None,
            cache_policy: str = None,
            fuzzy_similarity: float = None,
            **db_config
        ) -> "ConfigBuilder":
            """
            Add a database layer to L2.

            Args:
                layer_type: Type of layer ("dict", "redis", "elasticsearch", "postgres")
                priority: Layer priority (higher = checked first)
                write: Whether to write to this layer (auto: True for cache, False for postgres)
                search_mode: List of search modes (auto: ["exact"] for redis, ["exact", "fuzzy"] for others)
                ttl: Cache TTL in seconds (auto: 0 for dict, 3600 for redis, 86400 for elasticsearch)
                cache_policy: "always", "miss", or "hit" (auto: "always")
                fuzzy_similarity: Minimum similarity for fuzzy search (auto: 0.6 for dict, 0.3 for others)
                **db_config: Database-specific config (host, port, database, etc.)

            Examples:
                builder.l2.add("dict")
                builder.l2.add("redis", priority=2, ttl=3600, host="localhost", port=6379)
                builder.l2.add("elasticsearch", priority=1, hosts=["http://localhost:9200"])
                builder.l2.add("postgres", priority=0, database="entities_db", user="postgres")
            """
            # Set defaults based on layer type
            if write is None:
                write = layer_type != "postgres"  # Don't write to postgres by default

            if search_mode is None:
                search_mode = ["exact"] if layer_type == "redis" else ["exact", "fuzzy"]

            if ttl is None:
                ttl = {"dict": 0, "redis": 3600, "elasticsearch": 86400, "postgres": 0}.get(layer_type, 0)

            if cache_policy is None:
                cache_policy = "miss" if layer_type == "elasticsearch" else "always"

            # Build layer config
            layer = {
                "type": layer_type,
                "priority": priority,
                "write": write,
                "search_mode": search_mode,
                "ttl": ttl,
                "cache_policy": cache_policy,
                "field_mapping": self._default_field_mapping()
            }

            # Add database-specific config
            if layer_type == "dict":
                if fuzzy_similarity is None:
                    fuzzy_similarity = 0.75
                layer["fuzzy"] = {
                    "max_distance": 64,
                    "min_similarity": fuzzy_similarity,
                    "n_gram_size": 3,
                    "prefix_length": 1
                }

            elif layer_type == "redis":
                layer["config"] = {
                    "host": db_config.get("host", "localhost"),
                    "port": db_config.get("port", 6379),
                    "db": db_config.get("db", 0)
                }

            elif layer_type == "elasticsearch":
                if fuzzy_similarity is None:
                    fuzzy_similarity = 0.3
                layer["config"] = {
                    "hosts": db_config.get("hosts", ["http://localhost:9200"]),
                    "index_name": db_config.get("index_name", "entities")
                }
                layer["fuzzy"] = {"min_similarity": fuzzy_similarity}

            elif layer_type == "postgres":
                if fuzzy_similarity is None:
                    fuzzy_similarity = 0.3
                layer["config"] = {
                    "host": db_config.get("host", "localhost"),
                    "port": db_config.get("port", 5432),
                    "database": db_config.get("database", "entities_db"),
                    "user": db_config.get("user", "postgres"),
                    "password": db_config.get("password", "postgres")
                }
                layer["fuzzy"] = {"min_similarity": fuzzy_similarity}

            self.parent._l2_layers.append(layer)
            return self.parent

        def embeddings(
            self,
            enabled: bool = True,
            model_name: str = "knowledgator/gliner-linker-large-v1.0",
            dim: int = 768,
            precompute_on_load: bool = False
        ) -> "ConfigBuilder":
            """Configure embeddings for L2 (BiEncoder support)"""
            self.parent._l2_embeddings = {
                "enabled": enabled,
                "model_name": model_name,
                "dim": dim,
                "precompute_on_load": precompute_on_load
            }

            # Add embedding fields to all layers
            for layer in self.parent._l2_layers:
                layer["field_mapping"]["embedding"] = "embedding"
                layer["field_mapping"]["embedding_model_id"] = "embedding_model_id"

            return self.parent

        def _default_field_mapping(self) -> Dict[str, str]:
            """Default field mapping"""
            mapping = {
                "entity_id": "entity_id",
                "label": "label",
                "aliases": "aliases",
                "description": "description",
                "entity_type": "entity_type",
                "popularity": "popularity"
            }

            # Add embedding fields if embeddings enabled
            if self.parent._l2_embeddings:
                mapping["embedding"] = "embedding"
                mapping["embedding_model_id"] = "embedding_model_id"

            return mapping

    class L3Builder:
        """L3 configuration builder"""

        def __init__(self, parent):
            self.parent = parent

        def configure(
            self,
            model: str = "knowledgator/gliner-linker-large-v1.0",
            token: Optional[str] = None,
            device: str = "cpu",
            threshold: float = 0.5,
            flat_ner: bool = True,
            multi_label: bool = False,
            batch_size: int = 1,
            use_precomputed_embeddings: bool = False,
            cache_embeddings: bool = False,
            max_length: Optional[int] = 512
        ) -> "ConfigBuilder":
            """Configure L3 entity disambiguation"""
            self.parent._l3_config = {
                "model_name": model,
                "huggingface_token": token,
                "device": device,
                "threshold": threshold,
                "flat_ner": flat_ner,
                "multi_label": multi_label,
                "batch_size": batch_size,
                "use_precomputed_embeddings": use_precomputed_embeddings,
                "cache_embeddings": cache_embeddings,
                "max_length": max_length
            }
            return self.parent

    class L4Builder:
        """L4 configuration builder (optional GLiNER reranker with chunking)"""

        def __init__(self, parent):
            self.parent = parent

        def configure(
            self,
            model: str = "knowledgator/gliner-linker-large-v1.0",
            token: Optional[str] = None,
            device: str = "cpu",
            threshold: float = 0.5,
            flat_ner: bool = True,
            multi_label: bool = False,
            max_labels: int = 20,
            max_length: Optional[int] = 512
        ) -> "ConfigBuilder":
            """Configure L4 GLiNER reranker with candidate chunking.

            Args:
                model: GLiNER model (uni-encoder)
                threshold: Minimum score for entity predictions
                max_labels: Maximum candidate labels per inference call.
                    Candidates exceeding this are split into chunks.
            """
            self.parent._l4_config = {
                "model_name": model,
                "token": token,
                "device": device,
                "threshold": threshold,
                "flat_ner": flat_ner,
                "multi_label": multi_label,
                "max_labels": max_labels,
                "max_length": max_length
            }
            return self.parent

    class L0Builder:
        """L0 configuration builder"""

        def __init__(self, parent):
            self.parent = parent

        def configure(
            self,
            min_confidence: float = 0.0,
            include_unlinked: bool = True,
            return_all_candidates: bool = False,
            strict_matching: bool = True,
            position_tolerance: int = 2
        ) -> "ConfigBuilder":
            """Configure L0 aggregation parameters"""
            self.parent._l0_config = {
                "min_confidence": min_confidence,
                "include_unlinked": include_unlinked,
                "return_all_candidates": return_all_candidates,
                "strict_matching": strict_matching,
                "position_tolerance": position_tolerance
            }
            return self.parent

    def __init__(self, name: str = "pipeline", description: str = None):
        self.name = name
        self.description = description or f"{name} - auto-generated configuration"
        self._l1_config = None
        self._l1_type = None
        self._l2_layers = []
        self._l2_embeddings = None
        self._l3_config = None
        self._l4_config = None
        self._l0_config = {
            "min_confidence": 0.0,
            "include_unlinked": True,
            "return_all_candidates": False,
            "strict_matching": True,
            "position_tolerance": 2
        }
        self._schema_template = "{label}: {description}"

        # Initialize builders
        self.l1 = self.L1Builder(self)
        self.l2 = self.L2Builder(self)
        self.l3 = self.L3Builder(self)
        self.l4 = self.L4Builder(self)
        self.l0 = self.L0Builder(self)

    def set_schema_template(self, template: str) -> "ConfigBuilder":
        """Set label formatting template for L2/L3/L0"""
        self._schema_template = template
        return self

    def get_config(self) -> Dict[str, Any]:
        """
        Get pipeline configuration as Python dictionary.

        Returns:
            dict: Complete pipeline configuration
        """
        return self.build()

    def build(self) -> Dict[str, Any]:
        """Build pipeline configuration dictionary"""
        if not self._l1_type or not self._l1_config:
            raise ValueError("L1 configuration is required. Call builder.l1.spacy() or builder.l1.gliner() first.")

        if not self._l3_config:
            raise ValueError("L3 configuration is required. Call builder.l3.configure() first.")

        # Auto-add dict layer if no L2 layers specified
        if not self._l2_layers:
            self._l2_layers.append({
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
            })

        # Build L2 config
        l2_config = {
            "max_candidates": 10 if self._l2_embeddings else 5,
            "min_popularity": 0,
            "layers": self._l2_layers
        }

        if self._l2_embeddings:
            l2_config["embeddings"] = self._l2_embeddings

        nodes = [
            # L1 Node
            {
                "id": "l1",
                "processor": self._l1_type,
                "inputs": {
                    "texts": {
                        "source": "$input",
                        "fields": "texts"
                    }
                },
                "output": {"key": "l1_result"},
                "config": self._l1_config
            },
            # L2 Node
            {
                "id": "l2",
                "processor": "l2_chain",
                "requires": ["l1"],
                "inputs": {
                    "mentions": {
                        "source": "l1_result",
                        "fields": "entities"
                    }
                },
                "output": {"key": "l2_result"},
                "schema": {"template": self._schema_template},
                "config": l2_config
            },
            # L3 Node
            {
                "id": "l3",
                "processor": "l3_batch",
                "requires": ["l1", "l2"],
                "inputs": {
                    "texts": {
                        "source": "$input",
                        "fields": "texts"
                    },
                    "candidates": {
                        "source": "l2_result",
                        "fields": "candidates"
                    },
                    "l1_entities": {
                        "source": "l1_result",
                        "fields": "entities"
                    }
                },
                "output": {"key": "l3_result"},
                "schema": {"template": self._schema_template},
                "config": self._l3_config
            },
        ]

        # Determine which result L0 reads entity predictions from
        l0_entity_source = "l3_result"
        l0_requires = ["l1", "l2", "l3"]

        # Optional L4 reranker node
        if self._l4_config:
            nodes.append({
                "id": "l4",
                "processor": "l4_reranker",
                "requires": ["l1", "l2", "l3"],
                "inputs": {
                    "texts": {
                        "source": "$input",
                        "fields": "texts"
                    },
                    "candidates": {
                        "source": "l2_result",
                        "fields": "candidates"
                    },
                    "l1_entities": {
                        "source": "l1_result",
                        "fields": "entities"
                    }
                },
                "output": {"key": "l4_result"},
                "schema": {"template": self._schema_template},
                "config": self._l4_config
            })
            l0_entity_source = "l4_result"
            l0_requires.append("l4")

        # L0 Node
        nodes.append({
            "id": "l0",
            "processor": "l0_aggregator",
            "requires": l0_requires,
            "inputs": {
                "l1_entities": {
                    "source": "l1_result",
                    "fields": "entities"
                },
                "l2_candidates": {
                    "source": "l2_result",
                    "fields": "candidates"
                },
                "l3_entities": {
                    "source": l0_entity_source,
                    "fields": "entities"
                }
            },
            "output": {"key": "l0_result"},
            "config": self._l0_config,
            "schema": {"template": self._schema_template}
        })

        config = {
            "name": self.name,
            "description": self.description,
            "nodes": nodes
        }

        return config

    def save(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        config = self.build()

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Configuration saved to {filepath}")
