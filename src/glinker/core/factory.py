import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from .registry import processor_registry
from .dag import DAGPipeline, DAGExecutor, PipeNode, InputConfig, OutputConfig


def load_yaml(path: str | Path) -> dict:
    """Load YAML configuration file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class ProcessorFactory:
    """Factory for creating pipelines from configs"""
    
    @staticmethod
    def create_from_registry(
        processor_name: str,
        config_dict: dict,
        pipeline: list[tuple[str, dict]] = None
    ):
        """
        Create single processor from registry
        
        For internal use by DAGExecutor
        """
        factory = processor_registry.get(processor_name)
        return factory(config_dict, pipeline)
    
    @staticmethod
    def create_pipeline(config_path: str | Path, verbose: bool = False) -> DAGExecutor:
        """
        Create DAG pipeline from YAML config
        
        Supports:
        - Single node (just L2)
        - Multiple nodes (L1 → L2 → L3)
        - Complex DAGs with dependencies
        
        Example config:
            name: "my_pipeline"
            nodes:
              - id: "l2"
                processor: "l2_chain"
                inputs:
                  mentions: {source: "$input", fields: "mentions"}
                output: {key: "result"}
                config: {...}
        """
        config = load_yaml(config_path)
        
        nodes = []
        for node_cfg in config['nodes']:
            inputs = {}
            for name, data in node_cfg['inputs'].items():
                inputs[name] = InputConfig(**data)
            
            node = PipeNode(
                id=node_cfg['id'],
                processor=node_cfg['processor'],
                inputs=inputs,
                output=OutputConfig(**node_cfg['output']),
                requires=node_cfg.get('requires', []),
                config=node_cfg['config'],
                schema=node_cfg.get('schema')
            )
            nodes.append(node)
        
        pipeline = DAGPipeline(
            name=config['name'],
            description=config.get('description'),
            nodes=nodes
        )
        
        return DAGExecutor(pipeline, verbose=verbose)
    
    @staticmethod
    def create_from_dict(config_dict: dict, verbose: bool = False) -> DAGExecutor:
        """
        Create pipeline from dict (for programmatic use)
        
        Same as create_pipeline but accepts dict instead of file path
        """
        nodes = []
        for node_cfg in config_dict['nodes']:
            inputs = {}
            for name, data in node_cfg['inputs'].items():
                inputs[name] = InputConfig(**data)
            
            node = PipeNode(
                id=node_cfg['id'],
                processor=node_cfg['processor'],
                inputs=inputs,
                output=OutputConfig(**node_cfg['output']),
                requires=node_cfg.get('requires', []),
                config=node_cfg['config'],
                schema=node_cfg.get('schema')
            )
            nodes.append(node)
        
        pipeline = DAGPipeline(
            name=config_dict['name'],
            description=config_dict.get('description'),
            nodes=nodes
        )

        return DAGExecutor(pipeline, verbose=verbose)

    @staticmethod
    def create_simple(
        model_name: str,
        device: str = "cpu",
        threshold: float = 0.5,
        template: str = "{label}",
        max_length: Optional[int] = 512,
        token: Optional[str] = None,
        entities: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
        precompute_embeddings: bool = False,
        verbose: bool = False,
        reranker_model: Optional[str] = None,
        reranker_max_labels: int = 20,
        reranker_threshold: Optional[float] = None,
        external_entities: bool = False,
    ) -> DAGExecutor:
        """
        Create a minimal L2 -> L3 -> L0 pipeline from a model name.

        Skips L1 (mention extraction). L2 serves as an in-memory entity store
        that returns all loaded entities as candidates. L3 runs GLiNER for
        entity linking. L0 aggregates in loose mode.

        When ``external_entities`` is True, the pipeline expects pre-extracted
        entity mentions in the input under the ``entities`` key. Each entry
        must be a list of dicts per text with at least ``text``, ``start``,
        and ``end`` keys. L0 uses strict matching in this mode.

        Optionally adds an L4 reranker after L3 for chunked candidate
        re-evaluation when ``reranker_model`` is provided.

        Args:
            model_name: HuggingFace model ID or local path.
            device: Torch device ("cpu", "cuda", "cuda:0", ...).
            threshold: Minimum score for entity predictions.
            template: Format string for entity labels (e.g. "{label}: {description}").
            max_length: Max sequence length for tokenization.
            token: HuggingFace auth token for gated models.
            entities: Optional entity data to load immediately. Accepts a file
                path (str/Path to JSONL), a list of dicts, or a dict mapping
                entity_id to entity data.
            precompute_embeddings: If True and *entities* are provided,
                pre-embed all entity labels after loading (BiEncoder models only).
            verbose: Enable verbose logging.
            reranker_model: Optional GLiNER model for L4 reranking. When set,
                an L4 node is added after L3.
            reranker_max_labels: Max candidate labels per L4 inference call.
            reranker_threshold: Score threshold for L4. Defaults to *threshold*.
            external_entities: If True, the pipeline reads pre-extracted entity
                mentions from ``$input.entities`` instead of discovering them.
                Input ``entities`` must be a list (one per text) of lists of
                dicts, each with ``text``, ``start``, and ``end`` keys.

        Returns:
            Configured DAGExecutor ready for ``execute``.
        """
        if external_entities:
            warnings.warn(
                "external_entities=True: the pipeline expects pre-extracted "
                "NER mentions in the input under the 'entities' key. "
                "Each element must be a list of dicts per text with at least "
                "'text', 'start', and 'end' keys, e.g.: "
                '"entities": [[{"text": "Aspirin", "start": 0, "end": 7}]]',
                UserWarning,
                stacklevel=2,
            )

        l2_inputs = {
            "texts": {"source": "$input", "fields": "texts"},
        }
        l3_inputs = {
            "texts": {"source": "$input", "fields": "texts"},
            "candidates": {"source": "l2_result", "fields": "candidates"},
        }

        if external_entities:
            l2_inputs["mentions"] = {"source": "$input", "fields": "entities"}
            l3_inputs["l1_entities"] = {"source": "$input", "fields": "entities"}

        nodes = [
            {
                "id": "l2",
                "processor": "l2_chain",
                "requires": [],
                "inputs": l2_inputs,
                "output": {"key": "l2_result"},
                "schema": {"template": template},
                "config": {
                    "max_candidates": 30,
                    "min_popularity": 0,
                    "layers": [
                        {
                            "type": "dict",
                            "priority": 0,
                            "write": True,
                            "search_mode": ["exact"],
                        }
                    ],
                },
            },
            {
                "id": "l3",
                "processor": "l3_batch",
                "requires": ["l2"],
                "inputs": l3_inputs,
                "output": {"key": "l3_result"},
                "schema": {"template": template},
                "config": {
                    "model_name": model_name,
                    "device": device,
                    "threshold": threshold,
                    "flat_ner": True,
                    "multi_label": False,
                    "use_precomputed_embeddings": True,
                    "cache_embeddings": False,
                    "max_length": max_length,
                    "token": token,
                },
            },
        ]

        l0_entity_source = "l3_result"
        l0_requires = ["l2", "l3"]

        if reranker_model:
            nodes.append({
                "id": "l4",
                "processor": "l4_reranker",
                "requires": ["l2", "l3"],
                "inputs": {
                    "texts": {"source": "$input", "fields": "texts"},
                    "candidates": {"source": "l2_result", "fields": "candidates"},
                },
                "output": {"key": "l4_result"},
                "schema": {"template": template},
                "config": {
                    "model_name": reranker_model,
                    "device": device,
                    "threshold": reranker_threshold if reranker_threshold is not None else threshold,
                    "flat_ner": True,
                    "multi_label": False,
                    "max_labels": reranker_max_labels,
                    "max_length": max_length,
                    "token": token,
                },
            })
            l0_entity_source = "l4_result"
            l0_requires.append("l4")

        l0_inputs = {
            "l2_candidates": {"source": "l2_result", "fields": "candidates"},
            "l3_entities": {"source": l0_entity_source, "fields": "entities"},
        }

        if external_entities:
            l0_inputs["l1_entities"] = {"source": "$input", "fields": "entities"}

        nodes.append({
            "id": "l0",
            "processor": "l0_aggregator",
            "requires": l0_requires,
            "inputs": l0_inputs,
            "output": {"key": "l0_result"},
            "schema": {"template": template},
            "config": {
                "strict_matching": external_entities,
                "min_confidence": 0.0,
                "include_unlinked": True,
                "position_tolerance": 2,
            },
        })

        config = {
            "name": "simple",
            "description": "Simple pipeline - L3 only with entity database",
            "nodes": nodes,
        }
        executor = ProcessorFactory.create_from_dict(config, verbose=verbose)

        if entities is not None:
            executor.load_entities(entities)
            if precompute_embeddings:
                executor.precompute_embeddings()

        return executor