from pathlib import Path
import yaml
from .registry import processor_registry
from .dag_executor import DAGPipeline, DAGExecutor
from .pipe_node import PipeNode
from .input_config import InputConfig, OutputConfig


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
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
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