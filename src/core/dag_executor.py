from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, deque
from pydantic import BaseModel, Field
import logging

from .pipe_node import PipeNode
from .pipe_context import PipeContext
from .field_resolver import FieldResolver
from .registry import processor_registry


logger = logging.getLogger(__name__)


class DAGPipeline(BaseModel):
    name: str = Field(...)
    nodes: List[PipeNode] = Field(...)
    description: Optional[str] = None


class DAGExecutor:
    def __init__(self, pipeline: DAGPipeline, verbose: bool = False):
        self.pipeline = pipeline
        self.verbose = verbose
        
        self.nodes_map: Dict[str, PipeNode] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        
        self.processors: Dict[str, Any] = {}
        
        self._build_dependency_graph()
        self._initialize_processors()
    
    def _build_dependency_graph(self):
        for node in self.pipeline.nodes:
            self.nodes_map[node.id] = node
        
        for node in self.pipeline.nodes:
            for dep_id in node.requires:
                self.dependency_graph[dep_id].append(node.id)
                self.reverse_graph[node.id].add(dep_id)
            
            for input_config in node.inputs.values():
                source = input_config.source
                
                if source == "$input" or source.startswith("outputs["):
                    continue
                
                if source in self.nodes_map:
                    self.dependency_graph[source].append(node.id)
                    self.reverse_graph[node.id].add(source)
                
                if input_config.reshape and input_config.reshape.by:
                    reshape_source = input_config.reshape.by.split('.')[0]
                    if reshape_source in self.nodes_map:
                        if reshape_source not in self.reverse_graph[node.id]:
                            self.dependency_graph[reshape_source].append(node.id)
                            self.reverse_graph[node.id].add(reshape_source)
    
    def _initialize_processors(self):
        if self.verbose:
            logger.info(f"Initializing {len(self.nodes_map)} processors...")
        
        for node_id, node in self.nodes_map.items():
            try:
                processor_factory = processor_registry.get(node.processor)
                processor = processor_factory(config_dict=node.config, pipeline=None)
                self.processors[node_id] = processor
                
                if self.verbose:
                    logger.info(f"  Created processor for '{node_id}' ({node.processor})")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create processor '{node.processor}' for node '{node_id}': {e}"
                )
        
        if self.verbose:
            logger.info(f"All processors initialized and cached")
    
    def _topological_sort(self) -> List[List[str]]:
        in_degree = {}
        for node_id in self.nodes_map:
            in_degree[node_id] = len(self.reverse_graph.get(node_id, set()))
        
        queue = deque([
            node_id for node_id, degree in in_degree.items() 
            if degree == 0
        ])
        
        levels = []
        visited = set()
        
        while queue:
            current_level = list(queue)
            levels.append(current_level)
            
            next_queue = deque()
            for node_id in current_level:
                visited.add(node_id)
                
                for dependent_id in self.dependency_graph.get(node_id, []):
                    in_degree[dependent_id] -= 1
                    
                    if in_degree[dependent_id] == 0:
                        next_queue.append(dependent_id)
            
            queue = next_queue
        
        if len(visited) != len(self.nodes_map):
            unvisited = set(self.nodes_map.keys()) - visited
            raise ValueError(
                f"Cycle detected in pipeline DAG! "
                f"Unvisited nodes: {unvisited}"
            )
        
        return levels
    
    def execute(self, pipeline_input: Any) -> PipeContext:
        context = PipeContext(pipeline_input)
        execution_levels = self._topological_sort()
        
        if self.verbose:
            logger.info(f"Executing pipeline: {self.pipeline.name}")
            logger.info(f"Total nodes: {len(self.nodes_map)}")
            logger.info(f"Execution levels: {len(execution_levels)}")
        
        for level_idx, level_nodes in enumerate(execution_levels):
            if self.verbose:
                logger.info(f"\n{'='*60}")
                logger.info(
                    f"Level {level_idx + 1}/{len(execution_levels)} "
                    f"({len(level_nodes)} nodes)"
                )
                logger.info(f"{'='*60}")
            
            for node_id in level_nodes:
                self._run_node(node_id, context)
        
        if self.verbose:
            logger.info(f"\nPipeline completed successfully!")
        
        return context
    
    def _run_node(self, node_id: str, context: PipeContext):
        node = self.nodes_map[node_id]
        
        if self.verbose:
            logger.info(f"\nExecuting: {node.id} (processor: {node.processor})")
        
        if node.condition and not self._evaluate_condition(node.condition, context):
            if self.verbose:
                logger.info(f"  Skipped (condition not met)")
            return
        
        kwargs = {}
        for param_name, input_config in node.inputs.items():
            try:
                value = FieldResolver.resolve(context, input_config)
                kwargs[param_name] = value
                
                if self.verbose:
                    logger.info(f"  Input '{param_name}': {input_config.source}")
            except Exception as e:
                raise ValueError(
                    f"Failed to resolve input '{param_name}' for node '{node_id}': {e}"
                )
        
        processor = self.processors[node_id]
        
        if node.schema and hasattr(processor, 'schema'):
            processor.schema = node.schema
        
        try:
            result = processor(**kwargs)
            
            if self.verbose:
                logger.info(f"  Processing...")
        except Exception as e:
            if self.verbose:
                logger.error(f"  Failed: {e}")
            raise RuntimeError(f"Node '{node_id}' failed: {e}")
        
        if node.output.fields:
            result = FieldResolver._resolve_fields(result, node.output.fields)
        
        context.set(node.output.key, result)
        
        if self.verbose:
            logger.info(f"  Output: '{node.output.key}'")
            logger.info(f"  Success")
    
    def _evaluate_condition(self, condition: str, context: PipeContext) -> bool:
        return True