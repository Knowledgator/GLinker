from typing import Dict, List, Set, Any, Optional, Literal, Union
from collections import defaultdict, deque, OrderedDict
from pydantic import BaseModel, Field
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT/OUTPUT CONFIG
# ============================================================================

class ReshapeConfig(BaseModel):
    """Configuration for data reshaping"""
    by: str = Field(..., description="Reference structure path: 'l1_result.entities'")
    mode: Literal["flatten_per_group", "preserve_structure"] = Field(
        "flatten_per_group",
        description="Reshape mode"
    )


class InputConfig(BaseModel):
    """
    Unified input data specification
    
    Examples:
        source: "l1_result"
        fields: "entities[*].text"
        reduce: "flatten"
    """
    source: str = Field(
        ...,
        description="Data source: key ('l1_result'), index ('outputs[-1]'), or '$input'"
    )
    
    fields: Union[str, List[str], None] = Field(
        None,
        description="JSONPath fields: 'entities[*].text' or ['label', 'score']"
    )
    
    reduce: Literal["all", "first", "last", "flatten"] = Field(
        "all",
        description="Reduction mode for lists"
    )
    
    reshape: Optional[ReshapeConfig] = Field(
        None,
        description="Data reshaping configuration"
    )
    
    template: Optional[str] = Field(
        None,
        description="Field concatenation template: '{label}: {description}'"
    )
    
    filter: Optional[str] = Field(
        None,
        description="Filter expression: 'score > 0.5'"
    )
    
    default: Any = None


class OutputConfig(BaseModel):
    """Output specification"""
    key: str = Field(..., description="Key for storing in context")
    fields: Union[str, List[str], None] = Field(
        None,
        description="Fields to save (optional, defaults to all)"
    )


# ============================================================================
# PIPE NODE
# ============================================================================

class PipeNode(BaseModel):
    """
    Single node in DAG pipeline
    
    Represents one processing stage with:
    - Inputs (where to get data)
    - Processor (what to do)
    - Output (where to store result)
    - Dependencies (execution order)
    """
    
    id: str = Field(..., description="Unique node identifier")
    
    processor: str = Field(..., description="Processor name from registry")
    
    inputs: Dict[str, InputConfig] = Field(
        default_factory=dict,
        description="Input parameter mappings"
    )
    
    output: OutputConfig = Field(..., description="Output specification")
    
    requires: List[str] = Field(
        default_factory=list,
        description="Explicit dependencies (node IDs)"
    )
    
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processor configuration"
    )
    
    schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Schema for field mappings/transformations"
    )
    
    condition: Optional[str] = Field(
        None,
        description="Conditional execution expression"
    )
    
    class Config:
        fields = {'schema': 'schema'}


# ============================================================================
# PIPE CONTEXT
# ============================================================================

class PipeContext:
    """
    Pipeline execution context
    
    Stores all outputs from pipeline stages and provides unified access:
    - By key: "l1_result"
    - By index: "outputs[-1]" (last output)
    - Pipeline input: "$input"
    """
    
    def __init__(self, pipeline_input: Any = None):
        self._outputs: OrderedDict[str, Any] = OrderedDict()
        self._execution_order: List[str] = []
        self._pipeline_input = pipeline_input
        self._metadata: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """
        Store output
        
        Args:
            key: Output key
            value: Output value
            metadata: Optional metadata (timing, source, etc.)
        """
        self._outputs[key] = value
        self._execution_order.append(key)
        
        if metadata:
            self._metadata[key] = metadata
    
    def get(self, source: str) -> Any:
        """
        Unified data access
        
        Examples:
        - "$input" → pipeline input
        - "outputs[-1]" → last output
        - "outputs[0]" → first output
        - "l1_result" → by key
        """
        if source == "$input":
            return self._pipeline_input
        
        if source.startswith("outputs["):
            index_str = source.replace("outputs[", "").replace("]", "")
            index = int(index_str)
            
            if index < 0:
                index = len(self._execution_order) + index
            
            if 0 <= index < len(self._execution_order):
                key = self._execution_order[index]
                return self._outputs[key]
            
            return None
        
        return self._outputs.get(source)
    
    def has(self, key: str) -> bool:
        """Check if output exists"""
        return key in self._outputs
    
    def get_all_outputs(self) -> Dict[str, Any]:
        """Get all outputs as dict"""
        return dict(self._outputs)
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for output"""
        return self._metadata.get(key)
    
    def get_execution_order(self) -> List[str]:
        """Get list of output keys in execution order"""
        return list(self._execution_order)
    
    @property
    def data(self) -> Dict[str, Any]:
        """For compatibility"""
        return dict(self._outputs)


# ============================================================================
# FIELD RESOLVER
# ============================================================================

class FieldResolver:
    """Resolve fields from data using path expressions"""
    
    @staticmethod
    def resolve(context: PipeContext, config: InputConfig) -> Any:
        """Main resolve method"""
        data = context.get(config.source)
        if data is None:
            return config.default
        
        if config.fields:
            data = FieldResolver._extract_fields(data, config.fields)
        
        if config.template:
            data = FieldResolver._format_template(data, config.template)
        
        if isinstance(data, list) and config.reduce:
            data = FieldResolver._apply_reduce(data, config.reduce)
        
        if config.filter:
            data = FieldResolver._apply_filter(data, config.filter)
        
        return data
    
    @staticmethod
    def _extract_fields(data: Any, path: str) -> Any:
        """
        Extract field from data using path
        
        Examples:
            'entities' -> data.entities
            'entities[*]' -> [item for item in data.entities]
            'entities[*].text' -> [item.text for item in data.entities]
            'entities[*][*].text' -> [[e.text for e in group] for group in data.entities]
        """
        parts = path.split('.')
        current = data
        
        for part in parts:
            if '[' in part:
                current = FieldResolver._handle_brackets(current, part)
            else:
                current = FieldResolver._get_attr(current, part)
            
            if current is None:
                return None
        
        return current
    
    @staticmethod
    def _handle_brackets(data: Any, part: str) -> Any:
        """Handle parts with brackets like 'entities[*]' or 'items[0]' or '[*][*]'"""
        if part.startswith('['):
            field_name = None
            brackets = part
        else:
            bracket_idx = part.index('[')
            field_name = part[:bracket_idx]
            brackets = part[bracket_idx:]
        
        current = data
        if field_name:
            current = FieldResolver._get_attr(current, field_name)
        
        if current is None:
            return None
        
        while '[' in brackets:
            start = brackets.index('[')
            end = brackets.index(']')
            content = brackets[start+1:end]
            brackets = brackets[end+1:]
            
            if content == '*':
                if not isinstance(current, list):
                    current = [current]
            elif ':' in content:
                parts = content.split(':')
                s = int(parts[0]) if parts[0] else None
                e = int(parts[1]) if parts[1] else None
                current = current[s:e]
            else:
                idx = int(content)
                current = current[idx]
        
        return current
    
    @staticmethod
    def _get_attr(data: Any, field: str) -> Any:
        """Get attribute from data (works with dict, object, list)"""
        if isinstance(data, list):
            return [FieldResolver._get_attr(item, field) for item in data]
        
        if isinstance(data, dict):
            return data.get(field)
        
        return getattr(data, field, None)
    
    @staticmethod
    def _format_template(data: Union[List[Any], Any], template: str) -> Union[List[str], str]:
        """Format data using template"""
        if isinstance(data, list):
            results = []
            for item in data:
                try:
                    if hasattr(item, 'dict'):
                        results.append(template.format(**item.dict()))
                    elif isinstance(item, dict):
                        results.append(template.format(**item))
                    else:
                        results.append(str(item))
                except:
                    results.append(str(item))
            return results
        else:
            try:
                if hasattr(data, 'dict'):
                    return template.format(**data.dict())
                elif isinstance(data, dict):
                    return template.format(**data)
                else:
                    return str(data)
            except:
                return str(data)
    
    @staticmethod
    def _apply_reduce(data: List[Any], mode: str) -> Any:
        """Reduce list based on mode"""
        if mode == "first":
            return data[0] if data else None
        
        elif mode == "last":
            return data[-1] if data else None
        
        elif mode == "flatten":
            def flatten(lst):
                result = []
                for item in lst:
                    if isinstance(item, list):
                        result.extend(flatten(item))
                    else:
                        result.append(item)
                return result
            
            return flatten(data)
        
        return data
    
    @staticmethod
    def _apply_filter(data: List[Any], filter_expr: str) -> List[Any]:
        """Filter list based on expression"""
        if not isinstance(data, list):
            return data
        
        pattern = r'(\w+)\s*(>=|<=|>|<|==|!=)\s*(.+)'
        match = re.match(pattern, filter_expr)
        
        if not match:
            return data
        
        field, operator, value = match.groups()
        
        try:
            if value.startswith("'") or value.startswith('"'):
                value = value.strip("'\"")
            elif '.' in value:
                value = float(value)
            else:
                value = int(value)
        except:
            pass
        
        result = []
        for item in data:
            try:
                if isinstance(item, dict):
                    item_value = item.get(field)
                else:
                    item_value = getattr(item, field, None)
                
                if item_value is None:
                    continue
                
                passes = False
                if operator == '>':
                    passes = item_value > value
                elif operator == '>=':
                    passes = item_value >= value
                elif operator == '<':
                    passes = item_value < value
                elif operator == '<=':
                    passes = item_value <= value
                elif operator == '==':
                    passes = item_value == value
                elif operator == '!=':
                    passes = item_value != value
                
                if passes:
                    result.append(item)
            except:
                continue
        
        return result


# ============================================================================
# DAG EXECUTOR
# ============================================================================

class DAGPipeline(BaseModel):
    """DAG pipeline configuration"""
    name: str = Field(...)
    nodes: List[PipeNode] = Field(...)
    description: Optional[str] = None


class DAGExecutor:
    """Executes DAG pipeline with topological sort"""
    
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
        """Build dependency graph from nodes"""
        for node in self.pipeline.nodes:
            self.nodes_map[node.id] = node
        
        for node in self.pipeline.nodes:
            # Explicit dependencies
            for dep_id in node.requires:
                self.dependency_graph[dep_id].append(node.id)
                self.reverse_graph[node.id].add(dep_id)
            
            # Implicit dependencies from inputs
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
        """Initialize all processors once"""
        from .registry import processor_registry
        
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
        """Topological sort with level grouping"""
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
        """Execute full pipeline"""
        context = PipeContext(pipeline_input)
        execution_levels = self._topological_sort()
        
        if self.verbose:
            logger.info(f"Executing pipeline: {self.pipeline.name}")
            logger.info(f"Total nodes: {len(self.nodes_map)}")
            logger.info(f"Execution levels: {len(execution_levels)}")
        
        # TODO: add parallel execution with ThreadPoolExecutor
        # for levels with multiple nodes, they can run in parallel
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
        """Execute single node"""
        node = self.nodes_map[node_id]
        
        if self.verbose:
            logger.info(f"\nExecuting: {node.id} (processor: {node.processor})")
        
        if node.condition and not self._evaluate_condition(node.condition, context):
            if self.verbose:
                logger.info(f"  Skipped (condition not met)")
            return
        
        # Resolve inputs
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
        
        # Get cached processor
        processor = self.processors[node_id]
        
        # Apply schema if needed
        if node.schema and hasattr(processor, 'schema'):
            processor.schema = node.schema
        
        # Execute processor
        try:
            result = processor(**kwargs)
            
            if self.verbose:
                logger.info(f"  Processing...")
        except Exception as e:
            if self.verbose:
                logger.error(f"  Failed: {e}")
            raise RuntimeError(f"Node '{node_id}' failed: {e}")
        
        # Extract output fields if specified
        if node.output.fields:
            result = FieldResolver._extract_fields(result, node.output.fields)
        
        # Store output
        context.set(node.output.key, result)
        
        if self.verbose:
            logger.info(f"  Output: '{node.output.key}'")
            logger.info(f"  Success")
    
    def _evaluate_condition(self, condition: str, context: PipeContext) -> bool:
        """Evaluate conditional expression"""
        return True