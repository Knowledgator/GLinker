from collections import OrderedDict
from typing import Dict, Any, List, Optional


class PipeContext:
    """
    Pipeline execution context
    
    Stores all outputs from pipeline stages and provides unified access:
    - By key: "l1_result"
    - By index: "outputs[-1]" (last output)
    - Pipeline input: "$input"
    
    Replaces PipelineDict with extended functionality
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
        
        # Access by key
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
        """For compatibility with PipelineDict"""
        return dict(self._outputs)