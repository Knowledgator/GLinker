from typing import Any, Union, List
import re
from .pipe_context import PipeContext
from .input_config import InputConfig


class FieldResolver:
    @staticmethod
    def resolve(context: 'PipeContext', config: 'InputConfig') -> Any:
        from .input_config import InputConfig
        from .pipe_context import PipeContext
        
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