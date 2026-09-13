import json
import typing
import types
import threading
from dataclasses import is_dataclass

_parser_cache = {}

class ConversionCache(threading.local):
    def __init__(self):
        self.cache = {}

_conversion_cache = ConversionCache()

def clear_conversion_cache():
    """Clear the thread-local Observation conversion cache."""
    _conversion_cache.cache.clear()

def get_parser(cls):
    if cls in _parser_cache:
        return _parser_cache[cls]
        
    field_parsers = {}
    
    for name, f in cls.__dataclass_fields__.items():
        t = f.type
        # Extract underlying type if it's Union / Optional
        origin = typing.get_origin(t)
        args = typing.get_args(t)
        
        # Check for UnionType / Union
        if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                t = non_none[0]
                origin = typing.get_origin(t)
                args = typing.get_args(t)
                
        # Now detect base type
        if is_dataclass(t):
            # It's a nested dataclass
            field_parsers[name] = ('dataclass', t)
        elif origin is list:
            item_type = args[0] if args else None
            # Item type could be Union (e.g. Pokemon | None)
            item_origin = typing.get_origin(item_type)
            item_args = typing.get_args(item_type)
            if item_origin is typing.Union or (hasattr(types, "UnionType") and item_origin is types.UnionType):
                item_non_none = [a for a in item_args if a is not type(None)]
                if item_non_none:
                    item_type = item_non_none[0]
            
            if is_dataclass(item_type):
                field_parsers[name] = ('list_dataclass', item_type)
            else:
                field_parsers[name] = ('primitive', None)
        else:
            field_parsers[name] = ('primitive', None)
            
    _parser_cache[cls] = field_parsers
    return field_parsers


def to_dataclass(dic: dict, cls: type):
    """
    Convert a dictionary to a dataclass instance recursively.
    Highly optimized for performance.
    """
    if dic is None:
        return None

    parsers = get_parser(cls)
    d = {}
    
    for key, value in dic.items():
        if key in parsers:
            p_type, p_cls = parsers[key]
            if value is None:
                d[key] = None
            elif p_type == 'dataclass':
                d[key] = to_dataclass(value, p_cls)
            elif p_type == 'list_dataclass':
                d[key] = [to_dataclass(v, p_cls) for v in value]
            else:
                d[key] = value

    return cls(**d)


def json_to_dataclass(bs: bytes, cls: type):
    """
    Convert a JSON byte string to a dataclass instance.
    """
    js = bs.decode()
    dic = json.loads(js)
    return to_dataclass(dic, cls)
