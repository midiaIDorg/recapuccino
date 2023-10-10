import importlib


def dynamically_import_foo(path: str, _object_sep: str = "::"):
    module_str, foo = path.split(_object_sep)
    return getattr(importlib.import_module(module_str), foo)
