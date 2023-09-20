import importlib


def dynamically_import_foo(path: str):
    module_str, foo = path.rsplit(".", 1)
    foo = foo.split("::")
    obj = getattr(importlib.import_module(module_str), foo[0])
    if len(foo) == 2:
        obj = getattr(obj, foo[1])
    return obj
