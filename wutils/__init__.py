import pkgutil
import importlib

__all__ = [] # names of imported functions
registry = {} # dictionary to track modules of functions

# Check all modules in package
for module_info in pkgutil.iter_modules(__path__):
    module_name = module_info.name
    module = importlib.import_module(f".{module_name}", package=__name__)

    # Import functions from module
    for attr_name in dir(module):
        attr = getattr(module, attr_name)

        if callable(attr) and attr.__module__ == module.__name__:

            # Rename "main" to module name
            if attr_name == "main":
                attr_name = f"{module_name}"

            # Check for duplicate functions
            if attr_name in registry:
                raise ImportError(
                    f"Conflict: Function '{attr_name}' in"
                    f" module '{module_name}' already exists in"
                    f" module '{registry[attr_name]}'."
                )

            # Register function name and add to package namespace
            registry[attr_name] = module_name
            globals()[attr_name] = attr
            __all__.append(attr_name) # wildcard importable