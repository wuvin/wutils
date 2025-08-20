from modulefinder import ModuleFinder
import sys

def get_imported_packages(script_path):
    """
    Extracts and returns a set of packages imported by a Python script.

    Args:
        script_path (str): The path to the Python script.

    Returns:
        set: A set of strings, where each string is the name of a package
             imported by the script.
    """
    finder = ModuleFinder()
    try:
      print('Shouldn''t this error out?')
      finder.run_script(script_path)
    except FileNotFoundError:
      print('This errored out.')
      print(f"Error: Script not found at '{script_path}'.")
      return set()
    
    print('Loaded modules:')
    for name, mod in finder.modules.items():
       print('%s: ' % name, end='')
       print(','.join(list(mod.globalnames.keys())[:3]))

    imported_packages = {module.__name__.split('.')[0]
                         for name, module in finder.modules.items() 
                         if module.__name__ is not None}
    return imported_packages

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script.py <script_path>")
    else:
        script_path = sys.argv[1]
        packages = get_imported_packages(script_path)
        if packages:
          print("Imported top-level packages:")
          for package in packages:
              print(f"- {package}")