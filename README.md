# wutils

Useful scripts with common usage across multiple projects.

---

## Setup and Installation

### Getting started

**Clone the repository**
~~~bash
	git clone https://github.com/wuvin/wutils
~~~

**Create environment to install package (optional)**
~~~bash
	python -m venv .venv
	source .venv/bin/activate # if on Ubuntu; on Windows, .venv\Scripts\activate
~~~

**Install package and dependencies**  
(*Note*: [setup.py](setup.py) is configured to read [requirements.txt](requirements.txt))
~~~bash
	pip install .
~~~

---

## Help for Creating Python Packages

A way to modularize code development is to turn a collection of similar Python scripts into a package from which another project can import functions.  Below is a simple guide.

### (1) Organize into a package structure

The *root directory* should follow a standard structure:

```text
	wutils/
	├── __init__.py
	├── example_module_my_functions.py
	└── example_module_other_functions.py
	setup.py
	requirements.txt
	README.md
	LICENSE
```

- `wutils/`: This is the package directory, which is distinct from the *root directory*. The name of the *root directory* is not so important -- it can be the same as the package -- but the name of the package directory **must be** the name used to import the package.
- `__init__.py`:  This is a required file that indicates `wutils` to be treated as a package.  It can be empty, but it has other optional uses:
	- Initialize package-level data.
	- Define what gets imported by `from wutils import *`.
	- Make certain functions or sub-modules directly accessible at package level.
- `example_module_my_functions.py` and `example_module_other_functions.py`: These are the modules belonging to the package. Each script containing functions becomes a module.
- `setup.py`: Dependency installation is managed by a specially-named executable or setup file such as this.  The modern standard is `pyproject.toml`, which is another option, but it is fine to use `setup.py`.
- `requirements.txt`: This is the blueprint file that `pip` uses to install project dependencies within an environment.  There are alternatives; `conda` environments can use `environment.yml` or `environment.yaml`.

### (2) Make package installable

Technically optional, but doing so is standard for managing dependencies when sharing development.  This involves adding a `setup.py` file or similar.  With `setup.py`, another user can install the package in a new virtual environment by executing `pip install .` within the root directory.

During development, if you want any changes to the source files in `wutils` to be immediately reflected in the installed package without needing to reinstall, then you can install in editable mode: `pip install -e .`.

### (3) Specify dependencies

Within an existing environment, *all* dependencies can be exported by executing:
~~~bash
	pip freeze > requirements.txt
~~~

This is simple and quick, but it does list *all* packages of the active environment, not just those specific to the project.  A new installation may capture unnecessary dependencies.



## Example of `pyproject.toml`:

```toml
	# pyproject.toml
	[build-system]
	requires = ["setuptools>=61.0"]
	build-backend = "setuptools.build_meta"

	[project]
	name = "wutils"
	version = "0.1.0"
	description = "A collection of broadly useful Python functions"
	authors = [
	  {name = "Your Name", email = "your.email@example.com"},
	]
	readme = "README.md"
	requires-python = ">=3.8"
	license = { file = "LICENSE" }
	keywords = ["utilities", "math", "strings"]
	classifiers = [
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: Ubuntu 22.04",
	]

	[project.urls]
	Homepage = "https://github.com/wuvin/wutils"
	Repository = "https://github.com/wuvin/wutils.git"
```

---

## Authors

Kevin Wu  
wu.kevi@northeastern.edu

## Helpful Links

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)