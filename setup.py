"""Setuptools shim.

Package metadata and dependencies live in pyproject.toml; this file
exists only so that legacy ``setup.py`` based installs still work.
"""

from setuptools import setup

setup()
