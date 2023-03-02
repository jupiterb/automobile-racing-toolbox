from setuptools import setup, find_packages

requirements_common = ["pytest", "numpy", "Pillow", "pynput"]

windows_requirements = ["pywin32"]

import sys

requirements = requirements_common

if "linux" not in sys.platform:
    requirements += windows_requirements

setup(
    name="Automobile Training Common Package",
    version="0.1.1",
    description="Common packages for various applications in Automobile Racing Toolbox",
    packages=find_packages(),
    install_requires=requirements,
)
