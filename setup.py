#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("offpolicy", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


setup(
    name="offpolicy",  # Replace with your own username
    version=get_version(),
    description="GLQ and off-policy algorithms of marlbenchmark",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author='kideng',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="multi-agent reinforcement learning platform pytorch",
    python_requires='>=3.6',
    install_requires=[
        "wandb",
        "torch",
        "gym==0.11",
        "seaborn",
        "tensorboardX",
        "absl-py",
        "pyglet==1.5.27"
    ]
)
