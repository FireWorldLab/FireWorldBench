#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 FireWorldLab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup

setup(
    name="fireworldbench",
    version="0.1.0",
    description="FireWorldBench: a multimodal benchmark for fire-physics world understanding",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="FireWorldLab",
    project_urls={
        "Dataset": "https://huggingface.co/datasets/Guaogua/FireWorldBench",
        "Source": "https://github.com/FireWorldLab/FireWorldBench",
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["requests>=2.31", "huggingface_hub>=0.23"],
)
