"""Setup script for pure_transformer package."""

from setuptools import setup, find_packages

setup(
    name="pure_transformer",
    version="0.1.0",
    description="Pure Transformer LLM with SOTA RL Training (GRPO/ProRL)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
)
