from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="ioptics",
    version="0.1.0",
    description="Integrated optical neural network simulation and training framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["ioptics", "ioptics.*"]),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.7.1,<3",
        "torchvision>=0.23.0",
        "matplotlib>=3.10.7,<4",
        "numpy>=2.3.3,<3",
        "scikit-learn>=1.7.2,<2",
        "pandas>=2.3.3,<3",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0.2,<10",
            "jupyterlab>=4.4.10,<5",
        ],
    },
    include_package_data=True,
)
