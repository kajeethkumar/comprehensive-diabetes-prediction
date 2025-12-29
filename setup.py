from setuptools import setup, find_packages

setup(
    name="diabetes-ml-pipeline",
    version="1.0.0",
    description="Unified ML pipeline for diabetes classification and regression with feature selection",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.5",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "aeon>=0.8.0",
        "joblib>=1.3",
        "tqdm>=4.66",
    ],
)
