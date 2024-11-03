from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mri-tumor",
    version="0.1.0",
    author="Justin-Marian Popescu",
    author_email="pmarianjustin@gmail.com",
    description="--- LATER TO BE ADDED ---",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justin-marian/mri-tumor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "requests>=2.25.0",
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
        "matplotlib>=3.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
