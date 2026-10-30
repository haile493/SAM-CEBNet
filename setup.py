from setuptools import find_packages, setup

setup(
    name="sam_cebnet",
    version="1.0.0",
    description="SAM-CEBNet: Foundation-Guided Corner & Edge Refinement Network Family for High-Precision Remote Sensing Building Extraction and Change Detection",
    author="Hai Nguyen",
    author_email="hai49@alumni.cmu.edu",
    url="https://github.com/haile493/SAM-CEBNet",
    packages=find_packages(include=["ag_models*", "models*", "datasets*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "pillow>=9.5.0",
        "tifffile>=2023.4.12",
        "shapely>=2.0.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
    ],
)
