from setuptools import setup, find_packages

setup(
    name="quercus",
    version="0.8.0",
    description="Quantitative Unsupervised Extraction & Remote Classification of US Species",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="QUERCUS Project",
    url="https://github.com/MayerT1/quercus",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.28",
        "Pillow>=9.0",
        "numpy>=1.23",
        "opencv-python-headless>=4.7",
        "rasterio>=1.3",
        "pyproj>=3.4",
        "shapely>=2.0",
        "geopandas>=0.13",
        "scikit-learn>=1.2",
        "pandas>=1.5",
        "matplotlib>=3.6",
        "tqdm>=4.65",
        "earthengine-api>=0.1.370",
        "folium>=0.14",
    ],
    extras_require={
        "sam": ["segment-geospatial"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
