# This Python file uses the following encoding: utf-8
import glob

from setuptools import find_packages, setup

setup(
    name="recapuccino",
    packages=find_packages(),
    version="0.0.1",
    description="Perform various types of recalibrations in mass spectrometry.",
    long_description="Perform various types of recalibrations in mass spectrometry (if only: `ceil` is the limit)",
    author="Matteo Lacki & David Teschner",
    author_email="matteo.lacki@gmail.com",
    url="https://github.com/MatteoLacki/recapuccino.git",
    keywords=[
        "M/Z recalibration",
        "Mass spectrometry recalibrations.",
        "Mokapot",
        "Percolator",
        "Rock'n'Roll",
        "Drugs",
        "Linux",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=[
        "duckdb",  # importing data with SQLs
        "numpy",
        "numba",
        "pandas",
        "pandas_ops",  # some custom shit around pandas
        "patsy",
        "matplotlib",
        "sklearn",
        "seaborn",  # just for plotting: might make it optional later on
    ],
    scripts=glob.glob("tools/*.py"),
)
