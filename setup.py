# setup.py

from setuptools import setup, find_packages

setup(
    name='transformers',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
)

# python -m unittest discover -s tests
