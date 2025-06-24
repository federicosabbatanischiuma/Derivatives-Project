from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='fin404-project',
    version='1.0.0',
    description='EPFL Fin404 Derivatives Project: The VIX and related derivatives',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Gabriele Calandrino, Alex Martinez de Francisco, Federico Sabbatani Schiuma, Letizia Seveso',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'fin404=main:main',
        ],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)
