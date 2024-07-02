from setuptools import setup, find_packages
from typing import Tuple


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name='guidellm',
    version='0.1.0',
    author='Neuralmagic, Inc.',
    description='Guidance platform for deploying and managing large language models.',
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    license="Apache",
    url="https://github.com/neuralmagic/guidellm",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'datasets',
        'loguru',
        'numpy',
        'openai',
        'requests',
        'transformers',
        'click'
    ],
    extras_require={
        'dev': [
            'pytest',
            'sphinx',
            'ruff',
            'mypy',
            'black',
            'isort',
            'flake8',
            'pre-commit',
        ],
    },
    entry_points={
        'console_scripts': [
            'guidellm=guidellm.main:main',
        ],
    },
    python_requires=">=3.8.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
