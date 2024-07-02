from setuptools import setup, find_packages

setup(
    name="CLAgent",
    version="0.1.0",
    author="Edan Meyer",
    author_email="ejmejm98@gmail.com",
    description="An efficient continual learning agent built with JAX.",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.9',
)