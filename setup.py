from setuptools import setup, find_packages

setup(
    name='nqetools',
    version='0.1.0',
    author='Louie Slocombe',
    author_email='louies@hotmail.co.uk',
    description='A centralised set of tools for doing nuclear quantum calculations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LouieSlocombe/nqetools',
    packages=find_packages(include=['nqetools', 'nqetools.*']),
    package_data={
        'nqetools': [
            'templates/*',
            'thermostats/*',
            'opes/*',
        ],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'ase',
        'sella',
        'scipy',
        'ipi',
        'mace-torch',
        'plumed',
        'pyfftw',
        'chemiscope',
        'geodesic_interpolate @ git+https://github.com/LouieSlocombe/geodesic_interpolate.git',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ],
    },
)
