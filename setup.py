from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

setup(
    name='SynthPop',
    version='0.1dev',
    description='Population Synthesis',
    author='UrbanSim Inc.',
    author_email='udst@urbansim.com',
    license='BSD',
    url='https://github.com/udst/synthpop',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'census>=0.5',
        'numexpr>=2.3.1',
        'numpy>=1.8.0',
        'pandas>=0.13.1',
        'scipy>=0.13.3',
        'us>=0.8'
    ]
)
