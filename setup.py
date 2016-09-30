from setuptools import setup, find_packages

setup(
    name='SynthPop',
    version='0.2dev',
    description='Population Synthesis',
    author='Autodesk',
    author_email='udst@autodesk.com',
    license='BSD',
    url='https://github.com/udst/synthpop',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'numexpr>=2.3.1',
        'numpy>=1.8.0',
        'pandas>=0.13.1',
        'scipy>=0.13.3',
        #'census>=0.5',
        #'us>=0.8'
    ]
)

# tested with:
#import setuptools      # ubuntu (16.04) package
#setuptools.__version__ '20.7.0'

#numexpr.__version__ '2.4.3'
#np.__version__ '1.11.0'
#pd.__version__ '0.18.1'
#scipy.__version__ '0.17.0'

#import us
#us.__version__ '0.9.1'
#import census
#census.__version__ '0.7'

