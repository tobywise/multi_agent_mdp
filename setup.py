from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'readme.md').read_text(encoding='utf-8')

setup(
    name='maMDP',  
    version='0.2',  
    description='A package for multi-agent MDPs',  
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/tobywise/multi_agent_mdp',  
    author='Toby Wise',  
    classifiers=[ 
        'Development Status :: 3 - Alpha',
        'License :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3.6, <4',
    packages=find_packages(),
    install_requires=['numpy', 'numba', 'matplotlib', 'scipy', 'fastprogress'], 
    extras_require={  
        'test': ['pytest'],
    },
)