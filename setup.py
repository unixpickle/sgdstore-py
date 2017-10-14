"""
Setup script for installing this package.
"""

from os import path

import codecs
from setuptools import setup, find_packages

README_PATH = path.join(path.abspath(path.dirname(__file__)), 'README.rst')
with codecs.open(README_PATH, encoding='utf-8') as readme:
    LONG_DESCRIPTION = readme.read()

setup(
    name='sgdstore',
    version='0.0.1',
    description='Memory-augmented RNN with live SGD',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/unixpickle/sgdstore-py',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ai rnn learning memory sgd backpropagation',

    packages=find_packages(exclude=['examples']),
    install_requires=[
        'numpy>=1.0.0,<2.0.0',
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"]
    }
)
