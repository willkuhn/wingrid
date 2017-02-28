# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='wingrid',
    version='0.1.0',
    description='Quantitatively compare appearances of insect wings',
    long_description=readme,
    author='William R. Kuhn',
    author_email='willkuhn@crossveins.com',
    url='https://github.com/willkuhn/wingrid',
    license=license,
    packages=find_packages()
)
