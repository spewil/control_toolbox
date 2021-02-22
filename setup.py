from setuptools import setup, find_namespace_packages

setup(
    name='ofc',
    version='1.0',
    description='Optimal Feedback Control in Python',
    author='Spencer Wilson',
    author_email='spencer@spewil.com',
    #   url='https://www.python.org/sigs/distutils-sig/',
    packages=find_namespace_packages(exclude=["tests"]),
)
