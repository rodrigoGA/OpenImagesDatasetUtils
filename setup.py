from setuptools import find_packages, setup
setup(
    name='openimagesdatasetutils',
    packages=find_packages(),
    version='0.0.1',
    description='Utils para trabajar con openimagedataset',
    author='Rodrigo',
    license='MIT',
    setup_requires=[ 'numpy>=1.0.0', 'pandas>=1.1.0', 'pillow>=8.0.0'],
)