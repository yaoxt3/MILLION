from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'gym==0.21.0',
    'mujoco-py==2.1.2.14',
    'numpy==1.21.3',
    'tensorboard',
    'GPUtil',
    'torch>=1.7'
]


setup(
    name='learning_to_be_taught',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
