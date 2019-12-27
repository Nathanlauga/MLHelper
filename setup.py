from setuptools import setup, find_packages

long_description = """
Python tool to help creating a model.
"""

setup(
    name='MLHelper',
    version='1.0.0',
    description="Python tool to help creating a model.",
    license='MIT',
    author='Nathan LAUGA',
    author_email='nathan.lauga@protonmail.com',
    # url='https://github.com/Nathanlauga/transparentai',
    packages=['MLHelper'],
    include_package_data=True,
    install_requires=[
        'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'ipython'
    ],
    long_description=long_description,
    python_requires='>=3.5'
)
