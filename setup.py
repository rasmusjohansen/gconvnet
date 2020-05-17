import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gconvnet", # Replace with your own username
    version="0.0.1",
    author="rasmusjohansen",
    author_email="rasnjo@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rasmusjohansen/gconvnet",
    packages=setuptools.find_packages('src'),
    package_dir = {'' : 'src'},
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6'
)