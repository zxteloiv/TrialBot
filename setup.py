import setuptools
import trialbot

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trialbot",
    version=trialbot.__version__,
    author="zxteloiv",
    author_email="zxteloiv@gmail.com",
    description="A lightweight training framework for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zxteloiv/trialbot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
