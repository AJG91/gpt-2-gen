from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1",
    author="Alberto J. Garcia",
    description="",
    long_description=open("README.md").read(),
    packages=find_packages(
        where="src",
        include=["*"],
        exclude=["tests*", "docs*"]
    ),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    zip_safe=False
)