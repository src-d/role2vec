import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 5, 0):
    typing = ["typing"]
else:
    typing = []

setup(
    name="role2vec",
    description="Part of source{d}'s stack for machine learning on source code. Provides API and "
                "tools to train and use models for role prediction of UAST nodes extracted from "
                "Babelfish.",
    version="0.0.1-alpha",
    license="Apache 2.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/role2vec",
    download_url="https://github.com/src-d/role2vec",
    packages=find_packages(exclude=("role2vec.tests",)),
    entry_points={
        "console_scripts": ["role2vec=role2vec.__main__:main"],
    },
    keywords=["machine learning on source code", "word2vec", "id2vec",
              "github", "swivel", "nbow", "bblfsh", "babelfish"],
    install_requires=["ast2vec[tf]>=0.3.4-alpha"] + typing,
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries"
    ]
)
