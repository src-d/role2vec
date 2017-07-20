from setuptools import setup, find_packages

setup(
    name="role2vec",
    description="Part of source{d}'s stack for machine learning on source "
                "code. Provides API and tools to train and use models for "
                "embedding AST nodes.",
    version="1.0.0",
    license="Apache 2.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/role2vec",
    download_url='https://github.com/src-d/role2vec',
    packages=find_packages(exclude=("role2vec.tests",)),
    keywords=["machine learning on source code", "github", "embedding",
              "word2vec", "AST"],
    install_requires=["ast2vec>=1.0.0", ],
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 4 - Beta",
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
