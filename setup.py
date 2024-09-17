from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="SplitFXM",
    version="0.3.4",
    description="1D Finite-Difference/Volume Split Newton Solver",
    url="https://github.com/gpavanb1/SplitFXM",
    author="gpavanb1",
    author_email="gpavanb@gmail.com",
    license="CC BY-NC 4.0 for non-commercial use, commercial license available",
    packages=["splitfxm", "splitfxm.equations", "splitfxm.models"],
    install_requires=["numpy", "numdifftools", "matplotlib", "splitnewton"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="amr newton python finite-difference armijo optimization pseudotransient splitting",
    project_urls={  # Optional
        "Bug Reports": "https://github.com/gpavanb1/SplitFXM/issues",
        "Source": "https://github.com/gpavanb1/SplitFXM/",
    },
    zip_safe=False,
)
