from setuptools import setup, find_packages

setup(
    name="hdr",
    version="0.1.0",
    description="Harness Done Right - Task formalization and execution framework",
    packages=find_packages(),
    py_modules=["hdr"],
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "locache>=0.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
