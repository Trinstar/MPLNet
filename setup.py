import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as open_file:
    install_requires = open_file.read().splitlines()

setuptools.setup(
    name="MPLNet",
    version="1.0.0",
    url="https://github.com/Trinstar/MPLNet",
    packages=setuptools.find_packages(),
    author="Linpan Xu",
    author_email="tannyecho@gmail.com",
    description="Multi-grained Prompt Learning with Vision-Language Model for Remote Sensing Image Scene Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
)
