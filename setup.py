import setuptools
import os

# Read README.md if it exists, otherwise use a default description
try:
    with open("README.md", "r") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "AI-powered supply chain optimization features using machine learning models trained on real-world supply chain data"

__version__ = '0.0.0'

REPO_NAME = "Supply_Chain_Ai"
AUTHOR_USER_NAME = "NaveenSa98"
SRC_REPO = "Supply_Chain_Ai"
AUTHOR_EMAIL = "n.t.de.a.samaranayake@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="AI-powered supply chain optimization features using machine learning models trained on real-world supply chain data",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)