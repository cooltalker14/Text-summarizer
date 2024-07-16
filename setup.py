import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ ="0.0.0"

REPO_NAME = "Text-summarizer"
AUTHOR_USER_NAME = "cooltalker14"
SRC_REPO = "textSummarizer"
AUTHOR_EMAIL ="swap14544@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A simple text summarizer using GPT-2 model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/" + AUTHOR_USER_NAME + "/" + REPO_NAME,
    project_urls={"Bug Tracker": f"https://github/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",        
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
)
