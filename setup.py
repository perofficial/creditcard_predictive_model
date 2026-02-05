from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="credit-scoring-model",
    version="1.0.0",
    author="Pro National Bank - Data Science Team",
    author_email="datascience@pronationalbank.com",
    description="Credit Scoring Model per valutazione affidabilità creditizia",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/credit-scoring-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "credit-train=scripts.train_model:main",
            "credit-evaluate=scripts.evaluate_model:main",
            "credit-predict=scripts.predict:main",
        ],
    },
)
