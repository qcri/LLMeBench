# Contribution Guidelines for **LLMeBench**

## Introduction
Welcome to the _LLMeBench_ project! We are excited to have you here. This document outlines the process and guidelines for contributing to this project. We welcome all forms of contribution: code, documentation, bug reports, feature requests, and more.

## Prerequisites
Before you begin, please ensure you have the following installed:

- Git
- Python (>=3.8)


## How to Contribute
1. **Fork the Repository:** Fork the repository to your own GitHub account.
2. **Clone the Fork:** Clone your fork locally on your computer.
3. **Set up environment:** Follow the instructions in the [README](https://github.com/qcri/LLMeBench#installation) to set up your development environment.
4. **Create a new branch:** Create a new branch for your contribution.
5. **Make changes:** Implement your changes, additions, or fixes.
6. **Test:** Run tests to ensure your changes do not break existing functionality.
7. **Submit a pull request:** Push your changes to your fork on GitHub and then create a pull request for review.

## Code Style Guidelines
Please adhere to the PEP 8 style guide for Python code. Use meaningful variable names, add comments, and organize your code well.

## Adding New Dataset, Task, Models or Features
If you are adding [new dataset, task, models or features](tutorial), please ensure:

- The code is well-documented.
- You have added corresponding tests.
- Update the **README** to reflect any new changes or requirements.

## Reporting Issues
Please use the GitHub issue tracker to report bugs or request features. Be as descriptive as possible and include any relevant code snippets, error messages, or screenshots.

## Testing
Before submitting your changes, please make sure to run all the relevant scripts. This often involves running:

- Format the code
```
bash scripts/format_code.sh
```
- Run the tests
```
bash scripts/run_tests.sh
```
For a more detail: https://github.com/qcri/LLMeBench#testing


## Documentation
Documentation is crucial for understanding the project's code and for users to effectively use the package. If you make changes that require updates to the documentation, please also include those changes in your pull request.

## Community
For general questions or discussions, please use [discord](). Feel free to reach out for any questions or clarifications.

## Review Process
TBA

## Recognition
Contributors names will be listed under the [Contributors page]()


## Versioning
The _LLMeBench_ package adheres to [semantic versioning](https://semver.org/). This system helps us to track changes in the package and communicate the nature of those changes to our users and contributors. In essence, version numbers are in the form of _MAJOR.MINOR.PATCH_.

- _MAJOR_ version: Incremented when there are incompatible API changes.
- _MINOR_ version: Incremented when new features are added in a backward-compatible manner.
- _PATCH_ version: Incremented when backward-compatible bug fixes are introduced.

### Branch Naming
To contribute to a specific version, please name your branch according to the version you are targeting. For example, if you are contributing to version 1.0, name your branch _feature/1.0/your-feature_ or _fix/1.0/your-fix_.

For example, to add ArSAS sentiment dataset,
```
git checkout -b feature/1.0/ArSAS_sentiment
```


<!-- ### Release cycle
We aim for a stable release every quarter, with occasional minor releases in between for urgent bug fixes or smaller feature additions. -->

<!-- ### Changelog
All changes will be documented in the CHANGELOG.md file in the root of the repository. If you contribute a feature, bug fix, or other changes, please also add a corresponding entry to the changelog. -->

<!-- ### Deprecation policy -->

## Troubleshooting
This section provides guidance for resolving common issues you might encounter while contributing to the _LLMeBench_ package. If you run into a problem not covered here, please open an issue in the GitHub repository.

### Installation related
- **Installation errors:** If you encounter errors during the [installation](https://github.com/qcri/LLMeBench/#installation) process, make sure you are using a compatible Python version.

- **Environment conflicts:** If you experience dependency conflicts, consider using a virtual environment to isolate the project's dependencies.

### Git and branching
- **Branch conflicts:** If you have conflicts after pulling the latest changes from the main branch, you may need to resolve the conflicts manually or consider rebasing your branch.

- **Failed push:** If you are unable to push your changes to GitHub, double-check your remote settings and that you have the correct permissions.

<!-- ### Code style and linting -->

## Example Contributions
We welcome contributions of all types, however, we are particularly interested in those that align with the current needs and goals of the _LLMeBench_ framework. Below are some example contributions that would be especially valuable:

### Adding new datasets
- **Datasets:** Contribute commonly-used datasets in the field of large language models for benchmarking.
  - Update the dataset loading scripts to include your new dataset.
  - Provide proper citations and documentation for the dataset.
  <!-- - Include sample tests to verify the dataset works as expected. -->

### Adding new languages
- **Under-represented languages:** Add support for languages that are not well-represented in current large language models.
- **Language variants:** Add support for regional variants or dialects.


### Adding new models
State-of-the-Art Models: Contribute implementations or interfaces for state-of-the-art language models (e.g., LLaMA, Vecuna).


_Please ensure that your contributions align with the existing code style guidelines and that you update any relevant documentation and tests._

## Security and Sensitive Information
It is crucial to ensure that no sensitive information, especially API keys, is included in any of your contributions. Inadvertently committing API keys or other sensitive information to the repository poses a significant risk.

If you are working with API keys, tokens, or other sensitive information during development, make sure to:

- **Use environment variables:** Store your API keys or other sensitive information in environment variables or use a secrets management service.
- **Git ignore:** Double-check that files containing sensitive information are listed in your .gitignore before submitting a pull request.
- **Review before commit:** Always review your changes before committing and again before submitting a pull request to ensure no sensitive information is included.

If you do accidentally commit API keys or any sensitive information, please contact the maintainers immediately to address the issue.



## Roadmap

### Short-term goals
- **Expand language support**:
  - Introduce core tasks and datasets for additional languages to diversify the framework's language capabilities.
- **Enrich dataset collection**:
  - Add a variety of datasets to enhance the framework's benchmarking scope.
- **Broaden model support**:
  - Develop a generic interface for integration with Hugging Face models, enabling more versatile benchmarking options.
- **Incorporate additional few-shot methods**:
  - Add support for various few-shot learning techniques to expand the evaluation metrics available.
- **Enable dual modes of interaction**:
  - Facilitate interaction with both online and offline models, offering users greater flexibility.
- **Multi-dataset evaluation**:
  - Allow for running a single asset across multiple datasets to provide a more comprehensive performance analysis.


### Long-term goals
- **Implement results summarization**:
  - Create a feature for automatically summarizing and comparing results from multiple assets, possibly including graphical representations.
- More coming ...

<!-- ## Code of Conduct -->


## License
By contributing, you agree that your contributions will be licensed under the same license used by the original project [MIT-LICENSE](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).
