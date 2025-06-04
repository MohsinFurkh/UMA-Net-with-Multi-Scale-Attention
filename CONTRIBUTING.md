# Contributing to UMA-Net

First off, thank you for considering contributing to UMA-Net! We welcome all contributions, including bug reports, bug fixes, documentation improvements, and feature requests.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the [issue tracker](https://github.com/MohsinFurkh/UMA-Net-with-Multi-Scale-Attention/issues) as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When you are creating an enhancement suggestion, please include as many details as possible.

### Your First Code Contribution

1. Fork the repository.
2. Create a new branch with a descriptive name: `git checkout -b feature/amazing-feature` or `bugfix/fix-issue-123`
3. Make your changes and commit them: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

### Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations, and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. The pull request will be reviewed by the maintainers.

## Development Environment Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Before committing your changes, please run:
```bash
black .
isort .
flake8
mypy .
```

## Testing

Run the test suite with:
```bash
pytest
```

## Documentation

To build the documentation:
```bash
cd docs
make html
```

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
