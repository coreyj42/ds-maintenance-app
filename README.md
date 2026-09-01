#   DATA SCIENCE
##  MAINTENANCE APP

- This repository contains the pipelines and functions to predict future maintenance events.
- It relies heavily on ds-package-forecast for core forecasting functions.
- The pipelines are run in Airflow.

### PROJECT INIT

#### CI / CD
- CD example for GCP Artifact Registry is available in .github/workflows

## PROJECT REQUIREMENTS

####    Poetry installation
- Install Poetry: https://python-poetry.org/docs/
- Windows: 
    - Need to install scoop first: https://scoop.sh/

####    Poetry documentation
- You do not need to setp-up because this project backbone already exists. However, it is important to understand the below key concepts 
- Follow this documentation: https://python-poetry.org/docs/basic-usage/
- Important parts are:
    - Project setup
    - Setting a Python version
    - Operationg modes
    - Specifying dependencies
    - Using poetry run 
    - Installing dependencies
- Other parts can read later

####    Poetry install
- Run `poetry install` to create a dedicated virtual environment and install the base dependencies

### README
- Writing your README matters.

### TESTS
- Pytest is the library to be used for unit-testing: https://docs.pytest.org/en/stable/
- Testing matters, please write unit-tests frequently to detect bugs early

### LINT
- Flake8 is used for Linting: https://flake8.pycqa.org/en/latest/
- Please, do not change the .flake8 configuration so it's common to all of our projects

### FORMAT
- Black is used for formatting the codebase: https://black.readthedocs.io/en/stable/index.html
- That ensures a common coding style across projects
