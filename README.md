# AIF-Instructions

Instructions for setting up and using the AIF project.

## Prerequisites

- Conda environment named `mkdocs` (already created)

## Setup

1. Activate the conda environment:
    ```bash
    conda activate mkdocs
    ```

2. Install required dependencies:
    ```bash
    pip install mkdocs mkdocs-material
    ```

## Usage

### Building the documentation

```bash
mkdocs build
```

### Serving locally

```bash
mkdocs serve
```

Then open your browser to `http://127.0.0.1:8000`

To serve on a custom port and host:

```bash
mkdocs serve -a 0.0.0.0:8001
```

Then open your browser to `http://0.0.0.0:8001`

### Deploying

```bash
mkdocs gh-deploy
```

## Project Structure

```
.
├── docs/
│   └── index.md
├── mkdocs.yml
└── README.md
```

