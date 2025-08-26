# TracePredicate: Medical Device Regulatory Lineage Analysis System

## Overview
A comprehensive research platform to analyze FDA 510(k) medical device regulatory lineages, calculate Lineage Drift Index (LDI), and correlate with real-world risk metrics. Currently focused on hip implants (KWA) as the primary case study.

## Key Features
- **FDA 510(k) Data Collection**: Automated scraping and parsing of regulatory documents
- **MAUDE Database Integration**: Adverse event analysis with Human Factors Engineering classification
- **Network Analysis**: Directed graph construction of device predicate relationships
- **LDI Calculation**: Mathematically optimized risk scoring system
- **Statistical Analysis**: Correlation analysis between LDI and real-world risk metrics
- **Interactive Visualization**: Web-based platform for exploring device lineages

## Project Structure
```
TracePredicate/
├── trace_predicate/           # Main Python package
│   ├── data_collection/       # FDA data scraping modules
│   ├── llm_extraction/        # LLM-assisted parameter extraction
│   ├── network_analysis/      # Graph construction and analysis
│   ├── ldi_engine/           # LDI calculation system
│   ├── optimization/         # Weight optimization algorithms
│   ├── statistical_analysis/ # Hypothesis testing and correlations
│   ├── database/             # Database models and operations
│   ├── api/                  # FastAPI endpoints
│   └── utils/                # Common utilities
├── frontend/                 # Web interface
├── tests/                    # Test suites
├── config/                   # Configuration files
├── data/                     # Data storage
├── docs/                     # Documentation
└── notebooks/                # Jupyter analysis notebooks
```

## Installation

1. Install Poetry (if not already installed):
```bash
pip install poetry
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Quick Start

```bash
# Initialize the database
trace-predicate init-db

# Start data collection for hip implants
trace-predicate collect --device-code KWA --limit 50

# Calculate LDI scores
trace-predicate calculate-ldi --device-family hip-implants

# Start web interface
trace-predicate serve
```

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black .

# Type checking
poetry run mypy .

# Start development server
poetry run uvicorn trace_predicate.api.main:app --reload
```

## License
Research use only. Please cite appropriately if used in academic work.