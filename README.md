# Decision Aggregation under Quantal Response

## Overview
This repository contains the implementation for the paper "Decision Aggregation under Quantal Response". The project includes:

1. Numerical analysis of decision aggregation mechanisms
2. Empirical studies of decision-making behavior under quantal response models

## Study Components

### 1. Numerical Analysis
- **File:** `src/numerical.py`
- **Description:** Implements numerical simulation of decision aggregation under quantal response models. Generates synthetic data and analyzes aggregation performance.

### 2. Empirical Studies

#### Bayesian Decision-Making Study
- **Data:** `data/bayesian_study.json`
- **Analysis:** `src/bayesian_analysis.py`
- **Description:** Examines individual decision-making under Bayesian reasoning with quantal response.

#### Multiple-Choice Response Study
- **Data:** `data/multiple_choice.json`
- **Analysis:** `src/response_analysis.py`
- **Description:** Investigates group decision-making on multiple-choice questions using quantal response models.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run numerical analysis:
   ```bash
   python src/numerical.py
   ```

3. Run Bayesian analysis:
   ```bash
   python src/bayesian_analysis.py
   ```

4. Run response analysis:
   ```bash
   python src/response_analysis.py
   ```

## Requirements
- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## License
MIT License
