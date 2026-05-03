# Three-Body Simulation

Numerical three-body simulation with Python, SciPy and Matplotlib.

PT-BR: simulacao numerica do problema dos tres corpos para estudo de modelagem fisica e visualizacao.

## Overview

This project solves a simplified gravitational three-body system and visualizes trajectories with Python scientific tooling.

## Stack

- Python
- NumPy
- SciPy
- Matplotlib

## Architecture

- `teoria_v3.py` is the latest simulation entry point.
- `teoria_3_corpos.py` and `teoria_3_corpos_v2.py` preserve earlier iterations.
- Matplotlib renders trajectory and animation outputs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib
```

## Usage

```bash
python teoria_v3.py
```

## Project Status

`study`

This is a supporting project for numerical methods, simulation and visualization.

## Roadmap

- Add parameter documentation.
- Save generated plots to an output folder.
- Add a short explanation of the physical assumptions.
