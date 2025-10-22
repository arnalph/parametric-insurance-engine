# Parametric Rainfall Insurance Analysis

This project models and visualizes rainfall-triggered insurance payouts across agricultural regions using IMD gridded data. It supports both binary and graduated contracts, enabling spatial risk assessment and contract design for crops like rice and cotton.

## 🔧 Features

- Pointwise loss analysis for multiple locations
- Binary and graduated payout functions
- Expected Annual Loss (EAL) and Maximum Annual Loss (MAL) metrics
- Scatter and bar chart visualizations (matplotlib + Plotly)
- Modular code for easy integration with new datasets or contract types
- Two Jupyter notebooks for interactive analysis:
  - `pointwise_analysis.ipynb`: performs point-level calculations and contract modeling for individual locations
  - `region_comparison.ipynb`: aggregates and compares EAL/MAL metrics across multiple regions for spatial benchmarking
