# Experiment 2: HN-E (Hardware-Noise Extrapolation)

## Overview
Experiment 2 validates the HN-E (Hardware-Noise Extrapolation) head of the N2LN model.

## Dataset
- **Noise scales**: 1.0, 1.5, 2.0, 2.5, 3.0
- **Qubits**: 4
- **Circuits**: 5000 (mix of random and Clifford)
- **Shots per measurement**: 1000

## Training
- Load Phase 4 checkpoint (SN-D)
- Freeze SN-D head
- Train HN-E head only
- Phase 2 curriculum

## Evaluation
- Compare against ZNE baseline
- Metrics: TVD, Fidelity, KL

## Status
🟡 Data generation in progress
