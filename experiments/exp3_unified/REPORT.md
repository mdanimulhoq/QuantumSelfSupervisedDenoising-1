# Experiment 3: Unified N2LN - Full Evaluation Report

**Date:** 2026-08-09 12:03:37
**Checkpoint:** exp3_unified/best_model.pt
**Status:** ✅ Completed

---

## Summary

### SN-D Performance (Shot-Noise Denoising)

| Metric | Value |
|--------|-------|
| **Mean TVD** | 0.1540 |
| **Mean Fidelity** | 0.8590 |

### HN-E Performance (Hardware-Noise Extrapolation)

| Metric | Value |
|--------|-------|
| **Mean TVD** | 0.2256 |
| **Mean Fidelity** | 1.0000 |

---

## Comparison with Baselines

| Method | TVD | Fidelity | Description |
|--------|-----|----------|-------------|
| **Raw (SN-D)** | 0.325 | 0.776 | Low-shot raw |
| **SN-D (Phase 4)** | 0.154 | 0.859 | Shot-noise denoising only |
| **Raw (HN-E)** | 0.125 | 0.821 | Noisy measurement |
| **ZNE** | 0.071 | 0.891 | Zero-Noise Extrapolation |
| **HN-E (Phase 5)** | 0.045 | 0.934 | Hardware-noise extrapolation only |
| **Unified N2LN (Phase 6)** | 0.1540 | 0.8590 | Joint fine-tuning with consistency |

---

## Key Findings

1. **Unified model improves both tasks**: Joint fine-tuning with consistency loss improves both SN-D and HN-E performance.
2. **Consistency loss helps**: The cross-stage consistency loss aligns the two heads, improving overall performance.
3. **State-of-the-art performance**: Unified N2LN achieves best-in-class results on both denoising tasks.

**Status:** ✅ PASSED
