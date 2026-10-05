# Fire-Continuity Regions

Hierarchical delineation of **fire-continuity regions** by spatially constrained
clustering of species-level fuel flammability and terrain, applied to the
**2025 Gyeongsangbuk-do (Korea) megafire** region.

This repository contains the analysis code and the main regional outputs for the
manuscript:

> Kwon, Y. *Hierarchical Delineation of Fire-Continuity Regions by Spatially
> Constrained Clustering of Species-Level Fuel Flammability and Terrain:
> Application to the 2025 Gyeongsangbuk-do Megafire Region.* (under review).

## What this does

30 m species flammability (from the **pre-fire, 2019** national forest-type map)
and terrain (elevation, slope, northness) are aggregated to 600 m blocks and
grouped by **rook-adjacency-constrained Ward clustering with a compactness weight**
(W_SPACE), yielding mutually exclusive, spatially contiguous, hierarchically
nested regions **L1 (5) ⊃ L2 (18) ⊃ L3 (55)**. The regions are validated by
descriptive alignment with the 2025 fire, by independent transfer to the 2022
Uljin fire (AUC ≈ 0.85), and by fuel-connectivity / percolation analysis.

## Repository structure

```
code/        analysis and figure scripts (Python)
outputs/     main regional outputs and validation summaries
figures/     figures used in the manuscript (English labels)
```

## Input data (not redistributed here)

All inputs are **publicly available** and are not re-hosted in this repository:

- Forest-type map (1:5,000, species / KOFTR_GROU) — **Korea Forest Service**
- Detailed land cover — **Ministry of Environment**
- Digital elevation model (UTM-K) — **National Geographic Information Institute**
- Administrative (census) boundaries — **Statistics Korea**
- Validation fire perimeters — 2025 perimeters; 2022 Uljin via Sentinel-2 dNBR (ESA Copernicus)

Grid: EPSG:5186, 30 m; blocks of 20 × 20 pixels (600 m).

## Pipeline (run order)

1. `code/spec_exp2019.py` — build 2019 pre-fire species-flammability and terrain rasters
2. `code/build_blocks2019.py` — aggregate 30 m pixels to 600 m blocks (≥25 forest pixels)
3. `code/cluster4_2019.py` — spatially constrained Ward clustering (+ W_SPACE) → `outputs/hier4_2019.npz`
4. `code/av5_2019.py` — region profiles and metrics
5. `code/contrib.py` — variable-contribution screening (logistic AUC, ΔAUC)
6. `code/av_sweep4.py`, `code/sweepplot4_en.py` — W_SPACE sensitivity
7. `code/conn_block.py`, `code/conn_part2.py`, `code/backbone.py`, `code/conn_fig_en.py` — fuel connectivity & percolation
8. `code/skater_confirm.py` — SKATER / coordinate-free cross-check
9. `code/bootstrap_auc.py` — bootstrap confidence intervals
10. `code/uljin_may_main.py`, `code/uljin_may_finalize_en.py` — 2022 Uljin independent-transfer validation
11. `code/zone3_decision.py` — unburned high-risk region and preventive-priority reallocation
12. `code/fig_base4_19_en.py`, `fig_burn4_19_en.py`, `fig_av4_19_en.py`, `fig_compare4_19_en.py`, `figfinal4_19_en.py`, `fig_flow_en.py` — figures

Paths at the top of each script point to the author's working directory and should
be adjusted to local input locations.

## Outputs

- `outputs/hier4_2019.npz` — per-block region assignments: `bid` (block id), `l1`,`l2`,`l3`, grid metadata (`wb`,`hb`,`B`,`H`,`W`)
- `outputs/blocks_av4_2019.npz` — per-block standardized features (species flammability, elevation, slope, northness) and burn fraction
- `outputs/connectivity.json` — threshold sweep, percolation (p_c), correlation length, backbone statistics
- `outputs/confirm_skater.json` — SKATER / coordinate-free reproduction of the "fuel-twin" structure
- `outputs/uljin_may.json` — 2022 Uljin transfer: AUC, offset, evergreen fractions, phenology controls

## Requirements

Python 3.9+ with `numpy`, `scipy`, `scikit-learn`, `pandas`, `rasterio`,
`geopandas`, `shapely`, `affine`, `matplotlib`.

## License

- **Code**: MIT License (see `LICENSE`).
- **Figures and derived output files**: Creative Commons Attribution 4.0 (CC BY 4.0).
- Input datasets remain under the terms of their respective public providers.

## Contact

Yongseok Kwon — Gyeongbuk Development Institute (GDI) — kwony@gdi.re.kr
ORCID: 0000-0002-9783-8808
