# dataset_pothole440_seg_grouped

Leakage-controlled YOLO instance/region segmentation benchmark reconstructed from `./Pothole440-main/`.

- **Split Policy:** Deterministic `seed=42` split at the `98`-group base capture ID level (~70/15/15: 68 train / 15 val / 15 test).
- **Leakage Control:** Zero base-group overlap across splits; validation and test contain strictly unaugmented original captures (`15` + `15`); all `val`/`test` augmentations excluded.
- **Mask Caveat:** Source Pothole440 labels are binary semantic masks (without instance IDs or interior hole encoding in YOLO-Seg).
- **Usage & Citation:** Strictly for non-commercial academic research. Cite Ali et al. (2026), *IEEE Transactions on Industrial Informatics*, DOI `10.1109/TII.2026.3706348`.
