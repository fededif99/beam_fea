# Changelog

All notable changes to the `beam_fea` public release are documented here.

> **Note**: This changelog covers public releases only. For the full granular
> development history, see the private `beam_fea_dev` repository.

---

## [v1.0.0] - 2026-02-08

### Initial Public Release

- Sparse CSR matrix assembly with `spsolve` / ARPACK backend.
- Internal force (shear, moment) recovery via statically consistent element shape function superposition.
- Engineering material library (20+ pre-defined entries).
- Graded, curved, and multi-span mesh generation.
- Euler-Bernoulli and Timoshenko element formulations.
- Markdown report generation with saved PNG plots.
