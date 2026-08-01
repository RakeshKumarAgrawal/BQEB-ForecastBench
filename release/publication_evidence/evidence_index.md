# Publication Evidence Package

## Purpose

This package records the repository, environment, artifacts, and provenance used to assess BQEB-ForecastBench for publication readiness after Commit 5. It is evidence only: no benchmark, model, dataset, table, figure, or report was regenerated.

## Contents

- `repository_inventory.md`: repository identity, branches, source commit, structure, and file counts.
- `artifact_inventory.md`: generated artifact paths, sizes, UTC timestamps, categories, and originating components.
- `repository_tree.txt`: recursive source and evidence-package tree with development caches excluded.
- `environment_summary.md`: Python, operating system, Git, package, execution-environment, and random-seed details.
- `publication_manifest.json`: machine-readable package identity, artifact references, and experiment traceability.
- `evidence_index.md`: navigation and verification summary for this package.

## Repository Status

- Repository: `BQEB-ForecastBench`
- Current branch: `develop`
- Default branch: `main`
- Evidence source HEAD: `eb238c9506b00bf9d9c30a13020ca485793d1e3d`
- Repository version: `0.4.0`
- Installed package version: `0.4.0`
- Release candidate: yes

## Generated Artifacts

The inventory covers 30 artifacts: 5 dataset artifacts, 1 model artifact, 1 checkpoint, 6 evaluation artifacts, 2 tables, 6 figure/caption artifacts, 5 reports, 1 experiment manifest, 2 training artifacts, and 1 publication manifest.

Publication outputs comprise Table 5 in CSV and Markdown, Figures 4 and 5 in PNG and SVG with captions, and three benchmark reports. Their paths are enumerated in `publication_manifest.json` and their filesystem evidence is recorded in `artifact_inventory.md`.

## Verification Status

- Required evidence files: PASS
- Repository and package version consistency: PASS
- Current branch and HEAD consistency: PASS
- Configuration hash consistency: PASS
- Dataset fingerprint consistency: PASS
- Experiment manifest reference and SHA-256: PASS
- Evaluation and publication artifact existence: PASS
- Environment package inventory: PASS
- Repository quality gates: PASS

## Traceability Information

The evidence package describes repository snapshot `eb238c9506b00bf9d9c30a13020ca485793d1e3d`. The frozen experiment manifest is `artifacts/experiments/experiment_manifest.json`, with experiment ID `forecastbench-baselines-20260801T061253803994Z-d6289b2c451f` and source commit `5bde0fe9959fed7307a3aae8f1927acd59fdfb8b`.

- Configuration hash: `d6289b2c451f33a39392fc26b556bbfecc84ff9f52d9953bb94b9f2259902b8e`
- Dataset fingerprint: `be6b8dcca5d1bdd5472c64c5380f3ca548c91db5f3a03fdd87dbb6b50764d1ea`
- Experiment manifest SHA-256: `c8f5807bd11bec78871156f24fefe4b56a62e1908c3252383b23f50d1cd95a6b`

The distinct commits are intentional: the experiment commit identifies the code snapshot represented by the frozen experiment manifest, while the evidence source HEAD includes the subsequent release-metadata finalization commit.
