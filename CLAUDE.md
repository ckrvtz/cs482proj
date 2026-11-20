# CLAUDE.md

Standing instructions for LLM-assisted development on this repository. Read this file in full at the start of every session.

## What this project is

A CS482 data mining project: detect and predict **desertification** in dryland areas from free **Sentinel-2 L2A** imagery, accessed through the **Sentinel Hub APIs on the Copernicus Data Space Ecosystem (CDSE)**, using Python and Jupyter notebooks.

- **`docs/ROADMAP.md` is the source of truth** for scope, study design, task IDs, file paths, schemas and acceptance criteria. Read §1–§5 of it before doing any task.
- **`docs/Milestone1.md`, `Milestone2.md`, `Milestone3.md`** are the course requirements. They override the roadmap if the two disagree; fix the roadmap when that happens and log it.

## How to work on a task

1. Find the task ID in `docs/ROADMAP.md` §6 and check its dependencies in §10 (Status tracker). If a dependency isn't done, say so before you start.
2. Do **only** that task. Note out-of-scope problems in your reply or under Open questions (§12). Don't fix them silently.
3. Put reusable logic in `src/desertification/`. Notebooks import from it and stay thin: they orchestrate, display and explain.
4. Add or update `pytest` tests for every new pure function (index formulas, trend statistics, label rules, split logic, gridding).
5. Run the tests and execute the notebooks you changed, top to bottom.
6. Check **every** acceptance criterion in the task block. Report each as met or not met, with evidence (output, file path, number).
7. Update `docs/ROADMAP.md`: tick the task in §10 with the date and a note, add non-obvious choices to the Decision log in `docs/DECISIONS.md` (append-only), and add anything unresolved to §12.
8. Commit with a message that starts with the task ID, e.g. `M1-T05: pilot extraction with caching and provenance`.

If a task is ambiguous, or a decision is really the team's to make (AOI choice, label thresholds, scope changes), ask instead of guessing.

## Commands

(Set up in P0-T01/P0-T02. Python 3.12; pins in `requirements.txt` and `requirements-torch.txt`.)

```bash
conda env create -f environment.yml && conda activate desert   # or a venv: see README "Setup" (install order matters)
pip install -e ".[dev]"                                        # install src/desertification + pytest/ruff
pytest -q                                                      # unit tests
ruff check src tests && ruff format src tests                  # lint + format
jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb   # run a notebook headless
scripts/build_pdf.sh reports/milestone1/milestone1_report.md               # Markdown report → PDF (pandoc)
python scripts/check_env.py [--require-gpu]                               # verify imports, versions, GPU
python scripts/check_secrets.py                                           # fail if a .env value is in any file git would commit
```

## Compute and storage

- Development runs on a **local machine with an NVIDIA GTX 1080 Ti** (Pascal, sm_61, 11 GB VRAM). Colab is only a fallback.
- PyTorch code gets its device from `desertification.hardware.select_device()` (`cuda` if usable, else `cpu`) and must still run on CPU. Size batches for 11 GB.
- torch is pinned to `2.14.1+cu126` because that build still has Pascal kernels (`sm_60`, which runs on sm_61 cards; `get_arch_list()` shows `sm_60`, not `sm_61`). Don't change the torch version or CUDA index without re-running `python scripts/check_env.py --require-gpu` on the GPU machine.
- Imagery stays **local**. `data/raw/` and `data/interim/` are the download cache and the only copy: **never delete or overwrite them without asking**. No commercial cloud storage.

## Repository layout (summary — full version in ROADMAP §5)

- `config/project.yaml`: periods, cell size, resolution, thresholds, seeds, paths. `config/aois.geojson`: AOIs and per-AOI season months, UTM EPSG.
- `src/desertification/`: package (`sh_client`, `ingest`, `aux_data`, `preprocess`, `features`, `trends`, `labels`, `splits`, `models`, `evaluate`, `viz`, `evalscripts/*.js`).
- `notebooks/NN_name.ipynb`: numbered workflow. `notebooks/final/desertification_pipeline.ipynb` is the graded M3 notebook.
- `data/raw`, `data/interim`: gitignored. `data/processed`: parquet tables. `data/sample`, `data/labels`: small and committed.
- `reports/figures`: generated figures. `reports/results`: metric CSVs. `reports/milestoneN/`: report drafts.
- `docs/REFERENCES.md`: every citation (data, papers, libraries, reused code). `docs/DECISIONS.md`: the decision log (append-only).

## Code conventions

- Python ≥ 3.11, type hints on public functions, NumPy-style docstrings that **state units** (reflectance 0–1, metres, mm, degrees) and array shapes.
- `pathlib.Path` everywhere. **No hard-coded absolute paths, years, AOIs or thresholds.** Read them from `config/project.yaml` / `config/aois.geojson` through `desertification.config`.
- Use `logging` in `src/`, not `print`. Notebooks may print.
- Functions should be small and pure where possible. Do I/O at the edges (`ingest`, `aux_data`, notebook cells).
- Tabular data is pandas DataFrames saved as **parquet** with the exact column names and dtypes in ROADMAP §7. If you need a schema change, update §7 in the same commit.
- Rasters: `rasterio`/`rioxarray`, GeoTIFF, **AOI UTM CRS**, CRS and transform always kept with the data.
- Randomness: one global seed from config, passed explicitly (`random_state=`, `torch.manual_seed`, NumPy `Generator`).
- Formatting/linting: `ruff`. Keep dependencies in `environment.yml`/`requirements.txt` pinned, and record new ones in the Decision log (`docs/DECISIONS.md`).

## Notebook conventions

- The first markdown cell holds the title, task ID(s), purpose, inputs, outputs, and how long it takes to run.
- The first code cell holds imports, `%load_ext autoreload` / `%autoreload 2`, config loading, seed setting, and printed library versions.
- Notebooks have to run cleanly with **Restart & Run All**. No hidden state, no cells that depend on execution order beyond top to bottom.
- Expensive steps (API downloads, long training) check for cached outputs first and are behind a clearly named flag (e.g. `REDOWNLOAD = False`).
- Keep committed notebooks small: no giant embedded rasters or dataframes (display `.head()`, summaries, downsampled plots). Aim for under ~5 MB each.
- Every figure: title, axis labels with units, legend, colorbar label, and source attribution *"Contains modified Copernicus Sentinel data [years]"* where relevant. Save report figures to `reports/figures/<milestone>_<name>.png` (dpi ≥ 150) from code.
- Markdown cells explain *why* (the decision and its justification), not just *what*. The report and video are built from them.

## Sentinel Hub / data rules

- **Credentials:** only through environment variables `SH_CLIENT_ID` / `SH_CLIENT_SECRET`, loaded from a gitignored `.env`. Never write a secret into code, config, notebook output, logs, commit messages or chat. If you find a committed secret, stop and tell the user.
- **Endpoint:** CDSE (`https://sh.dataspace.copernicus.eu`) with the CDSE token URL, via `desertification.sh_client`. Don't use the commercial `services.sentinel-hub.com` endpoints.
- **Quota: 30,000 PU and 30,000 requests for the whole project.** CDSE resets quotas monthly, but the project is done within one month, so never plan around a reset (ROADMAP §3.4, Decision log #28). Caps are in ROADMAP §3.4; scope shrinks if usage requires. Multi-date composites and FLOAT32 output multiply PU cost (Decision log #26). Before making **more than ~10 API requests in one go**, or any bulk/scaled download, **stop and ask the user**, giving an estimate of request count and processing units and the remaining budget.
- Log every request batch in `data/api_usage_log.csv` with `desertification.usage.make_entry` / `append_usage`; check a planned batch with `usage.check_cap` first.
- API-calling code takes an explicit opt-in (`allow_api=True`; notebooks use an `ALLOW_API_CALLS` flag that defaults to `False`) and reads the cache first (`desertification.cache`).
- **Cache everything.** Every download writes to `data/raw/...` with a JSON **provenance sidecar** (request params, evalscript SHA-256, time interval, bbox, CRS, resolution, timestamp, sentinelhub version). Downloads skip if a matching file + sidecar exist.
- Use **Catalog API** to check availability, **Process API** for rasters, **Statistical API** for cheap single-geometry time series.
- Sentinel-2 L2A gotchas you must respect (details in ROADMAP §3.1):
  - Sentinel Hub harmonises the 2022-01-25 processing-baseline offset by default. **Do not subtract 1000 / apply BOA_ADD_OFFSET yourself** unless harmonisation is explicitly disabled.
  - Mask clouds with SCL (invalid set in config). Always carry the **valid-observation count**. Bright desert sand can be misflagged as cloud, so check for holes.
  - Max 2500 × 2500 px per Process API request. Split larger AOIs.
  - Seasons are **AOI-specific** (from `aois.geojson`), never calendar seasons.
- Never commit raw rasters or large files. `data/processed/` is committed only if its total is ≲ 50 MB; if it grows past that, raise it with the user (a GitHub Release asset is the likely option).
- Each external dataset (CHIRPS, DEM, WorldCover, aridity, Landsat, …) gets its licence and citation added to `docs/REFERENCES.md` when first used.

## Machine learning integrity rules (non-negotiable)

- **No label leakage.** For the predictive task, features come only from the early period (2019–2021, from config). Assert this in code. Features computed from the same series and window as a label must not be model inputs.
- **Spatial cross-validation only** (`StratifiedGroupKFold` on `spatial_block_id`, plus leave-one-AOI-out). Never report random K-fold on cells as a real result.
- **Locked test set.** Created once in M3-T06. Evaluate on it only in M3-T12, once per final model. Never tune on it. Disclose any post-test change.
- Fit scalers, imputers, encoders, feature selection and resamplers (SMOTE etc.) **inside** pipelines on training folds only (`sklearn`/`imblearn` `Pipeline`).
- Label thresholds are tuned once on training data, frozen, versioned (`label_version`) and logged.
- Report mean ± std over folds, include the baselines, and report failures and negative results honestly.
- **Every number in a report or slide comes from a file in `reports/results/` produced by code.** Never type in or round up metrics by hand. Never fabricate or "illustrate" results.

## Academic integrity and citations

- **Never fabricate citations, DOIs, URLs, quotes, API parameters or library functions.** If you aren't sure something exists, check the official docs or the installed package source (`python -c "import x; help(x.y)"`), and say what you checked. Mark unverified references `[VERIFY]` and unsupported claims `[TODO: …]`.
- Cite all third-party code, data, libraries and methods in `docs/REFERENCES.md`. If code is adapted from a tutorial or repo, put a comment with the source URL above it.
- Data attribution: *"Contains modified Copernicus Sentinel data [years], processed by Sentinel Hub."*
- People do the instructor discussion (M1-T04) and the manual validation labelling (M3-T05). Don't invent their content.
- Report drafts are drafts: team members must review, understand and be able to explain everything in them (the video requires each member to present their own contributions).

## Report writing

- Write reports in `reports/milestoneN/` as Markdown and build the PDF with `scripts/build_pdf.sh` (pandoc). Use relative image paths to `reports/figures/` so the build finds them. Follow the required sections in the ROADMAP task block and the milestone file. Keep a requirements checklist against the milestone file.
- Refer to figures by filename and tables by results CSV. Keep claims specific and quantified.
- Milestone 2 has a **2–3 page** limit; Milestone 3's video has a **30 minute** limit.

## Git

- Small, focused commits whose messages start with the task ID.
- Never commit `.env`, credentials, `data/raw/`, `data/interim/`, large binaries or notebook outputs containing secrets. Run `python scripts/check_secrets.py` before committing anything produced with credentials loaded.
- Never create, overwrite or delete `.env`: it holds the user's real credentials.
- Don't rewrite shared history. Don't push to `main` unless the team has agreed to; use feature branches.

## Don'ts (quick list)

- Don't hit the API in bulk without asking.
- Don't use random (non-spatial) CV, tune on the test set, or leak late-period information into features.
- Don't hard-code paths, years, AOIs or thresholds.
- Don't apply the L2A offset manually when using harmonised Sentinel Hub output.
- Don't invent data, metrics, citations or API behaviour.
- Don't expand scope beyond the assigned task without saying so.
