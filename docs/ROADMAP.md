# ROADMAP — Detecting Desertification with Sentinel-2 and Machine Learning

> **Status:** living document. It is the single source of truth for scope, tasks and acceptance criteria.
> **Companion file:** [`/CLAUDE.md`](../CLAUDE.md), which holds the standing rules for LLM-assisted development (coding conventions, data/API rules, integrity rules).
> **Course requirements:** [`Milestone1.md`](Milestone1.md), [`Milestone2.md`](Milestone2.md), [`Milestone3.md`](Milestone3.md). If this roadmap and a milestone file disagree, **the milestone file wins**: fix the roadmap and log the change in the [Decision log](#11-decision-log).

---

## Table of contents

0. [How to use this document](#0-how-to-use-this-document)
1. [Project summary and research questions](#1-project-summary-and-research-questions)
2. [Deliverables map](#2-deliverables-map)
3. [Data sources and technical foundation](#3-data-sources-and-technical-foundation)
4. [Study design](#4-study-design)
5. [Repository layout](#5-repository-layout)
6. [Phased task plan](#6-phased-task-plan)
7. [Data dictionary](#7-data-dictionary)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Risks and mitigations](#9-risks-and-mitigations)
10. [Status tracker](#10-status-tracker)
11. [Decision log](#11-decision-log) → [`DECISIONS.md`](DECISIONS.md)
12. [Open questions](#12-open-questions)
13. [Prompt templates](#13-prompt-templates)
14. [Starting reference list](#14-starting-reference-list)

---

## 0. How to use this document

### For people
- Pick the next unchecked task in the [Status tracker](#10-status-tracker) whose dependencies are done.
- Give an LLM the task ID and one of the [prompt templates](#13-prompt-templates). The task block below has everything it needs: goal, inputs, outputs, steps, and **acceptance criteria**.
- Review the output against the acceptance criteria before you tick the box. "The notebook runs" is not enough; check the numbers make physical sense.

### For LLMs
1. Read `CLAUDE.md` first, then this file's §1–§5, then the specific task block.
2. Do **only** the task you were given. If you find work belonging to another task, note it in your reply and leave it.
3. A task is done only when **every** acceptance criterion is met. If one can't be met, say which one and why. Don't quietly lower the bar.
4. When you finish a task:
   - tick it in the [Status tracker](#10-status-tracker) (`[x]`), with the date and a one-line note;
   - add any non-obvious choice you made (thresholds, AOI bounds, library choices, etc.) to the [Decision log](DECISIONS.md) (`docs/DECISIONS.md`);
   - add anything you couldn't resolve to [Open questions](#12-open-questions).
5. Never invent data, results, metrics, API behaviour or citations. If you are unsure whether an API parameter or library function exists, check the official docs or the installed package source before using it, and say what you checked.

### Task block format
Every task uses this format so it can be pasted straight into a prompt:

```
### <ID> — <title>
Milestone / week | Depends on | Owner
Goal:        one sentence
Inputs:      files/configs/data the task reads
Outputs:     exact file paths the task creates or changes
Steps:       ordered, concrete
Acceptance:  checkable criteria (all must hold)
Notes:       gotchas, hints, links
```

IDs: `P0-*` setup, `M1-*` Milestone 1, `M2-*` Milestone 2, `M3-*` Milestone 3.

---

## 1. Project summary and research questions

### 1.1 One-paragraph summary
We build a reproducible pipeline that pulls free Sentinel-2 Level-2A surface-reflectance imagery through the **Sentinel Hub APIs hosted on the Copernicus Data Space Ecosystem (CDSE)**. The pipeline turns that imagery into a tabular, grid-cell × time dataset of spectral bands, vegetation/soil indices and auxiliary variables (rainfall, elevation, land cover), and uses data mining and machine learning to **identify and characterise dryland areas undergoing desertification**: a sustained loss of vegetation cover and productivity, and a rise in bare-soil and surface-brightness signals, that rainfall variability alone does not explain.

### 1.2 Working definition of desertification
We follow the UNCCD definition: *land degradation in arid, semi-arid and dry sub-humid areas resulting from various factors, including climatic variations and human activities.* In practice this means:
- **Scope:** only drylands (Aridity Index between about 0.05 and 0.65). Every AOI must meet this, or be explicitly labelled as a control.
- **Observable proxy:** a statistically significant, **rainfall-adjusted** negative trend in vegetation (NDVI/MSAVI2) together with a positive trend in a bare-soil or brightness signal (BSI/albedo) over the study period.
- **Not desertification by itself:** one drought year, crop rotation or fallow, burn scars, seasonal phenology, and water-level changes that are not paired with land-surface degradation. The labelling logic (§4.5) has to handle these explicitly.

### 1.3 Research questions
- **RQ1 (detection):** Using a rainfall-adjusted trend analysis (RESTREND-style) on 10–20 m Sentinel-2 data, which grid cells in the study AOIs show significant degradation, stability or improvement from 2019 to 2025?
- **RQ2 (prediction, main ML task):** Can spectral, temporal and auxiliary features from an **early period** (2019–2021) predict which cells will be degrading in the **later period** (2023–2025)? Which model family does best under **spatial** cross-validation?
- **RQ3 (unsupervised structure):** Do unsupervised methods (clustering of multi-year trajectories, anomaly detection in Albedo–MSAVI feature space) find degradation patterns that agree with the trend-derived labels and with a hand-checked validation sample?
- **RQ4 (interpretation):** Which features (e.g. dry-season SWIR reflectance, NDVI seasonal amplitude, rainfall-use efficiency, slope) drive predictions, and do they agree with the physical understanding of dryland degradation?

### 1.4 Intended novel contribution (required by Milestone 2/3)
Most published desertification-trend work uses coarse sensors (AVHRR GIMMS at about 8 km, MODIS at 250 m–1 km). Our planned contributions:
1. **Field-scale RESTREND:** rainfall-adjusted trend labels at Sentinel-2 resolution (aggregated to about 100 m cells), across several contrasting dryland AOIs.
2. **Early-warning framing:** predicting *future* degradation from *earlier* observations, with strict temporal and spatial separation (no label leakage). This is a harder and more useful task than re-classifying the period the labels came from.
3. **Honest generalisation tests:** leave-one-AOI-out evaluation, to show whether a model trained in one dryland transfers to another.

Milestone 2 (§6, `M2-*`) must confirm these are actually new against the literature, and narrow or adjust them if they are not. Log any change in the Decision log.

---

## 2. Deliverables map

| Milestone | Duration | Graded deliverable | Produced by tasks | Format |
|---|---|---|---|---|
| M1 | 3 weeks | Report: ≥2 candidate sources with endpoints/docs, feasibility (≥10,000 records, ≥15 attributes), instructor-discussion summary, sample of extracted data, descriptive stats, cleaning, preliminary analysis, project ideas | `M1-T01` … `M1-T10` | PDF |
| M2 | ~1–2 weeks | 2–3 page review of existing data mining / ML work on this kind of data: impressive tasks, how they were done, code links, techniques not covered in class, our proposed extension/new contribution | `M2-T01` … `M2-T04` | PDF |
| M3 | 4 weeks | Report (team, work split, data knowledge, preprocessing, model experiments + metrics, justification, interpretation, comparison to prior work, link to executable `.ipynb`, link to video) **and** ≤30 min group video | `M3-T01` … `M3-T14` | PDF + video link |

**Milestone 3 grading weights** (plan effort to match): Data prep & feature engineering 20% · Model development & experimentation **40%** · Interpretation & report 20% · Video 20% (content 16%, A/V quality 2%, time limit 2%).

Course hard requirements to keep in view all the time:
- ≥ **10,000 records** with ≥ **15 attributes** (M1). Our cell-year table (§4.3) is sized to pass this comfortably.
- **Multiple models**, evaluated with appropriate metrics, **with hyperparameter tuning** (M3).
- Handle **missing values, encoding, scaling, feature extraction, class imbalance** (M3).
- **Executable, reproducible `.ipynb`** linked from the report (M3).
- **Video ≤ 30 min**, all members speak and each states their own contributions; link set to unlisted / anyone-with-link and tested in an incognito window (M3).
- Every group member submits the M3 report.

---

## 3. Data sources and technical foundation

### 3.1 Primary source — Sentinel-2 L2A via Sentinel Hub on CDSE

| Item | Value |
|---|---|
| Provider | Copernicus Data Space Ecosystem (ESA / EU), Sentinel Hub services |
| Account | Free registration at `https://dataspace.copernicus.eu`; create an **OAuth client** in the Sentinel Hub dashboard (`https://shapps.dataspace.copernicus.eu/dashboard/`) |
| Base URL | `https://sh.dataspace.copernicus.eu` |
| Token URL | `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token` |
| Docs | `https://documentation.dataspace.copernicus.eu/APIs/SentinelHub.html` |
| Python client | `sentinelhub` (sentinelhub-py), docs: `https://sentinelhub-py.readthedocs.io` |
| Collection | Sentinel-2 L2A (bottom-of-atmosphere reflectance, includes the Scene Classification Layer `SCL`) |
| APIs we use | **Catalog API** (scene search / availability / cloud cover), **Process API** (raster composites via evalscripts), **Statistical API** (cheap aggregated time series over one geometry) |
| Quota | **30,000 processing units (PU) and 30,000 requests available** on the project account (dashboard, 2026-11-20). CDSE documents the free "Copernicus General Users" tier as 10,000 PU and 10,000 requests per month, 300 of each per minute, and says quotas "are monthly and reset on the first day of every month" ([Quotas](https://documentation.dataspace.copernicus.eu/Quotas.html)). **Plan: the whole project is completed within one quota month, so 30,000 PU / 30,000 requests is the total budget for everything** (Decision log #28). Budget in §3.4. |
| Attribution | Every figure/report using the data must say: *"Contains modified Copernicus Sentinel data [years]."* |

**Sentinel-2 L2A bands we use**

| Band | Name | Centre (nm, approx.) | Native res. |
|---|---|---|---|
| B02 | Blue | 490 | 10 m |
| B03 | Green | 560 | 10 m |
| B04 | Red | 665 | 10 m |
| B05 | Red-edge 1 | 705 | 20 m |
| B06 | Red-edge 2 | 740 | 20 m |
| B07 | Red-edge 3 | 783 | 20 m |
| B08 | NIR | 842 | 10 m |
| B8A | Narrow NIR | 865 | 20 m |
| B11 | SWIR 1 | 1610 | 20 m |
| B12 | SWIR 2 | 2190 | 20 m |
| SCL | Scene classification | — | 20 m |

B01 and B09 (60 m, aerosol/water vapour) are left out by default. B10 does not exist in L2A.

**SCL classes**: 0 no data · 1 saturated/defective · 2 dark area / cast shadow · 3 cloud shadow · 4 vegetation · 5 not vegetated · 6 water · 7 unclassified · 8 cloud medium prob. · 9 cloud high prob. · 10 thin cirrus · 11 snow/ice.
Default **invalid** set for masking: `{0, 1, 2, 3, 8, 9, 10, 11}`. Water (6) is kept in composites but masked out of the vegetation analysis later.

**Known gotchas (read before writing any ingestion code)**
1. **Processing baseline 04.00 offset (from 2022-01-25):** L2A digital numbers gained a `BOA_ADD_OFFSET` of −1000. Sentinel Hub *harmonises* values by default (the `harmonizeValues` processing option), so reflectances are comparable across the change. **Do not apply the offset yourself** unless harmonisation was turned off. Verify it once in `M1-T05`: a stable desert pixel should show no step change at 2022-01-25.
2. **Bright desert surfaces can be misclassified as cloud** in SCL. Always store the **valid-observation count** per pixel/cell, and check composites over bright sand for holes. If a whole AOI loses too much data, log it and consider a looser mask (e.g. keep class 8) **for that AOI only**, documented in the Decision log.
3. **Process API output size limit** is 2500 × 2500 px per request. Split larger AOIs into tiles (`sentinelhub.BBoxSplitter` / `UtmZoneSplitter`).
4. **CRS:** request data in the AOI's **UTM zone** so pixels are square metres and cell areas are equal. Store the EPSG code with every file.
5. **Multi-temporal requests cost more PUs** (cost grows with the number of acquisitions). Run a pilot (`M1-T05`) to measure PU cost per AOI-season before scaling up.
6. **Revisit and coverage:** S2A (2015), S2B (2017) and S2C (launched 2024) give ~5-day revisit at the equator, but early years and some regions are sparser. Use the Catalog API to check per-AOI availability; the default study period is **2019-01-01 to 2025-12-31**.

**Reference evalscript — cloud-masked per-band median composite** (starting point for `src/desertification/evalscripts/median_composite.js`; test it before relying on it):

```js
//VERSION=3
// Cloud-masked per-band median over all acquisitions in the request's time range.
// Output: one band per entry in BANDS, plus the count of valid observations (last band).
const BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"];
const INVALID_SCL = [0, 1, 2, 3, 8, 9, 10, 11];

function setup() {
  return {
    input: [{ bands: BANDS.concat(["SCL", "dataMask"]) }],
    output: { bands: BANDS.length + 1, sampleType: "FLOAT32" },
    mosaicking: "ORBIT"
  };
}

function median(values) {
  if (values.length === 0) return NaN;
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

function evaluatePixel(samples) {
  const valid = samples.filter(s => s.dataMask === 1 && !INVALID_SCL.includes(s.SCL));
  const out = BANDS.map(b => median(valid.map(s => s[b])));
  out.push(valid.length);
  return out;
}
```

**Reference Python setup** (goes in `src/desertification/sh_client.py`):

```python
import os
from sentinelhub import SHConfig, DataCollection

def get_sh_config() -> SHConfig:
    """Return a Sentinel Hub config pointed at CDSE. Credentials come from env vars only."""
    config = SHConfig()
    config.sh_client_id = os.environ["SH_CLIENT_ID"]
    config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    return config

S2L2A_CDSE = DataCollection.SENTINEL2_L2A.define_from(
    "s2l2a_cdse", service_url="https://sh.dataspace.copernicus.eu"
)
```

### 3.2 Second candidate source (needed for Milestone 1's "two sources")
**Microsoft Planetary Computer STAC API** — `https://planetarycomputer.microsoft.com/api/stac/v1` (Python: `pystac-client` + `planetary-computer`). Free, read-only, no key needed for public collections. Relevant collections: Landsat Collection 2 Level-2 (record back to the 1980s, which gives historical context that Sentinel-2 can't), ESA WorldCover, Copernicus DEM. M1 has to document it as a real alternative with a sample pull, even if Sentinel Hub stays primary.
*Fallback role:* if CDSE quota or outages block progress, Sentinel-2 L2A is also on Planetary Computer. Switching is a Decision log entry, not a silent change.

### 3.3 Auxiliary data (features, labelling, stratification)

| Dataset | Use | Access (verify in `M1-T02`) |
|---|---|---|
| **CHIRPS v3** monthly precipitation (0.05°, 60°N–60°S, 1981–present) | Rainfall adjustment for RESTREND; rainfall features | UCSB Climate Hazards Center (`https://data.chc.ucsb.edu/products/CHIRPS/v3.0/`). v3 replaces v2: "Production of CHIRPS v2 will end after December 2026" (Decision log #24) |
| **Copernicus DEM GLO-30** | Elevation, slope, aspect | **Planetary Computer** `cop-dem-glo-30` (open). On CDSE, COPERNICUS_30 is restricted to authorized user categories and requests default to the 90 m DEM (Decision log #25) |
| **ESA WorldCover** 10 m (2020, 2021) | Land-cover fractions per cell; masks (water, built-up, cropland); stratification | AWS Open Data (`esa-worldcover` bucket) or Planetary Computer |
| **Aridity index** (P/ETo) | Confirm AOIs are drylands; feature | Computed from CHIRPS v3 rainfall ÷ **Hobbins reference ET** (CHC, monthly 0.05°, `https://data.chc.ucsb.edu/products/Hobbins_RefET/`) over 1991–2020, per the UNCCD Art. 1 definition. The Trabucco & Zomer Global Aridity Index on figshare blocks automated downloads (Decision log #31) |
| **Trends.Earth / SDG 15.3.1** layers (optional) | External comparison for our degradation labels | Trends.Earth docs / data |

Document licence and citation for each one in `docs/REFERENCES.md` when you first use it.

### 3.4 Quota budget and local data policy

**Budget: 30,000 PU / 30,000 requests for the entire project.** All CDSE work is planned within a single quota month, so don't count on the monthly reset (Decision log #28). PU is the binding limit; requests are unlikely to be. Scope (number of AOIs, years, resolution, bands) is adjusted as the ledger shows real usage: first after the M1-T05 pilot, then before each large batch.

| Use | PU cap | Notes |
|---|---|---|
| Connectivity, catalog checks, pilot (P0-T03, M1-T05) | 1,000 | Pilot measures the real PU cost per AOI-season |
| Scaled extraction (M1-T06) | 20,000 | Plan from pilot numbers; cut AOIs/resolution if the estimate goes over |
| Auxiliary layers from Sentinel Hub (DEM), EDA via Statistical API | 2,000 | |
| Reserve (re-runs, fixes, M3 extras) | 7,000 | Don't touch without asking the user |

- Every request batch is logged in `data/api_usage_log.csv` (committed, small): date, task ID, API, request count, PU before/after from the dashboard, purpose.
- Before a batch, compare the estimate against the remaining budget in that log. Stop and ask if a batch would push a row above its cap.

**Data stays local.** The project runs on a local machine. Raw and interim data (`data/raw/`, `data/interim/`) live only on that machine, are gitignored, and are the cache that prevents duplicate API requests, so **never delete them without asking**. Keep a backup copy outside the repo (e.g. an external drive) once M1-T06 finishes. `data/processed/` is committed if it stays ≲ 50 MB; if it grows past that, a GitHub Release asset will be considered then (Decision log entry). No commercial cloud hosting is planned.

---

## 4. Study design

### 4.1 Areas of interest (AOIs)
Choose **4–6 AOIs**, each about **20 km × 20 km** (so one Process API request at 20 m is ≤ 1000 × 1000 px), covering different dynamics. These are **candidate starting points with approximate centres**; `M1-T02` must confirm each with imagery, the aridity index and data availability, then fix the exact bounding boxes in `config/aois.geojson`.

| Candidate AOI | Approx. centre (lat, lon) | Expected dynamic | Role |
|---|---|---|---|
| Aral Sea / Aralkum (UZ/KZ) | ~45.0, 60.0 | Strong degradation, exposed seabed, salinisation | Degrading |
| Ferlo, Senegal (Sahel) | ~15.5, −14.5 | Mixed: grazing pressure, rainfall-driven variability | Mixed |
| Fakara / Tillabéri, Niger (Sahel) | ~13.5, 2.6 | Cropland expansion, fallow loss | Mixed |
| Kubuqi Desert, Inner Mongolia (CN) | ~40.2, 108.5 | Afforestation / re-greening | Improving (control) |
| Lake Urmia basin, Iran | ~37.7, 45.3 | Lake shrinkage, salt flats, agricultural pressure | Degrading |
| Stable hyper-arid erg: Grand Erg Occidental, Algeria | ~30.6, 0.6 (chosen in M1-T02; `[VERIFY]` on imagery) | Stable bare sand | Stable (control) |

Per-AOI config (in `config/aois.geojson` properties or `config/project.yaml`): `aoi_id`, `name`, `bbox_wgs84`, `utm_epsg`, `wet_season_months`, `dry_season_months`, `role`, `aridity_index`, `notes`.
**Seasons are AOI-specific** (e.g. the Sahel wet season is roughly Jun–Oct). Don't use calendar seasons across AOIs.

### 4.2 Time design

| Period | Years | Purpose |
|---|---|---|
| Full study period | 2019–2025 | Trend analysis / labels (RQ1) |
| Feature (early) period | 2019–2021 | Model inputs for the predictive task (RQ2) |
| Gap year | 2022 | Buffer so the feature and label windows don't overlap |
| Outcome (late) period | 2023–2025 | Defines the predictive target (RQ2) |

Each period's start and end live in `config/project.yaml`; nothing hard-codes years.

### 4.3 Unit of analysis and dataset sizing
- **Pixel:** 20 m Sentinel-2 grid (10 m bands resampled to 20 m).
- **Cell:** a square **100 m × 100 m** cell (5 × 5 pixels) in the AOI's UTM grid. Cell size is a config parameter (`cell_size_m`).
- **Record types:**
  - `cells` — one row per cell (static attributes: location, elevation, slope, land-cover fractions, aridity).
  - `cell_season` — one row per cell × year × season (spectral composites + indices). **This is the main M1 table.**
  - `cell_monthly` — one row per cell × month (a few indices only) for trend analysis.
  - `cell_model` — one row per cell (early-period features + late-period label) for supervised modelling.
- **Size check:** one 20 km AOI at 100 m = 200 × 200 = **40,000 cells**. Even one AOI × 7 years × 2 seasons gives 560,000 `cell_season` rows with 25+ attributes, far above the ≥10,000 × ≥15 requirement. Several AOIs leave room to subsample.

### 4.4 Features
**Spectral (per cell-season):** mean and standard deviation (within-cell) of B02–B12 median reflectances; valid-observation count; valid-pixel fraction.

**Indices** (computed from reflectances in [0, 1]; formulas live in `src/desertification/features.py` with unit tests):

| Index | Formula (Sentinel-2 bands) | Why |
|---|---|---|
| NDVI | (B08 − B04) / (B08 + B04) | Vegetation greenness |
| EVI | 2.5 (B08 − B04) / (B08 + 6 B04 − 7.5 B02 + 1) | Greenness, less saturation, aerosol correction |
| SAVI | 1.5 (B08 − B04) / (B08 + B04 + 0.5) | Vegetation over bright soil |
| MSAVI2 | (2 B08 + 1 − √((2 B08 + 1)² − 8 (B08 − B04))) / 2 | Sparse vegetation (the standard in drylands) |
| NDMI | (B08 − B11) / (B08 + B11) | Vegetation/soil moisture |
| NDWI | (B03 − B08) / (B03 + B08) | Open water (masking) |
| BSI | ((B11 + B04) − (B08 + B02)) / ((B11 + B04) + (B08 + B02)) | Bare soil |
| TGSI | (B04 − B02) / (B04 + B02 + B03) | Topsoil grain size (sand encroachment) |
| Albedo (approx.) | 0.356 B02 + 0.130 B04 + 0.373 B08 + 0.085 B11 + 0.072 B12 − 0.0018 | Surface brightness. Liang (2001) Landsat coefficients mapped to S2 bands; **this is an approximation, so cite and state it as one** |

**Temporal (per cell, computed within a period only):** seasonal mean/min/max/amplitude of NDVI and MSAVI2; wet-minus-dry difference; inter-annual coefficient of variation; Theil–Sen slope within the period; rain-use efficiency (seasonal NDVI integral / seasonal rainfall).

**Auxiliary:** seasonal and annual CHIRPS rainfall, rainfall anomaly vs. period mean, elevation, slope, aspect (encode as sin/cos), WorldCover class fractions, aridity index, AOI id (**never used as a model feature in leave-one-AOI-out evaluation**).

**Categorical encoding:** one-hot for WorldCover dominant class and season. AOI id is only used for grouping and stratification.

### 4.5 Labelling strategy (the hardest part — read carefully)
There is no ready-made, pixel-level ground truth for desertification. We build **weak labels** from a transparent rule, then check them against an independent hand-labelled sample.

**Step A — masks.** Drop cells that are mostly water (NDWI/WorldCover), built-up, or permanent cropland under irrigation (if detectable), or that have too few valid observations (thresholds in config). Record how many cells each mask removes, per AOI.

**Step B — RESTREND (rainfall-adjusted trend), per cell:**
1. Build an annual (or growing-season) vegetation series `V_t` (e.g. growing-season integral of MSAVI2) and the matching rainfall series `P_t` (CHIRPS, the same season plus optional lag).
2. Fit `V_t = a + b·P_t + ε_t` (OLS or robust regression).
3. Take the residual series `ε_t`; estimate its **Theil–Sen slope** and run a **Mann–Kendall** test. With only 7 annual points the test has little power, so prefer the **seasonal Mann–Kendall on monthly residuals** from `cell_monthly`. Record which variant was used.
4. Do the same trend analysis on BSI and albedo (no rainfall adjustment needed).

**Step C — 3-class label (v1; thresholds tuned once and then frozen, see Decision log):**
- `degrading` — residual vegetation slope < 0 with p < α **and** (BSI slope > 0 **or** albedo slope > 0)
- `improving` — residual vegetation slope > 0 with p < α
- `stable` — everything else
- Default α = 0.05, with **Benjamini–Hochberg FDR correction** across cells within each AOI.

**Step D — predictive target (RQ2):** apply Steps B–C to the **late period (2023–2025) relative to the early period** (e.g. the change in rainfall-adjusted mean vegetation between periods, tested per cell). Features come **only** from the early period. A binary variant (`degrading` vs. not) is acceptable if the 3-class problem is too sparse; log the decision.

**Step E — independent validation sample (strongly recommended):** stratified random sample of about **300 cells** (≈100 per class per weak label, across AOIs). At least **two team members** label each cell **blind to the weak label**, using Sentinel-2 true-colour timelapses (and historical high-res imagery where available). Labels: `degrading` / `stable` / `improving` / `uncertain`. Report inter-annotator agreement (Cohen's κ) and weak-label accuracy against the consensus. Stored in `data/labels/manual_validation.csv` (committed; it is small).

**Leakage rules for labels:**
- A feature computed from the same series and window as the label (e.g. the full-period NDVI slope) **must not** be a model input for RQ2. In RQ1-style "re-detection" experiments it may only be used as a clearly labelled sanity baseline.
- Label thresholds are tuned **once**, on training AOIs/blocks only, then frozen before model selection.

### 4.6 Modelling plan (satisfies M3's "multiple models + tuning")

| Family | Models | Notes |
|---|---|---|
| Baselines | Majority class; physically-motivated rule (e.g. early-period NDVI slope threshold); Logistic Regression | Every other model has to beat these |
| Classical ML | Random Forest; Gradient Boosting (LightGBM or XGBoost); SVM (RBF, on a subsample if slow); k-NN | Tabular `cell_model` features |
| Deep learning | MLP on tabular features; **1D-CNN or GRU/LSTM** on the early-period monthly index sequences (PyTorch) | Sequence model uses `cell_monthly` directly |
| Unsupervised | K-Means and Gaussian Mixture on standardised trajectory features; HDBSCAN; **Isolation Forest** for change anomalies in Albedo–MSAVI2 space; PCA/UMAP for visualisation | Compare to labels with ARI/NMI; profile clusters |
| Stretch | Patch-based CNN on image chips; embeddings from a pre-trained EO foundation model | Only if time allows after core results |

Hyperparameter tuning: `RandomizedSearchCV` or Optuna with **spatial group CV** (§8). Class imbalance: class weights, plus SMOTE/undersampling **inside** an `imblearn.pipeline.Pipeline` so resampling only ever touches training folds; tune the decision threshold for the `degrading` class on validation folds.

---

## 5. Repository layout

Target layout (created in `P0-T01`; keep it in sync with reality):

```
cs482proj/
├── CLAUDE.md                     # standing rules for LLM-assisted dev
├── README.md                     # human quick-start
├── environment.yml               # conda env (pinned)  — or requirements.txt
├── .env.example                  # SH_CLIENT_ID=, SH_CLIENT_SECRET= (no real values)
├── config/
│   ├── project.yaml              # periods, cell size, resolution, thresholds, seeds, paths
│   ├── aoi_candidates.yaml       # candidate AOIs (M1-T02); owner sets include: true/false
│   └── aois.geojson              # AOI polygons + per-AOI properties
├── src/desertification/
│   ├── __init__.py
│   ├── config.py                 # load project.yaml/aois.geojson + .env
│   ├── sh_client.py              # SHConfig for CDSE, collections, cached Catalog search
│   ├── cache.py                  # JSON cache + provenance sidecars (request hash)
│   ├── usage.py                  # quota ledger (data/api_usage_log.csv), budget caps
│   ├── security.py               # .env secret-leak check (used by scripts/check_secrets.py)
│   ├── hardware.py               # device selection, CUDA arch compatibility
│   ├── evalscripts/              # *.js evalscripts (versioned, commented)
│   ├── catalog.py                # scene availability queries
│   ├── ingest.py                 # Process/Statistical API downloads + caching + provenance
│   ├── aux_data.py               # CHIRPS, DEM, WorldCover, aridity
│   ├── aoi.py                    # AOI squares, seasons, aridity, availability (pure)
│   ├── preprocess.py             # masking, gridding/aggregation, QA
│   ├── features.py               # indices + temporal features
│   ├── trends.py                 # Theil–Sen, Mann–Kendall, RESTREND
│   ├── labels.py                 # weak-label rules
│   ├── splits.py                 # spatial blocks, group CV, test-set lock
│   ├── models.py                 # model factories / pipelines
│   ├── evaluate.py               # metrics, CIs, comparisons
│   └── viz.py                    # maps and plots (consistent style)
├── notebooks/
│   ├── 01_data_discovery.ipynb
│   ├── 01b_aoi_selection.ipynb      # M1-T02
│   ├── 02_ingestion.ipynb
│   ├── 03_cleaning_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_labeling.ipynb
│   ├── 06_unsupervised.ipynb
│   ├── 07_supervised.ipynb
│   ├── 08_deep_learning.ipynb
│   ├── 09_interpretation.ipynb
│   └── final/desertification_pipeline.ipynb   # M3 graded, end-to-end, reproducible
├── data/                         # raw/ interim/ are gitignored
│   ├── raw/                      # API responses (GeoTIFF/JSON) + provenance sidecars
│   ├── interim/                  # cell-gridded arrays
│   ├── processed/                # parquet tables (commit if total < ~50 MB; see §3.4)
│   ├── api_usage_log.csv         # PU/request ledger (committed)
│   ├── sample/                   # small committed samples for M1 report + tests
│   └── labels/                   # manual_validation.csv (committed)
├── reports/
│   ├── figures/                  # every figure is generated by code; filename = figure id
│   ├── results/                  # metric CSVs/JSON produced by code (§8)
│   ├── milestone1/  milestone2/  milestone3/
├── scripts/
│   ├── build_pdf.sh              # Markdown report → PDF (pandoc)
│   ├── check_env.py              # verify imports, versions, GPU
│   └── check_secrets.py          # fail if a .env value appears in a file git would commit
├── tests/                        # pytest unit tests (indices, trends, splits, labels)
└── docs/
    ├── ROADMAP.md  Milestone1-3.md
    ├── DECISIONS.md              # decision log (append-only)
    └── REFERENCES.md             # all citations (data, papers, libraries, code reused)
```

---

## 6. Phased task plan

> Week numbers are relative to each milestone's start. Put real due dates in the [Status tracker](#10-status-tracker) once you know them.

### Phase 0 — Project setup (before / at the start of M1)

#### P0-T01 — Scaffold the repository
M1 wk1 | Depends on: — | Owner: TBD
**Goal:** create the directory layout, packaging and tooling so every later task has a place to put its work.
**Inputs:** §5 of this file.
**Outputs:** the directories in §5 (with `.gitkeep` where empty); `pyproject.toml` making `src/desertification` installable (`pip install -e .`); `README.md`; `.env.example`; additions to `.gitignore` (`data/raw/`, `data/interim/`, `.env`, `*.tif`, `*.tiff`, `.ipynb_checkpoints/`, `wandb/`, `mlruns/`); `tests/test_smoke.py`.
**Steps:** create the layout → write `pyproject.toml` (Python ≥ 3.11) → minimal `config.py` that loads `config/project.yaml` → smoke test that imports the package and loads the config.
**Acceptance:** `pip install -e .` succeeds in a fresh env · `pytest` passes · `git status` shows no data or secrets tracked · README has setup steps that a teammate has followed successfully.

#### P0-T02 — Environment and dependencies
M1 wk1 | Depends on: P0-T01
**Goal:** a pinned, reproducible environment on the **local machine with an NVIDIA GTX 1080 Ti** (Pascal, compute capability 6.1, 11 GB VRAM). Colab is an optional fallback only.
**Outputs:** `environment.yml` (and/or `requirements.txt`) with pinned versions; GPU setup notes in README; `scripts/build_pdf.sh` (Markdown → PDF with pandoc, see M1-T10).
**GPU note:** recent PyTorch/CUDA builds have been dropping older GPU architectures. Pick a PyTorch + CUDA combination whose `torch.cuda.get_arch_list()` covers compute capability 6.1 (`sm_60` or `sm_61` kernels, or older-or-equal `compute_6x` PTX), and check it actually runs a tensor operation on the GPU. Don't assume the latest wheel works. Record the working combination in the Decision log.
**Report tooling:** `pandoc` plus a PDF engine (e.g. `xelatex` from TeX Live). Record the versions.
**Core packages:** `sentinelhub`, `numpy`, `pandas`, `pyarrow`, `xarray`, `rioxarray`, `rasterio`, `geopandas`, `shapely`, `pyproj`, `scipy`, `statsmodels`, `pymannkendall`, `scikit-learn`, `imbalanced-learn`, `lightgbm` (or `xgboost`), `torch`, `shap`, `optuna` (optional), `matplotlib`, `seaborn`, `folium` or `leafmap`, `pystac-client`, `planetary-computer`, `python-dotenv`, `pyyaml`, `tqdm`, `jupyter`, `nbconvert`, `papermill`, `pytest`, `ruff`.
**Acceptance:** env builds from scratch · `python -c "import sentinelhub, sklearn, torch, geopandas"` works · `torch.cuda.is_available()` is `True` and a small matrix multiply runs on the 1080 Ti · `scripts/build_pdf.sh` turns a test Markdown file with a figure into a PDF · versions recorded in the Decision log.

#### P0-T03 — CDSE credentials and connectivity check
M1 wk1 | Depends on: P0-T02
**Goal:** prove authenticated access to Sentinel Hub on CDSE.
**Outputs:** `src/desertification/sh_client.py`; `notebooks/01_data_discovery.ipynb` section "Connectivity check".
**Steps:** register on CDSE → create an OAuth client in the dashboard → put `SH_CLIENT_ID`/`SH_CLIENT_SECRET` in `.env` (never committed) → run a Catalog API search over one AOI for one month → print the scene count and cloud cover.
**Acceptance:** catalog search returns ≥1 scene · no secret in any committed file or notebook output · `data/api_usage_log.csv` created with the first entry and the starting balance (30,000 PU / 30,000 requests).

### Phase 1 — Milestone 1, Week 1: research, discovery, feasibility

#### M1-T01 — Document candidate data sources
M1 wk1 | Depends on: P0-T03
**Goal:** a written comparison of ≥2 sources (Sentinel Hub/CDSE and Planetary Computer STAC), plus the auxiliary datasets.
**Outputs:** `reports/milestone1/sources.md` (feeds the PDF).
**Content required for each source:** endpoint URLs; auth method; documentation links; data structure (raster tiles / STAC items / JSON from the Statistical API); spatial/temporal resolution and coverage; rate limits/quotas; licence and terms of use; why it is relevant to desertification; pros and cons.
**Acceptance:** both sources have every field filled in from official docs (with links) · terms/licence stated · one paragraph on why the topic is interesting (personal motivation + societal relevance, e.g. UN SDG 15.3).

#### M1-T02 — Select and validate AOIs
M1 wk1 | Depends on: P0-T03
**Goal:** pin down the 4–6 AOIs.
**Outputs:** `config/aois.geojson`; an AOI map figure `reports/figures/m1_aoi_map.png`; AOI rationale table in `reports/milestone1/sources.md`.
**Steps:** for each candidate in §4.1, check the aridity index, Catalog API scene counts per year (2019–2025) and median cloud cover, and look at a true-colour composite → fix the bbox (about 20 km × 20 km), UTM EPSG and wet/dry season months.
**Acceptance:** every AOI is a dryland (AI ≤ 0.65) or explicitly a control · ≥ 20 usable scenes per year per AOI, or a documented exception · seasons justified with a citation or a rainfall climatology plot · map renders.

#### M1-T03 — Feasibility and scale justification
M1 wk1 | Depends on: M1-T02
**Goal:** show the source can produce > 10,000 records with ≥ 15 attributes.
**Outputs:** a section in `reports/milestone1/sources.md` with the arithmetic (cells × years × seasons; attribute list).
**Acceptance:** numbers come from the actual AOI sizes in `aois.geojson` · the attribute list matches §7 · PU cost estimated from the pilot (M1-T05) or flagged as pending.

### Phase 2 — Milestone 1, Week 2: extraction, parsing, initial exploration

#### M1-T04 — Instructor discussion
M1 wk2 | Depends on: M1-T01..T03 · **Deferred (2026-11-20): not blocking; development goes ahead without it**
**Goal:** get feedback on the source and project idea.
**Outputs:** `reports/milestone1/instructor_discussion.md`: date, attendees, questions asked, feedback received, decision made.
**Acceptance:** the decision and any scope changes are copied into the Decision log, and this roadmap is updated if needed. *(People do this task; an LLM may only help prepare the questions.)*

#### M1-T05 — Pilot extraction (one AOI, one season)
M1 wk2 | Depends on: P0-T03, M1-T02
**Goal:** working, cached, provenance-tracked extraction for a small request, plus a measured PU cost.
**Outputs:** `src/desertification/evalscripts/median_composite.js`; `src/desertification/ingest.py` (`download_composite(aoi_id, start, end, resolution) -> Path`); GeoTIFF in `data/raw/<aoi_id>/<start>_<end>_<res>m.tif` plus a JSON sidecar (request params, evalscript hash, timestamp, CRS, sentinelhub version); section in `notebooks/02_ingestion.ipynb`.
**Steps:** implement the request (UTM CRS, 20 m) → cache: skip the download if the file and a matching sidecar already exist → download one AOI × one season → plot true colour + NDVI + valid-count band → record the PU used (from the dashboard) in `data/api_usage_log.csv` and the per-request cost in the Decision log.
**Acceptance:** re-running does **not** hit the API again (cache works) · valid-count band looks sensible (no large holes over bright sand; if there are, see gotcha #2) · reflectances fall in a plausible 0–1 range · harmonisation check across 2022-01-25 done on a stable target (gotcha #1).

#### M1-T06 — Scaled extraction
M1 wk2 | Depends on: M1-T05 · **Ask the user before running** (spends quota)
**Goal:** composites for all AOIs × years × seasons (`cell_season` source rasters) and monthly index rasters (`cell_monthly` source).
**Outputs:** `data/raw/...` files with sidecars; `evalscripts/monthly_indices.js` (outputs NDVI, MSAVI2, BSI, albedo, valid count from B02, B04, B08, B11, B12, SCL — fewer bands, cheaper); a download log `data/raw/download_log.csv`.
**Steps:** estimate total PU from the pilot → confirm it fits the **20,000 PU** extraction cap in §3.4 (or reduce: coarser resolution for monthly, fewer AOIs) → show the estimate to the user and get approval → batch-download with retries and backoff → log failures.
**Acceptance:** every planned request is present or listed as failed with a reason · total PU used recorded in `data/api_usage_log.csv` and within the cap · no duplicated downloads · re-running the extraction makes zero API calls.

#### M1-T07 — Auxiliary data ingestion
M1 wk2 | Depends on: M1-T02
**Goal:** CHIRPS monthly rainfall, DEM, WorldCover and aridity for each AOI, clipped and reprojected to each AOI's 20 m UTM grid (rainfall may stay coarse but must be sampled per cell).
**Outputs:** `src/desertification/aux_data.py`; files in `data/raw/aux/<dataset>/<aoi_id>/...` with sidecars.
**Acceptance:** each layer aligns with the S2 grid (an overlay plot shows correct registration) · licences/citations added to `docs/REFERENCES.md`.

#### M1-T08 — Parse to tables and do the initial exploration
M1 wk2 | Depends on: M1-T06, M1-T07
**Goal:** grid the rasters into 100 m cells and build the `cells`, `cell_season` and `cell_monthly` tables.
**Outputs:** `src/desertification/preprocess.py`, `features.py`; `data/processed/cells.parquet`, `cell_season.parquet`, `cell_monthly.parquet`; `data/sample/cell_season_sample.csv` (≈200 rows for the report); the first half of `notebooks/03_cleaning_eda.ipynb`.
**Acceptance:** schemas match §7 exactly (column names, dtypes, units) · `df.info()`, `df.describe()` and row counts per AOI/year shown · ingestion problems written up (quota, clouds/missing data, CRS, file sizes, run time).

### Phase 3 — Milestone 1, Week 3: cleaning, preliminary analysis, report

#### M1-T09 — Cleaning and preliminary statistics
M1 wk3 | Depends on: M1-T08
**Goal:** a clean, analysis-ready `cell_season` table with documented preprocessing.
**Outputs:** second half of `notebooks/03_cleaning_eda.ipynb`; figures `reports/figures/m1_*.png`.
**Required analysis:** missing-data analysis (by AOI, season, year; a missingness heatmap) · handling strategy (drop below a valid-obs threshold vs. impute; justified) · duplicate check (cell_id × year × season unique) · type fixes · outlier inspection (reflectance > 1, negative values) · feature distributions (histograms/violins by AOI) · correlation matrix (expect highly correlated bands/indices; note which ones are redundant) · NDVI/MSAVI2 time series per AOI with rainfall overlay · early insight paragraph(s).
**Acceptance:** every cleaning step says how many rows/values it affected · no unexplained NaNs remain · correlated feature groups identified for M3 · ≥ 6 informative, labelled figures.

#### M1-T10 — Milestone 1 report
M1 wk3 | Depends on: M1-T01..T09
**Outputs:** `reports/milestone1/milestone1_report.md` → `reports/milestone1/milestone1_report.pdf` via `scripts/build_pdf.sh` (pandoc). The report links to the relevant notebooks on GitHub.
**Required sections:** (1) Data sources: endpoints/URLs, docs, auth, terms, and a code snippet of the pull; (2) Why this data is interesting; (3) Feasibility: records × attributes; (4) Instructor discussion summary; (5) Description of the data: records, attributes, types, sample rows, descriptive stats; (6) Cleaning and preliminary analysis with figures; (7) Proposed data-mining directions (RQ1–RQ4); (8) References.
**Acceptance:** every requirement in `Milestone1.md` is checked off in a table at the end of the draft (remove the table before submitting if you like) · all figures have captions and the Copernicus attribution · PDF builds.

### Phase 4 — Milestone 2: review of existing work

#### M2-T01 — Literature and code survey
M2 wk1 | Depends on: M1-T10
**Goal:** find 6–10 strong prior works (papers **with code** where possible) on dryland degradation / desertification / land-cover change from satellite data.
**Outputs:** `reports/milestone2/survey_notes.md`: for each work: citation, link, code link, data, task, method, metrics, results, what impressed us, techniques not covered in class.
**Search directions:** RESTREND / TSS-RESTREND; BFAST and LandTrendr change detection; Trends.Earth SDG 15.3.1 implementation; eo-learn land-cover workflows; EuroSAT / BigEarthNet deep learning; Sahel greening debates; Albedo–NDVI/MSAVI feature-space desertification indices; Dynamic World; EO foundation models.
**Acceptance:** every citation is **verified** (it opens, and the authors/year/venue match) · every code link opens · nothing cited that nobody on the team has looked at.

#### M2-T02 — Run or read one reference implementation
M2 wk1 | Depends on: M2-T01
**Goal:** understand at least one existing codebase deeply enough to describe how it works (e.g. an eo-learn land-cover example, or the Trends.Earth productivity sub-indicator).
**Outputs:** `reports/milestone2/code_walkthrough.md` with file/function references and links.
**Acceptance:** describes the pipeline stages, key design choices, and at least 2 techniques not covered in class.

#### M2-T03 — Position our contribution
M2 wk1–2 | Depends on: M2-T01
**Goal:** confirm or revise §1.4 against what already exists.
**Outputs:** updates to §1.4 and the Decision log; a "Proposed extensions" section draft.
**Acceptance:** each claimed contribution names the closest prior work and says concretely how ours differs.

#### M2-T04 — Milestone 2 report (2–3 pages)
M2 wk2 | Depends on: M2-T01..T03
**Outputs:** `reports/milestone2/milestone2_report.md` → PDF via `scripts/build_pdf.sh`.
**Required sections:** learning tasks that impress us · how they were accomplished · code references with links · techniques not covered in class · our variants/extensions and the proposed new contribution · references.
**Acceptance:** 2–3 pages · well organised · every claim about others' work is backed by a citation.

### Phase 5 — Milestone 3, Weeks 1–2: group formation, data preparation, labels

#### M3-T01 — Team and work split
M3 wk1 | Depends on: —
**Outputs:** `reports/milestone3/team.md`: members, roles, task ownership (fill the Owner column in §10), communication plan, internal deadlines.
**Acceptance:** each member owns at least one modelling task **and** one report/video section (needed for "individual contributions" in the video).

#### M3-T02 — Masks and quality filtering
M3 wk1 | Depends on: M1-T09
**Outputs:** masking functions in `preprocess.py`; mask stats table in `notebooks/04_feature_engineering.ipynb`.
**Acceptance:** cells removed per AOI per mask reported · thresholds in `project.yaml` · unit tests for mask logic.

#### M3-T03 — Feature engineering
M3 wk1 | Depends on: M3-T02
**Goal:** build early-period features (§4.4) into `cell_model` and the monthly sequence tensors.
**Outputs:** `features.py` (temporal features); `data/processed/cell_model.parquet`; `data/processed/seq_early.npz` (cells × months × channels + cell_ids); `notebooks/04_feature_engineering.ipynb`.
**Required:** missing-value handling (documented impute strategy, fitted on training folds only when it's learned imputation) · encoding (one-hot as in §4.4) · scaling (inside model pipelines, not globally before splitting) · correlation/VIF analysis and a feature-reduction decision · feature distribution plots by AOI.
**Acceptance:** no feature uses data from after 2021 (assert this in code) · every feature documented in §7 · unit tests for every index formula against hand-computed values.

#### M3-T04 — Trend analysis and weak labels
M3 wk1–2 | Depends on: M3-T02, M1-T07
**Outputs:** `trends.py`, `labels.py`; `data/processed/labels.parquet` (cell_id, full-period label, predictive label, slopes, p-values, q-values); `notebooks/05_labeling.ipynb`; label maps per AOI `reports/figures/m3_labels_<aoi>.png`.
**Acceptance:** follows §4.5 exactly, or deviations are logged · class balance reported per AOI · label maps look plausible against true-colour imagery (spot-check ≥ 10 cells per class, notes recorded) · thresholds frozen and logged.

#### M3-T05 — Manual validation sample
M3 wk2 | Depends on: M3-T04
**Outputs:** `data/labels/manual_validation.csv` (cell_id, annotator, label, confidence, notes); a sampling script; agreement analysis in `05_labeling.ipynb`.
**Acceptance:** ≥ 2 annotators blind to the weak label · Cohen's κ reported · weak-label confusion matrix vs. consensus. *(Annotation is done by people. An LLM may build the sampling and timelapse-viewing tooling.)*

#### M3-T06 — Splits and the test-set lock
M3 wk2 | Depends on: M3-T03, M3-T04
**Outputs:** `splits.py`; `data/processed/splits.parquet` (cell_id, spatial_block_id, fold, is_test).
**Steps:** assign spatial blocks (e.g. 2 km × 2 km; larger than the range of spatial autocorrelation — estimate it with a variogram or correlogram on NDVI) → hold out a **test set** (e.g. 20% of blocks, stratified by AOI and label; plus the leave-one-AOI-out protocol) → build `StratifiedGroupKFold` folds on the rest.
**Acceptance:** no block appears in both train and test · test-set hash recorded in the Decision log · a test proves no cell_id overlap.

#### M3-T07 — Class imbalance analysis
M3 wk2 | Depends on: M3-T06
**Outputs:** section in `07_supervised.ipynb`: class ratios per AOI/fold; the chosen strategies to compare (class weights, SMOTE, random undersampling, threshold tuning).
**Acceptance:** resampling only happens inside training folds (verified by code structure) · strategy comparison reported with macro-F1 and `degrading`-class recall.

### Phase 6 — Milestone 3, Week 3: modelling and experimentation (40% of the grade)

#### M3-T08 — Baselines
M3 wk3 | Depends on: M3-T06
**Outputs:** `models.py` baseline factories; results in `07_supervised.ipynb`; `reports/results/baselines.csv`.
**Acceptance:** majority, rule-based and logistic-regression results on the CV folds with all §8 metrics.

#### M3-T09 — Classical ML with tuning
M3 wk3 | Depends on: M3-T08
**Outputs:** tuned RF, gradient boosting, SVM, k-NN; `reports/results/cv_results.csv` (model, params, fold, metrics); best params saved to `reports/results/best_params.json`.
**Acceptance:** search spaces documented · tuning uses spatial group CV only · tuning budget (iterations and run time) reported · mean ± std across folds for each metric.

#### M3-T10 — Deep learning
M3 wk3 | Depends on: M3-T06
**Outputs:** `notebooks/08_deep_learning.ipynb`; MLP and 1D-CNN/GRU on `seq_early.npz`; training curves; results appended to `cv_results.csv`.
**Acceptance:** fixed seeds · early stopping on a validation split taken from training folds only · class-weighted loss or a sampler · trains on the local GTX 1080 Ti (11 GB VRAM: size batches accordingly), with device chosen automatically so it still runs, slowly, on CPU · training time and hardware reported · compared fairly with the tabular models on the same folds.

#### M3-T11 — Unsupervised analysis
M3 wk3 | Depends on: M3-T03, M3-T04
**Outputs:** `notebooks/06_unsupervised.ipynb`; cluster maps; Albedo–MSAVI2 change-vector / Isolation Forest anomaly maps.
**Acceptance:** k chosen with evidence (elbow/silhouette/BIC) · clusters profiled (mean feature values, physical interpretation) · agreement with weak labels and manual labels (ARI, NMI, contingency table) · anomalies spot-checked on imagery.

#### M3-T12 — Final evaluation on the locked test set
M3 wk3–4 | Depends on: M3-T09, M3-T10
**Outputs:** `reports/results/test_results.csv`; leave-one-AOI-out table; confusion matrices; bootstrap 95% CIs (resampling spatial blocks); McNemar or paired-bootstrap test between the top 2 models; evaluation against the manual validation sample.
**Acceptance:** the test set is evaluated **once** per final model (log the date) · no model or threshold changes after seeing test results (or, if there were, they are disclosed in the report).

### Phase 7 — Milestone 3, Week 4: interpretation, report, video

#### M3-T13 — Interpretation
M3 wk4 | Depends on: M3-T12
**Outputs:** `notebooks/09_interpretation.ipynb`; SHAP summary + dependence plots for the best tree model; permutation importance; prediction/error maps per AOI; error analysis by land cover, AOI and rainfall regime; comparison of our results with prior work from M2.
**Acceptance:** every key finding is linked to a figure or table · physical plausibility discussed · limitations stated (weak labels, short record, sensor change, AOI selection bias).

#### M3-T14 — Final notebook, report and video
M3 wk4 | Depends on: M3-T13
**Outputs:**
- `notebooks/final/desertification_pipeline.ipynb`: runs top-to-bottom ("Restart & Run All") on the local machine from `data/processed/`, with a clearly marked optional cell to re-download raw data from CDSE (off by default); prints library versions, hardware and the git commit. It is **committed with its outputs**, so a reader following the report's link sees every result without re-running it. Run time is reported in its header cell.
- `reports/milestone3/milestone3_report.md` → PDF via `scripts/build_pdf.sh` with every section `Milestone3.md` requires: project name, members, **work distribution**; data knowledge (types, distributions, issues: correlation, missingness, transformations); preprocessing steps and results; model experiments with metrics; justification for model choices and tuning; interpretation; comparison with prior work and our contribution; **link to the executable notebook**; **link to the video**.
- Video (≤ 30 min, aim for 15–20): intro (purpose, objectives, dataset) → preprocessing and methods → results and model comparison → conclusion, lessons learned, next steps; every member speaks and states their contributions; slides plus notebook screen recording; uploaded as unlisted / anyone-with-link.
**Acceptance:** the notebook passes `jupyter nbconvert --to notebook --execute` in a freshly built environment on the local machine · the committed copy has its outputs and opens on GitHub · the video link has been tested in an incognito window · the report has a requirement checklist cross-referenced against `Milestone3.md` · every group member has the final PDF to submit.

---

## 7. Data dictionary

All reflectances are unitless surface reflectance in [0, 1] (harmonised). Lengths are in metres, rainfall in mm. CRS is stored per AOI.

### 7.1 `cells.parquet` (one row per cell)
| Column | Type | Description |
|---|---|---|
| `cell_id` | str | `<aoi_id>_<row>_<col>` (unique) |
| `aoi_id` | category | AOI identifier |
| `row`, `col` | int32 | Cell indices in the AOI grid |
| `x_utm`, `y_utm` | float64 | Cell centre in the AOI's UTM CRS (m) |
| `lon`, `lat` | float64 | Cell centre, WGS84 |
| `utm_epsg` | int32 | EPSG code |
| `spatial_block_id` | str | Assigned in M3-T06 |
| `elevation_m`, `slope_deg` | float32 | From Copernicus DEM |
| `aspect_sin`, `aspect_cos` | float32 | Encoded aspect |
| `wc_<class>_frac` | float32 | WorldCover class fractions (one column per class) |
| `wc_dominant` | category | Dominant WorldCover class |
| `aridity_index` | float32 | Global Aridity Index |
| `mask_*` | bool | Mask flags (water, built-up, low-data, ...) |

### 7.2 `cell_season.parquet` (one row per cell × year × season) — main M1 table
| Column | Type | Description |
|---|---|---|
| `cell_id`, `aoi_id` | str / category | Keys |
| `year` | int16 | Year (for a wet season crossing a year boundary, use the start year; documented) |
| `season` | category | `wet` / `dry` (AOI-specific months) |
| `B02_mean` … `B12_mean` | float32 | Mean over cell pixels of the per-pixel median composite (10 bands) |
| `B02_std` … `B12_std` | float32 | Within-cell standard deviation (optional, config flag) |
| `NDVI`, `EVI`, `SAVI`, `MSAVI2`, `NDMI`, `NDWI`, `BSI`, `TGSI`, `albedo` | float32 | Indices computed from the cell-mean reflectances **and** documented as such (or per-pixel then averaged — choose one, log it) |
| `valid_obs_median` | float32 | Median per-pixel valid-observation count |
| `valid_pixel_frac` | float32 | Fraction of cell pixels with ≥ 1 valid obs |
| `precip_season_mm` | float32 | CHIRPS seasonal total |

### 7.3 `cell_monthly.parquet` (one row per cell × month)
`cell_id`, `aoi_id`, `date` (month start), `NDVI`, `MSAVI2`, `BSI`, `albedo`, `valid_obs`, `precip_mm`.

### 7.4 `cell_model.parquet` (one row per cell — supervised input)
Keys + early-period features (§4.4) + static `cells` attributes. **No** columns derived from 2022 onwards. Target columns live in `labels.parquet` and are joined explicitly in the modelling code.

### 7.5 `labels.parquet`
`cell_id`, `veg_resid_slope`, `veg_resid_p`, `veg_resid_q`, `bsi_slope`, `bsi_p`, `albedo_slope`, `albedo_p`, `label_full` (3-class, 2019–2025), `label_pred` (3-class or binary, late vs. early), `label_version`.

---

## 8. Evaluation protocol

- **Primary metric:** macro-F1 (classes are imbalanced, and `degrading` matters most).
- **Also report:** balanced accuracy, per-class precision/recall/F1, Cohen's κ, confusion matrix, PR-AUC for `degrading` (one-vs-rest), and calibration (reliability plot) for the best model.
- **Cross-validation:** `StratifiedGroupKFold` (k = 5) grouped by `spatial_block_id`; **plus** leave-one-AOI-out to test transfer. **Never** use a plain random K-fold on cells: spatial autocorrelation inflates scores. If you show one for contrast, label it as such.
- **Test set:** locked in M3-T06, evaluated once per final model (M3-T12).
- **Uncertainty:** mean ± std across folds; 95% CIs by block bootstrap on the test set.
- **Model comparison:** paired comparison on the same folds; McNemar or paired bootstrap for the top two models.
- **Unsupervised:** silhouette, Davies–Bouldin, BIC (GMM); external agreement with labels via ARI/NMI.
- **Ground-truth check:** accuracy of weak labels and of model predictions against the manual validation sample.
- **Reporting:** every number in a report comes from a CSV in `reports/results/` produced by code. No hand-typed metrics.

---

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| PU/request quota (30,000 each) runs out | Ingestion stalls | Budget caps in §3.4; usage ledger; pilot first (M1-T05); cache everything; coarser monthly resolution; fewer AOIs; Planetary Computer fallback (Decision log entry) |
| Bright desert misclassified as cloud by SCL | Data holes, biased composites | Track valid counts; per-AOI mask tweak; visual QA |
| Weak labels are circular or noisy | Inflated or meaningless scores | Strict early/late separation; manual validation sample; report label uncertainty |
| Spatial autocorrelation leakage | Over-optimistic metrics | Spatial blocks + leave-one-AOI-out |
| Rainfall variability mistaken for degradation | False positives | RESTREND residuals; rainfall features; Sahel greening context |
| Short record (S2 2017+; analysis 2019–2025) | Low trend power | Monthly seasonal Mann–Kendall; FDR control; discuss as a limitation; optional Landsat context |
| Processing baseline change in 2022 | Artificial trend | Rely on Sentinel Hub harmonisation; verify on a stable target |
| Severe class imbalance (few `degrading`) | Poor recall | Class weights, SMOTE in folds, threshold tuning, PR-AUC |
| Irrigation / cropland dynamics confused with degradation | Label noise | WorldCover cropland masks/features; error analysis by land cover |
| Notebook not reproducible for the grader | Lost marks | Final notebook committed with outputs; processed data committed (or released if too large); clean-env execution test; pinned env |
| Local disk loss (raw data only exists locally) | Re-downloads cost quota | Back up `data/raw/` outside the repo after M1-T06; provenance sidecars allow exact re-requests |
| GTX 1080 Ti not supported by the latest PyTorch build | DL tasks blocked | torch pinned to `2.14.1+cu126` (has `sm_60` kernels); `scripts/check_env.py --require-gpu`; automatic CPU fallback via `hardware.select_device()` |
| LLM-introduced errors (wrong formulas, fabricated APIs/citations) | Silent bugs, academic-integrity problems | Unit tests for formulas; citation verification; CLAUDE.md integrity rules; human review against acceptance criteria |

---

## 10. Status tracker

Fill in owners and due dates. Tick tasks as they finish (`[x]`, date, note). `[~]` = done except a step that needs a person or the local machine (see Notes).

| ✓ | ID | Task | Owner | Due | Notes |
|---|---|---|---|---|---|
| [x] | P0-T01 | Scaffold repository | | 2026-11-20 | Layout, pyproject, config loader + validation, smoke tests. README setup followed successfully on the local machine (2026-11-20). Module stubs are left to their own tasks |
| [x] | P0-T02 | Environment & dependencies | | 2026-11-20 | Pinned Python 3.12 env; verified in a fresh cloud venv (CPU) and on the local machine with the GTX 1080 Ti: all imports OK, GPU matmul OK, 15/15 tests pass (Decision log #17) |
| [x] | P0-T03 | CDSE credentials & connectivity | | 2026-11-20 | Local run: Catalog search over `ferlo_test`, Jan 2023 returned 6 scenes (tile 28PEC; cloud cover 0–98.7%, median 37.2%; 2 scenes < 10%). Ledger updated; 0 PU and 1 request charged |
| [x] | M1-T01 | Document candidate sources | | 2026-11-20 | `reports/milestone1/sources.md`: both sources and the auxiliary datasets documented from official docs (links, quotes, access date), with the project owner's personal-motivation paragraph; builds to PDF with no warnings. Aridity-index facts left `[VERIFY]` for M1-T02 |
| [ ] | M1-T02 | Select & validate AOIs | | | |
| [ ] | M1-T03 | Feasibility & scale | | | |
| [ ] | M1-T04 | Instructor discussion | | | Deferred, non-blocking |
| [ ] | M1-T05 | Pilot extraction | | | |
| [ ] | M1-T06 | Scaled extraction | | | |
| [ ] | M1-T07 | Auxiliary data | | | |
| [ ] | M1-T08 | Parse to tables & initial exploration | | | |
| [ ] | M1-T09 | Cleaning & preliminary stats | | | |
| [ ] | M1-T10 | **Milestone 1 report** | | | |
| [ ] | M2-T01 | Literature & code survey | | | |
| [ ] | M2-T02 | Reference implementation walkthrough | | | |
| [ ] | M2-T03 | Position contribution | | | |
| [ ] | M2-T04 | **Milestone 2 report** | | | |
| [ ] | M3-T01 | Team & work split | | | Deferred until group is formed |
| [ ] | M3-T02 | Masks & quality filtering | | | |
| [ ] | M3-T03 | Feature engineering | | | |
| [ ] | M3-T04 | Trend analysis & weak labels | | | |
| [ ] | M3-T05 | Manual validation sample | | | |
| [ ] | M3-T06 | Splits & test-set lock | | | |
| [ ] | M3-T07 | Class imbalance analysis | | | |
| [ ] | M3-T08 | Baselines | | | |
| [ ] | M3-T09 | Classical ML + tuning | | | |
| [ ] | M3-T10 | Deep learning | | | |
| [ ] | M3-T11 | Unsupervised analysis | | | |
| [ ] | M3-T12 | Final test evaluation | | | |
| [ ] | M3-T13 | Interpretation | | | |
| [ ] | M3-T14 | **Final notebook, Milestone 3 report, video** | | | |

---

## 11. Decision log

Moved to [`docs/DECISIONS.md`](DECISIONS.md) (append-only, numbered entries). References like "Decision log #22" elsewhere in this roadmap point there.

---

## 12. Open questions

Resolve these and record the answer in [`DECISIONS.md`](DECISIONS.md).

1. Final AOI list and bounding boxes. *To be settled in M1-T02.*
2. PU cost per Process API request. *Measure in M1-T05.* (Catalog cost answered in Decision log #22.)
3. Instructor feedback on scope, label strategy and whether weak labels are acceptable. *Deferred (M1-T04).*
4. Group members for M3 and whether M1/M2 are individual or group submissions. *Deferred (M3-T01).*
5. Whether `data/processed/` stays under ~50 MB or needs a GitHub Release asset. *Revisit after M1-T08.*
6. *(Resolved 2026-11-20: quotas reset monthly, but the project is done within one month on the 30,000 / 30,000 allowance; Decision log #28.)*
7. Global Aridity Index: version (v3?), resolution, licence and citation could not be read (figshare blocked automated access). *Verify in M1-T02 before use.*
8. Trends.Earth licence: README says GPL v2.0 or later, but the repository also contains an Apache-2.0 file. *Verify before citing (only if used).*
9. *(Resolved 2026-11-20: personal-motivation paragraph added by the project owner.)*

---

## 13. Prompt templates

Copy, fill in `<…>`, and paste.

**Implement a task**
```
Read CLAUDE.md, then docs/ROADMAP.md §1–§5 and task <TASK-ID>.
Implement <TASK-ID> exactly as specified. Put reusable logic in src/desertification/,
keep notebooks thin, and add unit tests for any new pure function.
Before finishing: check every acceptance criterion and report each one as met / not met (with evidence).
Then update the Status tracker and Open questions in ROADMAP.md, and the Decision log in docs/DECISIONS.md.
Do not call the Sentinel Hub API beyond <N> requests without asking me first.
```

**Review a task's output**
```
Act as a strict reviewer. Read CLAUDE.md and task <TASK-ID> in docs/ROADMAP.md.
Review <files/notebook> against the acceptance criteria and CLAUDE.md rules.
Look especially for: data leakage, non-spatial CV, wrong index formulas or units,
unverified citations, hard-coded paths/years, secrets in outputs, unreproducible cells.
List concrete problems with file:line references. Don't fix anything yet.
```

**Draft a report section**
```
Read CLAUDE.md and docs/Milestone<N>.md. Draft section "<section>" of
reports/milestone<N>/milestone<N>_report.md using ONLY numbers from reports/results/*.csv,
data/processed/* summaries and figures in reports/figures/. Cite every figure by filename.
Mark anything you cannot support with data as [TODO: …]. Do not invent citations;
use only entries already in docs/REFERENCES.md, or flag new ones as [VERIFY].
```

**Debug**
```
Read CLAUDE.md. <Describe symptom, paste the traceback/output>.
Find the root cause before changing code. Explain the cause, the fix, and add a regression test.
```

**Literature help (Milestone 2)**
```
Read CLAUDE.md and docs/Milestone2.md. For the topic "<topic>", suggest search queries and
candidate works. For each, give only what you are confident of and mark everything else [VERIFY].
I will check every reference myself before it goes into docs/REFERENCES.md.
```

---

## 14. Starting reference list

A **starting point to verify, not a bibliography**. Check each one (authors, year, venue, link) before citing, then move it into `docs/REFERENCES.md`.

**Data and tools**
- Copernicus Sentinel-2 mission and L2A product documentation (ESA).
- Copernicus Data Space Ecosystem — Sentinel Hub API documentation.
- sentinelhub-py and eo-learn (Sinergise / Planet), GitHub: `sentinel-hub/sentinelhub-py`, `sentinel-hub/eo-learn`.
- Funk et al. (2015), CHIRPS rainfall dataset, *Scientific Data*.
- Zanaga et al., ESA WorldCover 10 m 2020/2021 product documentation.
- Trabucco & Zomer, Global Aridity Index and Potential Evapotranspiration database.
- Microsoft Planetary Computer STAC API documentation.
- Hussain & Mahmud (2019), pyMannKendall, *Journal of Open Source Software*.

**Methods and prior work**
- UNCCD definition of desertification; UN SDG indicator 15.3.1 good-practice guidance; Trends.Earth (Conservation International), GitHub: `ConservationInternational/trends.earth`.
- Evans & Geerken (2004), separating climate- and human-induced dryland degradation (RESTREND), *Journal of Arid Environments*.
- Wessels et al. (2007), RESTREND applied in South Africa, *Journal of Arid Environments*.
- Burrell, Evans & Liu (2017), TSS-RESTREND, *Remote Sensing of Environment*.
- Verbesselt et al. (2010), BFAST, *Remote Sensing of Environment*.
- Kennedy, Yang & Cohen (2010), LandTrendr, *Remote Sensing of Environment*.
- Fensholt et al. (2012), greening of the world's drylands, *Remote Sensing of Environment*.
- Liang (2001), narrowband-to-broadband albedo conversion, *Remote Sensing of Environment*.
- Qi et al. (1994), MSAVI, *Remote Sensing of Environment*; Huete (1988), SAVI.
- Xiao et al. (2006), Topsoil Grain Size Index (TGSI) for desertification.
- Helber et al. (2019), EuroSAT, *IEEE JSTARS*; Sumbul et al. (2019), BigEarthNet.
- Brown et al. (2022), Dynamic World, *Scientific Data*.
- Roberts et al. (2017), cross-validation strategies for data with spatial/temporal structure, *Ecography*.
- Lundberg & Lee (2017), SHAP, *NeurIPS*.
- Chawla et al. (2002), SMOTE, *JAIR*.
