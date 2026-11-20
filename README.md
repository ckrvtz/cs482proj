# Desertification detection with Sentinel-2

CS482 data mining project: detect and predict desertification in drylands from free
Sentinel-2 L2A imagery, accessed through the Sentinel Hub APIs on the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) (CDSE).

- **Plan, tasks and study design:** [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **Rules for LLM-assisted development:** [`CLAUDE.md`](CLAUDE.md)
- **Course requirements:** [`docs/Milestone1.md`](docs/Milestone1.md), [`Milestone2.md`](docs/Milestone2.md), [`Milestone3.md`](docs/Milestone3.md)

## Setup

Requires Python ≥ 3.11 and git. The project runs on a local machine (an NVIDIA GPU is
used for the deep-learning tasks). The full pinned environment (`environment.yml`,
including the geospatial and ML stack and GPU notes) is added in task P0-T02. Until
then, the package skeleton installs like this:

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/ckrvtz/cs482proj.git
   cd cs482proj
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install the package in editable mode with dev tools:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create your local credentials file and fill in your CDSE OAuth client
   (from the [Sentinel Hub dashboard](https://shapps.dataspace.copernicus.eu/dashboard/)).
   `.env` is gitignored; never commit it.
   ```bash
   cp .env.example .env
   ```
5. Check everything works:
   ```bash
   pytest
   ```

## Repository layout

```
config/            project.yaml (periods, grid, thresholds, seeds, paths), aois.geojson
src/desertification/   Python package; notebooks import from here
notebooks/         numbered workflow notebooks; final/ holds the graded M3 notebook
data/              raw/ and interim/ are local only (gitignored); processed/, sample/, labels/ are small and committed
reports/           figures/, results/ (metric CSVs), milestone1-3/ report drafts
scripts/           helper scripts (e.g. Markdown → PDF build)
tests/             pytest unit tests
docs/              roadmap, milestone requirements, references
```

Paths, years, AOIs and thresholds are read from `config/` via `desertification.config`;
don't hard-code them.

## Data and attribution

Raw imagery is downloaded once and cached under `data/raw/` on the local machine,
which avoids spending the limited CDSE processing-unit quota twice. Don't delete it
without checking with the team.

Contains modified Copernicus Sentinel data, processed by Sentinel Hub. All third-party
data, code and methods are cited in [`docs/REFERENCES.md`](docs/REFERENCES.md).
