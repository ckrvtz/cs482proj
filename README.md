# Desertification detection with Sentinel-2

CS482 data mining project: detect and predict desertification in drylands from free
Sentinel-2 L2A imagery, accessed through the Sentinel Hub APIs on the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) (CDSE).

- **Plan, tasks and study design:** [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **Rules for LLM-assisted development:** [`CLAUDE.md`](CLAUDE.md)
- **Course requirements:** [`docs/Milestone1.md`](docs/Milestone1.md), [`Milestone2.md`](docs/Milestone2.md), [`Milestone3.md`](docs/Milestone3.md)

## Setup

Tested with **Python 3.12** on Linux x86_64. PyTorch wheels for the pinned build also
exist for Windows x86_64. The project runs on a local machine; deep-learning tasks use an
NVIDIA GTX 1080 Ti but everything also runs on CPU.

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/ckrvtz/cs482proj.git
   cd cs482proj
   ```
2. Create the environment, **either** with conda:
   ```bash
   conda env create -f environment.yml
   ```
   ```bash
   conda activate desert
   ```
   **or** with a plain virtual environment (install in this order):
   ```bash
   python3.12 -m venv .venv
   ```
   ```bash
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
   ```bash
   pip install -r requirements-torch.txt
   ```
   ```bash
   pip install -r requirements.txt
   ```
   ```bash
   pip install -e ".[dev]"
   ```
3. Check the install. On the GPU machine use `--require-gpu`, which also runs a test
   matrix multiply on the card:
   ```bash
   python scripts/check_env.py --require-gpu
   ```
   ```bash
   pytest
   ```
4. Create your local credentials file and fill in your CDSE OAuth client
   (from the [Sentinel Hub dashboard](https://shapps.dataspace.copernicus.eu/dashboard/)).
   `.env` is gitignored; never commit it.
   ```bash
   cp .env.example .env
   ```

### GPU notes (GTX 1080 Ti)

- The 1080 Ti is a Pascal card (compute capability 6.1). Newer PyTorch builds have been
  dropping Pascal, so torch is pinned to **`2.14.1+cu126`** from the PyTorch CUDA 12.6
  index (`requirements-torch.txt`). That build ships `sm_60` kernels, which run on 6.1
  cards, so `torch.cuda.get_arch_list()` shows `sm_60` rather than `sm_61`.
- It needs an NVIDIA driver that supports CUDA 12.6 (`nvidia-smi` shows the driver's
  maximum CUDA version in its header). The CUDA toolkit itself comes with the pip wheels.
- Don't upgrade torch or switch to another CUDA index without re-running
  `scripts/check_env.py --require-gpu`.
- Code picks the device with `desertification.hardware.select_device()`, which falls back
  to CPU when the GPU is missing or unsupported.

### Building report PDFs

Reports are written in Markdown and converted with pandoc:
```bash
scripts/build_pdf.sh reports/milestone1/milestone1_report.md
```
This needs [pandoc](https://pandoc.org/installing.html) and a LaTeX engine (default
`xelatex`, from TeX Live on Linux or MiKTeX on Windows). On Debian/Ubuntu:
```bash
sudo apt-get install pandoc texlive-xetex texlive-latex-recommended texlive-fonts-recommended lmodern
```
Use another engine with `PDF_ENGINE=lualatex scripts/build_pdf.sh ...`. Image paths are
resolved relative to the report's folder, so `../figures/<name>.png` works.
On Windows, run the script from Git Bash or WSL.

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
