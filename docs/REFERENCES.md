# References

Every dataset, library, paper and piece of reused code used in this project is cited here
(see CLAUDE.md, "Academic integrity and citations"). Add an entry when something is first
used, and only after checking it yourself: authors, year, venue/version, and a link that opens.
Mark anything not yet checked with `[VERIFY]`.

Data attribution for figures and reports:
*"Contains modified Copernicus Sentinel data [years], processed by Sentinel Hub."*

## Data

<!-- Dataset — provider, version, licence, access URL, citation. Added by the task that first uses it. -->

All entries below were checked against the linked official pages on 2026-11-20 unless marked `[VERIFY]`.

- **Copernicus Sentinel-2 Level-2A**, accessed via Sentinel Hub on the Copernicus Data Space Ecosystem. Free, full and open; attribute "Copernicus Sentinel data [Year]", or for processed data "Contains modified Copernicus Sentinel data [Year]". Legal notice: https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice · CDSE terms: https://dataspace.copernicus.eu/terms-and-conditions · API docs: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub.html
- **Microsoft Planetary Computer** STAC API, https://planetarycomputer.microsoft.com/api/stac/v1/. Service under "Supplemental Terms of Use for Microsoft Planetary Computer Previews" (Jan 2023): https://planetarycomputer.microsoft.com/terms-of-use. Each dataset keeps its own licence (see its STAC collection metadata).
- **CHIRPS v3** (Climate Hazards Center, UC Santa Barbara). CC BY 4.0. Data: Climate Hazards Center Infrared Precipitation with Stations version 3 (CHIRPS3) Data Repository, https://doi.org/10.15780/G2JQ0P (2025), with access date. Paper: Funk, C., Peterson, P., Harrison, L., et al. (2026). The Climate Hazards Center Infrared Precipitation with Stations, Version 3. *Scientific Data* 13, 718. https://doi.org/10.1038/s41597-026-07096-4 · https://www.chc.ucsb.edu/data/chirps3
- **ESA WorldCover** 10 m, 2020 and 2021 (VITO / ESA). CC BY 4.0 (https://registry.opendata.aws/esa-worldcover-vito/). Recommended citation string `[VERIFY]` on https://esa-worldcover.org/en.
- **Copernicus DEM GLO-30** (via Planetary Computer `cop-dem-glo-30`). Licence and attribution wording `[VERIFY]`; licence links are on https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM.html.
- **Hobbins reference evapotranspiration (ETo)**, short-crop reference ET, monthly, 0.05°, global (90°N–60°S), 1980–present, distributed by the Climate Hazards Center at https://data.chc.ucsb.edu/products/Hobbins_RefET/ (readme: data derive from Mike Hobbins' fine-resolution ETos data, NOAA; "the dataset is experimental and open to change"). Used with CHIRPS v3 to compute the aridity index (P/ETo) in M1-T02. Formal citation and licence `[VERIFY]` on the NOAA PSL page the readme links to (https://www.esrl.noaa.gov/psd/eddi/globalrefet/), which was not reachable from our environment.
- **Global Aridity Index and Potential Evapotranspiration Database** (Trabucco & Zomer). Not used: figshare blocks automated downloads, so the aridity index is computed from CHIRPS v3 and Hobbins ETo instead (Decision log #31). Cite only if added later for a cross-check.

## Policy and definitions

- United Nations Convention to Combat Desertification (1994), Article 1 (definitions of "desertification" and of "arid, semi-arid and dry sub-humid areas" as areas where annual precipitation divided by potential evapotranspiration is between 0.05 and 0.65). https://www.unccd.int/sites/default/files/relevant-links/2017-01/UNCCD_Convention_ENG_0.pdf
- UN Statistics Division. SDG indicator metadata 15.3.1, "Proportion of land that is degraded over total land area" (sub-indicators: trends in land cover, land productivity and carbon stocks). https://unstats.un.org/sdgs/metadata/files/Metadata-15-03-01.pdf
- Trends.Earth (Conservation International, Lund University, NASA; GEF support). https://github.com/ConservationInternational/trends.earth. Licence `[VERIFY]` (README: GPL v2.0 or later; an Apache-2.0 file is also present).

## Software and libraries

<!-- Library — version, URL, citation if the authors request one. -->

- **sentinelhub-py** 3.12.1 — Sinergise. Python client for Sentinel Hub services (used for CDSE OAuth config and the Catalog API). MIT License. https://github.com/sentinel-hub/sentinelhub-py · docs: https://sentinelhub-py.readthedocs.io

## Papers and methods

<!-- Author(s) (year). Title. Venue. DOI/URL. One line on what we used it for. -->

- `[VERIFY]` **Kubuqi re-greening area** (109°00'–110°45' E, 40°00'–40°30' N), from a paper found by the project owner; used to place the Kubuqi AOI (Decision log #41). Full citation (authors, year, title, venue, DOI) to be added.

## Reused or adapted code

<!-- Source URL — what was adapted and where (file path). -->
