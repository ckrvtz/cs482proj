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
- **ESA WorldCover** 10 m, 2020 (v100) and 2021 (v200) (VITO / ESA), read from Microsoft Planetary Computer (`esa-worldcover`, asset `map`) in M1-T07. CC BY 4.0. Citations as given on https://esa-worldcover.org/en/data-access#citation (checked 2026-11-23): Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, L., Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). *ESA WorldCover 10 m 2020 v100*. https://doi.org/10.5281/zenodo.5571936 · Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M., Fritz, S., Lesiv, M., Herold, M., Tsendbazar, N.E., Xu, P., Ramoino, F., Arino, O. (2022). *ESA WorldCover 10 m 2021 v200*. https://doi.org/10.5281/zenodo.7254221. The two maps were made with different algorithm versions, so differences between them are not all real land-cover change.
- **Copernicus DEM GLO-30** (read from Microsoft Planetary Computer `cop-dem-glo-30` in M1-T07). Free, full and open licence (Copernicus DEM licence, https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf, linked from https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM.html; checked 2026-11-23). Required notice for modified data, used on our figures: "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved".
- **Hobbins reference evapotranspiration (ETo)**, short-crop reference ET, monthly, 0.05°, global (90°N–60°S), 1980–present, distributed by the Climate Hazards Center at https://data.chc.ucsb.edu/products/Hobbins_RefET/ (readme: data derive from Mike Hobbins' fine-resolution ETos data, NOAA; "the dataset is experimental and open to change"). Used with CHIRPS v3 to compute the aridity index (P/ETo) in M1-T02. Formal citation and licence `[VERIFY]` on the NOAA PSL page the readme links to (https://www.esrl.noaa.gov/psd/eddi/globalrefet/), which was not reachable from our environment.
- **Microsoft Planetary Computer** STAC API (https://planetarycomputer.microsoft.com/api/stac/v1), used to find and sign the DEM and WorldCover COGs (M1-T07). Each dataset keeps its own licence (above).
- **Global Aridity Index and Potential Evapotranspiration Database** (Trabucco & Zomer). Not used: figshare blocks automated downloads, so the aridity index is computed from CHIRPS v3 and Hobbins ETo instead (Decision log #31). Cite only if added later for a cross-check.

## Policy and definitions

- United Nations Convention to Combat Desertification (1994), Article 1 (definitions of "desertification" and of "arid, semi-arid and dry sub-humid areas" as areas where annual precipitation divided by potential evapotranspiration is between 0.05 and 0.65). https://www.unccd.int/sites/default/files/relevant-links/2017-01/UNCCD_Convention_ENG_0.pdf
- UN Statistics Division. SDG indicator metadata 15.3.1, "Proportion of land that is degraded over total land area" (sub-indicators: trends in land cover, land productivity and carbon stocks). https://unstats.un.org/sdgs/metadata/files/Metadata-15-03-01.pdf
- Trends.Earth (Conservation International, Lund University, NASA; GEF support). https://github.com/ConservationInternational/trends.earth. Licence `[VERIFY]` (README: GPL v2.0 or later; an Apache-2.0 file is also present).

## Software and libraries

<!-- Library — version, URL, citation if the authors request one. -->

- **rasterio** 1.5.2 (BSD-3-Clause), https://rasterio.readthedocs.io; **pyproj** 3.8.0 (MIT), https://pyproj4.github.io/pyproj; **pystac-client** 0.9.1 (Apache-2.0), https://github.com/stac-utils/pystac-client; **planetary-computer** 1.0.0 (MIT), https://github.com/microsoft/planetary-computer-sdk-for-python. Used for COG reads, reprojection and STAC search in M1-T02 and M1-T07. Licences `[VERIFY]` against each repository's LICENSE file.
- **sentinelhub-py** 3.12.1 — Sinergise. Python client for Sentinel Hub services (used for CDSE OAuth config and the Catalog API). MIT License. https://github.com/sentinel-hub/sentinelhub-py · docs: https://sentinelhub-py.readthedocs.io

## Papers and methods

<!-- Author(s) (year). Title. Venue. DOI/URL. One line on what we used it for. -->

- Guo, X., Yang, G., Wang, C., Ren, Y., Han, X., Wang, N., Jia, G., & Qiao, S. (2025). Improvement effects of different shrub sand fixation plantations on vegetation and soil in the Kubuqi Desert. *Frontiers in Plant Science*, 16, 1688154. https://doi.org/10.3389/fpls.2025.1688154. Section 2.1 locates the Kubuqi Desert Closed Reserve in Dalat County, Ordos (109°00'–110°45' E, 40°00'–40°30' N), where shrub sand-fixation plantations were studied; used to place the Kubuqi AOI (Decision log #41, #42). Checked against the article page on 2026-11-21.

- Horn, B. K. P. (1981). Hill shading and the reflectance map. *Proceedings of the IEEE*, 69(1), 14–47. `[VERIFY]` DOI 10.1109/PROC.1981.11918. The 3 × 3 finite-difference slope/aspect estimator implemented in `aux_data.slope_aspect` (M1-T07).

## Reused or adapted code

<!-- Source URL — what was adapted and where (file path). -->
