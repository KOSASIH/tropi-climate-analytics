# Data Sources Reference

## NASA Satellite Products

| Product | Dataset | Resolution | Cadence | Agent |
|---------|---------|-----------|---------|-------|
| Landsat 8 OLI | LC08_L2SP | 30m | 16-day | GEOSPATIAL |
| Landsat 9 OLI-2 | LC09_L2SP | 30m | 16-day | GEOSPATIAL |
| MODIS Terra NDVI | MOD13A2 | 1km | 16-day | ATMOSPHERE |
| MODIS Aqua SST | MYD11A2 | 1km | 8-day | ATMOSPHERE |
| MODIS Fire | MOD14A1 | 1km | Daily | ATMOSPHERE |
| MODIS Aerosol | MOD04_L2 | 10km | Daily | ATMOSPHERE |
| SMAP Soil Moisture | SPL3SMP | 9km | Daily | HYDROLOGIS |
| GPM IMERG | GPM_3IMERGHH | 0.1° | 30-min | HYDROLOGIS |
| CALIPSO Aerosol | CAL_LID_L2 | 5km | ~16-day | ATMOSPHERE |

## Indonesian Ground Networks

| Network | Operator | Stations | Variables | Agent |
|---------|---------|---------|----------|-------|
| AWS/AAWS | BMKG | ~150 | T, RH, Wind, Precip, P | DATA-FLOW |
| Rain Gauge | BMKG | ~550 | Daily rainfall | HYDROLOGIS |
| Air Quality | KLHK | ~100 | PM2.5, PM10, CO, O3, SO2 | ATMOSPHERE |
| Forest Cover | KLHK | - | Annual land cover map | GEOSPATIAL |
| Remote Sensing | LAPAN | - | SPOT-7, SAR | GEOSPATIAL |

## NASA API Access

Authentication: NASA Earthdata Login (register at urs.earthdata.nasa.gov)
CMR Search: https://cmr.earthdata.nasa.gov/search
Data Download: HTTPS via Earthdata with OAuth2 bearer token
Rate Limits: 2000 granule search results per query; no download rate limit

## Compliance Notes

- NASA open data policy: CC0 / public domain for all Level 2+ products
- BMKG data: Open Government Data Indonesia
- KLHK data: Ministry open data portal
- Attribution required in publications and stakeholder reports
