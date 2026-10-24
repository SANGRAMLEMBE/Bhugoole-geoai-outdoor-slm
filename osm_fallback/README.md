# Offline OpenStreetMap layer

`osm_india.sqlite` is a compact copy of the OpenStreetMap features the assistant needs (activity venues, major
roads with geometry, industrial areas, green cover, water and coastline, power lines) for all of India. It is
used **only when the live Overpass API fails** (configurable with `OSM_SOURCE_ORDER`), and answers that use it
say so in their notes and sources.

* Source: Geofabrik India extract, <https://download.geofabrik.de/asia/india.html>
* Licence: Map data © OpenStreetMap contributors, available under the
  [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). This derived database is
  shared under the same licence.
* Rebuild (~15-30 min): see `scripts/build_osm_fallback.py`. The data date is stored in the `meta` table.

The `.sqlite` file is not committed to git (it is large); the deploy script uploads it to the Space.
