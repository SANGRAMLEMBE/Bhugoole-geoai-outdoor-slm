# OpenStreetMap replica of India

The assistant asks the live Overpass API first. When the public Overpass servers are overloaded (they often are:
"all query workers are busy", or 40-80 s per query), it queries our own replica instead: a compact SQLite copy of
the OpenStreetMap features it needs (activity venues, major roads with geometry, industrial areas, green cover,
water and coastline, power lines) for all of India, with an R-tree spatial index.

* **Kept current:** on the live deployment, `refresh_osm` in `modal_app.py` downloads Geofabrik's daily India
  extract and rebuilds the replica every day (about 20 minutes), so it is at most about a day behind OSM.
* **Transparent:** answers that use it say so in their notes and sources, with the data date; `/health` shows it too.
* Source: Geofabrik India extract, <https://download.geofabrik.de/asia/india.html>
* Licence: Map data © OpenStreetMap contributors, available under the
  [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). This derived database is
  shared under the same licence.
* Build it yourself: `python scripts/build_osm_fallback.py india-latest.osm.pbf` (13 min on a workstation).

`osm_india.sqlite` is not committed to git (202 MB). The copy in this folder is only a starting point for a new
deployment until the first daily rebuild has run.
