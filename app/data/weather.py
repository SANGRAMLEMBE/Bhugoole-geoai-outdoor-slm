"""Open-Meteo hourly forecast and air-quality forecast, many locations per request."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from .. import config
from .http import HTTP

log = logging.getLogger("weather")

WX_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation_probability",
           "precipitation", "weather_code", "cloud_cover", "visibility", "wind_speed_10m", "wind_gusts_10m",
           "wind_direction_10m", "uv_index", "is_day"]
AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "us_aqi"]

WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
       82: "violent showers", 85: "snow showers", 86: "snow showers", 95: "thunderstorm",
       96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}


@dataclass
class Series:
    lat: float
    lon: float
    times: list[datetime]
    values: dict[str, list]
    elevation: float | None = None
    daily: dict[str, list] = field(default_factory=dict)
    _index: dict[datetime, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._index = {t: i for i, t in enumerate(self.times)}

    def at(self, t: datetime) -> dict | None:
        i = self._index.get(t.replace(minute=0, second=0, microsecond=0, tzinfo=None))
        if i is None:
            return None
        return {k: (v[i] if i < len(v) else None) for k, v in self.values.items()}

    def sun_events(self) -> dict:
        """date -> (sunrise, sunset) as naive local datetimes."""
        out = {}
        for d, sr, ss in zip(self.daily.get("time", []), self.daily.get("sunrise", []), self.daily.get("sunset", [])):
            try:
                out[d] = (datetime.fromisoformat(sr), datetime.fromisoformat(ss))
            except (TypeError, ValueError):
                continue
        return out


def _parse(payload, varnames) -> list[Series]:
    items = payload if isinstance(payload, list) else [payload]
    out = []
    for it in items:
        h = it.get("hourly", {})
        times = [datetime.fromisoformat(t) for t in h.get("time", [])]
        out.append(Series(it.get("latitude"), it.get("longitude"), times,
                          {k: h.get(k, [None] * len(times)) for k in varnames},
                          it.get("elevation"), it.get("daily", {})))
    return out


def _coords(points: list[tuple[float, float]]) -> dict:
    return {"latitude": ",".join(f"{p[0]:.4f}" for p in points),
            "longitude": ",".join(f"{p[1]:.4f}" for p in points)}


async def forecast(points: list[tuple[float, float]], days: int) -> list[Series]:
    params = {**_coords(points), "hourly": ",".join(WX_VARS), "daily": "sunrise,sunset",
              "timezone": config.TZ.key, "forecast_days": max(1, min(days, 16)), "wind_speed_unit": "kmh"}
    data = await HTTP.get_json(config.OPEN_METEO_FORECAST_URL, params, ttl=config.TTL_WEATHER,
                               source="Open-Meteo Forecast API")
    return _parse(data, WX_VARS)


def _trailing_mean(vals: list, window: int) -> list:
    """Trailing mean over `window` hours, ignoring gaps (CPCB averages O3 over 8 h)."""
    out = []
    for i in range(len(vals)):
        win = [v for v in vals[max(0, i - window + 1): i + 1] if v is not None]
        out.append(sum(win) / len(win) if win and vals[i] is not None else None)
    return out


async def air_quality(points: list[tuple[float, float]], days: int) -> list[Series] | None:
    # past_days=1 gives the 8-hour O3 average history for the first forecast hours
    params = {**_coords(points), "hourly": ",".join(AQ_VARS), "timezone": config.TZ.key,
              "forecast_days": max(1, min(days, 7)), "past_days": 1}
    try:
        data = await HTTP.get_json(config.OPEN_METEO_AQ_URL, params, ttl=config.TTL_WEATHER,
                                   source="Open-Meteo Air Quality API (CAMS)")
    except Exception as exc:  # noqa: BLE001 - AQ is important but not fatal
        log.warning("air quality unavailable: %s", exc)
        return None
    series = _parse(data, AQ_VARS)
    for s in series:
        s.values["ozone"] = _trailing_mean(s.values.get("ozone") or [], 8)
    return series
