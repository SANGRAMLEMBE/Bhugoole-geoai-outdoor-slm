"""Activity knowledge base: how each activity reacts to weather, air, daylight and place.

Thresholds are deliberately simple and documented so they can be explained and
defended. Sources for the physiology-based limits are listed in the README
(heat index guidance, WHO/CPCB air-quality guidance, WHO UV index guidance).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Venue:
    """An Overpass QL filter and how well a match suits the activity (0-1)."""
    ql: str
    quality: float
    label: str


@dataclass(frozen=True)
class Activity:
    id: str
    name: str
    keywords: tuple[str, ...]            # regexes, matched with word boundaries
    venues: tuple[Venue, ...]
    default_radius_km: float
    exertion: float                       # ~MET-like factor: 0.8 (photos) .. 3 (running)
    feels_ideal: tuple[float, float]      # comfortable apparent temperature, deg C
    feels_max: float                      # above this: avoid (healthy adult)
    rain_tolerance: float                 # precipitation probability (%) that is still fine
    wind_max: float                       # km/h sustained, above this it starts to hurt
    gust_max: float
    duration_h: int                       # typical session length
    daylight: str                         # "required" | "preferred" | "night" | "any"
    usual_hours: tuple[int, int]          # used when the user gives no part of day
    free_travel_min: float                # travel time that costs nothing
    min_wind: float = 0.0                 # e.g. kite flying needs wind
    uv_sensitivity: float = 1.0
    lit_ok: bool = False                  # floodlit venues extend the day
    prepare: tuple[str, ...] = ()
    watch: tuple[str, ...] = ()
    needs_venue: bool = True
    tags: tuple[str, ...] = field(default=())


PARKS = (
    Venue('nwr["leisure"="park"]["name"]', 0.8, "park"),
    Venue('nwr["leisure"="garden"]["name"]', 0.7, "garden"),
)

ACTIVITIES: dict[str, Activity] = {}


def _add(a: Activity):
    ACTIVITIES[a.id] = a


_add(Activity(
    id="running", name="Running / jogging",
    keywords=(r"run(?:ning)?", r"jog(?:ging)?", r"5\s?k", r"10\s?k", r"marathon", r"sprint(?:s|ing)?"),
    venues=(
        Venue('nwr["leisure"="track"]', 1.0, "running track"),
        *PARKS,
        Venue('nwr["leisure"="nature_reserve"]["name"]', 0.75, "nature reserve"),
        Venue('nwr["natural"="beach"]["name"]', 0.7, "beach"),
    ),
    default_radius_km=5, exertion=3.0, feels_ideal=(12, 27), feels_max=38,
    rain_tolerance=40, wind_max=35, gust_max=55, duration_h=1, daylight="preferred",
    usual_hours=(5, 21), free_travel_min=12, uv_sensitivity=1.2, lit_ok=True,
    prepare=("Water (about 500 ml per hour of running)", "Light, breathable clothing", "Phone and ID"),
    watch=("Uneven paths and traffic crossings", "Signs of heat stress: dizziness, chills, nausea"),
))
_add(Activity(
    id="cycling", name="Cycling (leisure or commute)",
    keywords=(r"cycl(?:e|ing|ist)", r"bik(?:e|ing)", r"bicycle", r"ride to work", r"pedal"),
    venues=(
        Venue('way["highway"="cycleway"]["name"]', 1.0, "cycle track"),
        Venue('nwr["leisure"="park"]["name"]', 0.6, "park"),
        Venue('nwr["natural"="beach"]["name"]', 0.5, "beach promenade"),
    ),
    default_radius_km=8, exertion=2.5, feels_ideal=(12, 28), feels_max=38,
    rain_tolerance=30, wind_max=28, gust_max=45, duration_h=1, daylight="preferred",
    usual_hours=(5, 21), free_travel_min=10, uv_sensitivity=1.1,
    prepare=("Helmet and front/rear lights", "Water", "Puncture kit or pump"),
    watch=("Wet roads and potholes", "Heavy vehicles at junctions", "Exhaust exposure on arterial roads"),
))
_add(Activity(
    id="golf", name="Golf",
    keywords=(r"golf(?:ing)?", r"tee time", r"driving range"),
    venues=(Venue('nwr["leisure"="golf_course"]', 1.0, "golf course"),
            Venue('nwr["golf"="driving_range"]', 0.7, "driving range")),
    default_radius_km=30, exertion=1.5, feels_ideal=(15, 29), feels_max=40,
    rain_tolerance=30, wind_max=28, gust_max=45, duration_h=4, daylight="required",
    usual_hours=(6, 18), free_travel_min=35, uv_sensitivity=1.3,
    prepare=("Sunscreen SPF 30+, cap and sunglasses", "At least 2 litres of water for 18 holes", "Book a tee time"),
    watch=("Lightning: leave the course at the first thunder", "Crosswinds affecting ball flight"),
))
_add(Activity(
    id="cricket", name="Cricket (recreational match or practice)",
    keywords=(r"cricket", r"nets? practice", r"box cricket", r"gully cricket"),
    venues=(
        Venue('nwr["leisure"~"^(pitch|stadium|sports_centre)$"]["sport"~"cricket"]', 1.0, "cricket ground"),
        Venue('nwr["leisure"~"^(park|pitch)$"]["name"~"[Mm]aidan|[Gg]round|[Gg]ymkhana"]', 0.75, "maidan / ground"),
        Venue('nwr["leisure"="pitch"]["sport"~"multi"]', 0.6, "multi-sport pitch"),
    ),
    default_radius_km=8, exertion=2.0, feels_ideal=(15, 30), feels_max=39,
    rain_tolerance=25, wind_max=35, gust_max=55, duration_h=3, daylight="required",
    usual_hours=(6, 18), free_travel_min=15, uv_sensitivity=1.3, lit_ok=True,
    prepare=("Water and electrolytes, a drinks break every 30-40 min", "Caps and sunscreen", "Helmet and pads for batting"),
    watch=("Wet outfield after rain (slippery, ball skids)", "Low sun glare for fielders late in the day"),
))
_add(Activity(
    id="football", name="Outdoor football / five-a-side",
    keywords=(r"football", r"soccer", r"five[- ]a[- ]side", r"5[- ]a[- ]side", r"futsal", r"turf"),
    venues=(
        Venue('nwr["leisure"~"^(pitch|sports_centre|stadium)$"]["sport"~"soccer|football"]', 1.0, "football pitch"),
        Venue('nwr["leisure"="pitch"]["sport"~"multi"]', 0.6, "multi-sport pitch"),
        Venue('nwr["leisure"="park"]["name"~"[Mm]aidan|[Gg]round"]', 0.5, "maidan / ground"),
    ),
    default_radius_km=8, exertion=3.0, feels_ideal=(12, 27), feels_max=37,
    rain_tolerance=35, wind_max=35, gust_max=55, duration_h=1, daylight="preferred",
    usual_hours=(6, 22), free_travel_min=15, uv_sensitivity=1.2, lit_ok=True,
    prepare=("Water and electrolytes", "Studs or turf shoes to suit the surface", "Shin guards"),
    watch=("Slippery turf after rain", "Heat exhaustion in high-intensity play"),
))
_add(Activity(
    id="tennis", name="Outdoor tennis",
    keywords=(r"tennis", r"pickleball"),
    venues=(
        Venue('nwr["leisure"~"^(pitch|sports_centre)$"]["sport"~"tennis"]', 1.0, "tennis court"),
        Venue('nwr["club"="sport"]["sport"~"tennis"]', 0.9, "tennis club"),
    ),
    default_radius_km=8, exertion=2.5, feels_ideal=(14, 28), feels_max=37,
    rain_tolerance=20, wind_max=22, gust_max=38, duration_h=1, daylight="preferred",
    usual_hours=(6, 22), free_travel_min=15, uv_sensitivity=1.2, lit_ok=True,
    prepare=("Water", "Cap or visor and sunglasses", "Check if the court needs booking"),
    watch=("Wet courts are slippery after rain", "Gusty wind makes ball control hard"),
))
_add(Activity(
    id="hiking", name="Hiking / nature walk",
    keywords=(r"hik(?:e|es|ing)", r"trek(?:s|king)?", r"nature walk", r"trail(?:s)?", r"fort walk"),
    venues=(
        Venue('nwr["historic"~"^(fort|castle)$"]["name"]', 0.9, "fort trail"),
        Venue('node["natural"="peak"]["name"]', 0.9, "peak"),
        Venue('nwr["leisure"="nature_reserve"]["name"]', 0.9, "nature reserve"),
        Venue('nwr["boundary"~"^(national_park|protected_area)$"]["name"]', 0.85, "protected area"),
        Venue('node["tourism"="viewpoint"]["name"]', 0.7, "viewpoint"),
    ),
    default_radius_km=35, exertion=3.0, feels_ideal=(12, 27), feels_max=36,
    rain_tolerance=30, wind_max=35, gust_max=55, duration_h=4, daylight="required",
    usual_hours=(5, 17), free_travel_min=45, uv_sensitivity=1.2,
    prepare=("2-3 litres of water and snacks", "Shoes with good grip", "Offline map and charged phone",
             "Tell someone your route"),
    watch=("Slippery rock and mud after rain", "Start down well before sunset", "Fog lowering visibility"),
))
_add(Activity(
    id="walking", name="Walking",
    keywords=(r"walk(?:s|ing)?", r"stroll", r"sair", r"evening walk", r"morning walk"),
    venues=(
        *PARKS,
        Venue('nwr["leisure"="nature_reserve"]["name"]', 0.8, "nature reserve"),
        Venue('nwr["natural"="beach"]["name"]', 0.75, "beach"),
        Venue('nwr["leisure"="track"]', 0.7, "walking track"),
    ),
    default_radius_km=4, exertion=1.3, feels_ideal=(14, 29), feels_max=39,
    rain_tolerance=40, wind_max=40, gust_max=60, duration_h=1, daylight="preferred",
    usual_hours=(5, 21), free_travel_min=10, uv_sensitivity=1.0, lit_ok=True,
    prepare=("Water", "Comfortable shoes"),
    watch=("Uneven footpaths", "Traffic at crossings"),
))
_add(Activity(
    id="picnic", name="Picnic / park outing",
    keywords=(r"picnic", r"park outing", r"outing", r"day out", r"family outing"),
    venues=(
        Venue('nwr["tourism"="picnic_site"]', 1.0, "picnic site"),
        *PARKS,
        Venue('nwr["natural"="beach"]["name"]', 0.7, "beach"),
        Venue('nwr["natural"="water"]["name"]["water"~"lake|reservoir"]', 0.6, "lakeside"),
    ),
    default_radius_km=10, exertion=1.0, feels_ideal=(16, 30), feels_max=39,
    rain_tolerance=20, wind_max=30, gust_max=45, duration_h=3, daylight="required",
    usual_hours=(7, 18), free_travel_min=25, uv_sensitivity=1.1,
    prepare=("Mat, shade (umbrella) and water", "Food in a cool box; carry waste back", "Mosquito repellent near water"),
    watch=("Ants and stray animals near food", "Afternoon heat in open lawns"),
))
_add(Activity(
    id="event", name="Outdoor event / gathering",
    keywords=(r"wedding", r"party", r"birthday", r"gathering", r"\bfair\b", r"function", r"reception",
              r"get[- ]together", r"sangeet", r"mehendi", r"mehndi", r"haldi", r"celebration", r"\bevent\b"),
    venues=(
        Venue('nwr["amenity"="events_venue"]', 1.0, "events venue"),
        Venue('nwr["leisure"="garden"]["name"]', 0.7, "garden"),
        Venue('nwr["leisure"="park"]["name"]', 0.6, "park"),
    ),
    default_radius_km=10, exertion=1.0, feels_ideal=(16, 30), feels_max=38,
    rain_tolerance=15, wind_max=28, gust_max=40, duration_h=3, daylight="any",
    usual_hours=(8, 22), free_travel_min=25, uv_sensitivity=1.0,
    prepare=("Shade or a canopy, plus a rain backup (tent or indoor option)", "Drinking water stations",
             "Fans or coolers if it is warm", "Secure decorations and tents against gusts"),
    watch=("Rain probability in the hours before the event", "Heat stress among elderly guests and children"),
))
_add(Activity(
    id="photography", name="Outdoor photography",
    keywords=(r"photo(?:s|graphy|walk)?", r"pictures?", r"pics", r"sunsets?", r"sunrises?", r"golden hour",
              r"shoot", r"timelapse"),
    venues=(
        Venue('node["tourism"="viewpoint"]', 1.0, "viewpoint"),
        Venue('nwr["natural"="beach"]["name"]', 0.9, "beach"),
        Venue('nwr["man_made"~"^(pier|lighthouse)$"]', 0.9, "pier / lighthouse"),
        Venue('nwr["historic"~"^(fort|monument|castle)$"]["name"]', 0.8, "heritage site"),
        Venue('nwr["leisure"="park"]["name"]', 0.5, "park"),
    ),
    default_radius_km=10, exertion=0.8, feels_ideal=(12, 32), feels_max=40,
    rain_tolerance=30, wind_max=35, gust_max=50, duration_h=1, daylight="any",
    usual_hours=(5, 20), free_travel_min=20, uv_sensitivity=0.8,
    prepare=("Charged batteries and memory card", "Lens cloth (sea spray, humidity)", "Rain cover for the camera"),
    watch=("Haze from high PM levels flattens contrast", "Tide and footing on rocks near the sea"),
))
# ---- extra activities beyond the required ten ----
_add(Activity(
    id="yoga", name="Outdoor yoga / meditation",
    keywords=(r"yoga", r"meditat(?:e|ion)", r"pranayam", r"surya namaskar", r"tai chi"),
    venues=(*PARKS, Venue('nwr["natural"="beach"]["name"]', 0.8, "beach")),
    default_radius_km=4, exertion=1.2, feels_ideal=(15, 28), feels_max=36,
    rain_tolerance=25, wind_max=25, gust_max=40, duration_h=1, daylight="preferred",
    usual_hours=(5, 20), free_travel_min=10, uv_sensitivity=1.0,
    prepare=("Mat and water", "A towel"),
    watch=("Deep breathing amplifies the effect of polluted air", "Ants and wet grass"),
))
_add(Activity(
    id="birdwatching", name="Birdwatching",
    keywords=(r"bird(?:ing|watching| watching)?", r"birds"),
    venues=(
        Venue('nwr["natural"="wetland"]["name"]', 1.0, "wetland"),
        Venue('nwr["leisure"="nature_reserve"]["name"]', 0.95, "nature reserve"),
        Venue('nwr["boundary"~"^(national_park|protected_area)$"]["name"]', 0.9, "protected area"),
        Venue('nwr["natural"="water"]["name"]["water"~"lake|reservoir|lagoon"]', 0.8, "lake"),
    ),
    default_radius_km=25, exertion=1.0, feels_ideal=(14, 30), feels_max=38,
    rain_tolerance=30, wind_max=20, gust_max=35, duration_h=2, daylight="required",
    usual_hours=(6, 18), free_travel_min=35, uv_sensitivity=1.0,
    prepare=("Binoculars and a field guide app", "Muted clothing", "Water and a hat"),
    watch=("Birds are most active in the first 2-3 hours after sunrise", "Mosquitoes near wetlands"),
))
_add(Activity(
    id="kite", name="Kite flying",
    keywords=(r"kites?", r"patang", r"kite flying"),
    venues=(
        Venue('nwr["natural"="beach"]["name"]', 1.0, "beach"),
        Venue('nwr["leisure"~"^(park|pitch)$"]["name"~"[Mm]aidan|[Gg]round"]', 0.9, "open ground"),
        Venue('nwr["leisure"="park"]["name"]', 0.6, "park"),
    ),
    default_radius_km=10, exertion=1.0, feels_ideal=(14, 32), feels_max=39,
    rain_tolerance=20, wind_max=32, gust_max=45, duration_h=2, daylight="required",
    usual_hours=(7, 18), free_travel_min=20, min_wind=9, uv_sensitivity=1.0,
    prepare=("Cotton string (glass-coated manja is banned and dangerous)", "Cap, sunglasses and water"),
    watch=("Stay well away from power lines", "Birds and passers-by in the line"),
    tags=("power_lines",),
))
_add(Activity(
    id="stargazing", name="Stargazing",
    keywords=(r"star ?gaz(?:e|ing)", r"stars", r"milky way", r"astronomy", r"meteor"),
    venues=(
        Venue('node["natural"="peak"]["name"]', 0.9, "peak"),
        Venue('node["tourism"="viewpoint"]', 0.8, "viewpoint"),
        Venue('nwr["natural"="beach"]["name"]', 0.7, "beach"),
    ),
    default_radius_km=40, exertion=0.8, feels_ideal=(10, 30), feels_max=40,
    rain_tolerance=15, wind_max=30, gust_max=45, duration_h=2, daylight="night",
    usual_hours=(19, 23), free_travel_min=45, uv_sensitivity=0.0,
    prepare=("Red torch, warm layer and a mat", "A sky-map app"),
    watch=("Moonlight washes out faint stars", "City light pollution: darker skies are farther from towns"),
))

ACTIVITY_IDS = list(ACTIVITIES)
REQUIRED_TEN = ["running", "cycling", "golf", "cricket", "football", "tennis", "hiking", "picnic", "event",
                "photography"]
