"""Per-city configuration: OSM extent, units, currency and road-class defaults.

Everything city-specific lives here so the rest of the pipeline is city-agnostic.
Free-flow speeds are only a fallback for edges without an OSM `maxspeed` tag; the
time-varying congestion on top of them belongs to the traffic stage (W_t in the
formulation), not to the static graph G.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class TrafficTargets:
    """Calibration targets and stochastic parameters for the traffic stage.

    `tti_*` is the hourly target *travel-time index* (congested / free-flow travel
    time, flow-weighted over the network) for hours 0..23. The background demand in
    each hour is scaled until the assigned network reaches that index.
    """

    tti_weekday: tuple[float, ...]
    tti_weekend: tuple[float, ...]
    am_peak_hour: float           # centre of the home->activity commute pattern
    pm_peak_hour: float           # centre of the activity->home pattern
    gravity_km: float             # decay length of the gravity OD model
    rain_prob_by_month: tuple[float, ...]   # P(rainy day), Jan..Dec
    incidents_per_100km_day: float          # on trunk/primary/motorway roads
    external_share: float         # share of trips with an end outside the city
    through_share: float          # share of external trips that cross the city
    source: str
    gateway_max_dist_m: float = 1500.0   # major-road nodes this close to the boundary are gateways


@dataclass(frozen=True)
class ExternalDest:
    """An out-of-city destination reached through a gateway (intercity ride).

    `gateway_lonlat` is snapped to the nearest detected gateway. Outside the map the
    trip continues `outside_km` in `outside_min` minutes; the cab then returns empty
    through the same gateway (the return takes as long again).
    """
    name: str
    gateway_lonlat: tuple[float, float] | None   # None: use the gateway of rank `gateway_rank`
    outside_km: float
    outside_min: float
    share: float                  # share of intercity requests going here
    gateway_rank: int = 0         # 0 = the gateway with the most road capacity


@dataclass(frozen=True)
class RideMarket:
    """The ride company: demand, fleet, prices, behaviour. ASSUMPTION throughout unless
    a source is given; every value is a knob of the simulated world."""

    requests_per_day: int                  # weekday; weekend uses weekend_factor
    weekend_factor: float
    demand_weekday: tuple[float, ...]      # relative requests per hour 0..23
    demand_weekend: tuple[float, ...]
    ride_gravity_km: float                 # OD decay length for rides
    intercity_share: float
    external: tuple[ExternalDest, ...]
    n_drivers: int
    shift_starts: tuple[tuple[float, float], ...]   # (start hour, weight)
    shift_median_h: float
    n_customers: int
    # tariff (in city currency): fare = max(min_fare, base + per_km*km + per_min*min)
    base_fare: float
    per_km: float
    per_min: float
    min_fare: float
    booking_fee: float                     # charged to customer, kept by platform
    commission: float                      # platform share of the fare
    cancel_fee: float                      # after acceptance + grace, paid to driver
    intercity_per_km: float                # outside-the-city distance rate
    tip_prob: float
    tip_frac_mean: float
    cost_per_km: float                     # driver operating cost kappa (fuel, wear)
    max_pickup_min: float                  # platform matching radius (ETA)


@dataclass(frozen=True)
class City:
    key: str
    name: str
    osm_query: str            # Nominatim query, or an OSM id like "R10351626"
    utm_crs: str              # metric CRS for lengths / areas
    currency: str
    timezone: str
    # Fallback free-flow speed (km/h) per OSM `highway` class. ASSUMPTION: values
    # are typical urban free-flow speeds, not legal limits; OSM tags override them.
    hwy_speeds_kph: dict[str, float] = field(default_factory=dict)
    # H3 resolution of the zone partition Z: res 8 hexes are ~0.74 km^2, res 7 ~5.2 km^2.
    # Chosen per city so zones are neighbourhood-sized for the city's extent.
    h3_res: int = 8
    # Draw the admin boundary on maps. Off where it is mostly open water (SF county
    # extends to the Farallon Islands) and would only add straight dashed lines.
    show_boundary: bool = True
    traffic: TrafficTargets | None = None
    market: RideMarket | None = None
    # Extent as a circle (lat, lon, radius km) where OSM has no city-sized admin polygon.
    circle: tuple[float, float, float] | None = None
    country: str = ""             # "IN" or "US"
    tier: str = ""                # "core", "metro", "medium", "small"
    population: int = 0           # approximate, for scaling demand

    @property
    def data_dir(self) -> Path:
        return DATA / self.key

    @property
    def out_dir(self) -> Path:
        return OUTPUTS / self.key


_COMMON = {
    "living_street": 10, "service": 12, "unclassified": 20, "road": 20,
}

CITIES: dict[str, City] = {
    "pune": City(
        key="pune",
        name="Pune",
        # OSM has no Pune Municipal Corporation polygon, and "Pune" geocodes to the
        # 15,600 km^2 district. Pune City Subdistrict (312 km^2) is the closest
        # admin boundary to the pre-2021 PMC area (~331 km^2). It excludes
        # Pimpri-Chinchwad and the Hinjewadi IT park.
        osm_query="R10351626",
        traffic=TrafficTargets(
            # TomTom Traffic Index 2025, Pune: 33:20 per 10 km on average (congestion
            # level 71%, i.e. TTI 1.71), 38:13 in the morning peak (TTI 1.96) and
            # 41:40 in the evening peak (TTI 2.14). ASSUMPTION: the hourly shape
            # around those anchors, and the weekend profile, are ours.
            tti_weekday=(1.05, 1.03, 1.02, 1.02, 1.03, 1.08, 1.20, 1.45, 1.75, 1.96, 1.90, 1.78,
                         1.70, 1.68, 1.70, 1.78, 1.90, 2.05, 2.14, 2.08, 1.85, 1.55, 1.30, 1.12),
            tti_weekend=(1.06, 1.04, 1.02, 1.02, 1.02, 1.04, 1.08, 1.15, 1.28, 1.42, 1.52, 1.58,
                         1.60, 1.58, 1.56, 1.60, 1.68, 1.78, 1.86, 1.88, 1.75, 1.50, 1.28, 1.12),
            am_peak_hour=9.5, pm_peak_hour=18.5,
            gravity_km=6.0,
            # ASSUMPTION: monsoon June-September; roughly the share of days with rain.
            rain_prob_by_month=(0.02, 0.02, 0.03, 0.05, 0.12, 0.55, 0.80, 0.75, 0.55, 0.20, 0.08, 0.03),
            incidents_per_100km_day=2.5,
            # ASSUMPTION: Pimpri-Chinchwad, Hinjewadi and the highway corridors
            # (Mumbai, Satara, Solapur, Nashik) lie outside the subdistrict.
            external_share=0.25, through_share=0.15,
            source="TomTom Traffic Index 2025 (tomtom.com/traffic-index/city/pune)",
        ),
        market=RideMarket(
            # ASSUMPTION: one mid-sized platform's share of Pune ride-hailing.
            requests_per_day=10000, weekend_factor=0.9,
            # ASSUMPTION: office-commute peaks later than in the US (IT parks start
            # ~10h), plus a strong evening peak.
            demand_weekday=(1.5, 0.8, 0.5, 0.4, 0.5, 1.2, 2.5, 4.0, 5.8, 6.8, 6.0, 5.0,
                            4.6, 4.4, 4.4, 4.8, 5.4, 6.4, 7.4, 7.6, 6.4, 4.8, 3.2, 2.3),
            demand_weekend=(2.4, 1.4, 0.8, 0.5, 0.5, 0.9, 1.6, 2.4, 3.4, 4.4, 5.2, 5.6,
                            5.8, 5.8, 5.6, 5.6, 6.0, 6.6, 7.2, 7.4, 6.8, 5.6, 4.2, 3.2),
            ride_gravity_km=5.0,
            intercity_share=0.05,
            external=(
                ExternalDest("Pune Airport (Lohegaon)", (73.9218, 18.5931), 4.0, 12.0, 0.40),
                ExternalDest("Hinjewadi / Wakad", (73.7665, 18.5673), 9.0, 30.0, 0.25),
                ExternalDest("Pimpri-Chinchwad", (73.8391, 18.5669), 8.0, 28.0, 0.20),
                ExternalDest("Mumbai (expressway)", (73.7688, 18.5180), 140.0, 180.0, 0.15),
            ),
            # Sized so peak supply roughly covers ~760 requests/h x ~33 min per trip.
            n_drivers=1400,
            # ASSUMPTION: Indian ride-hail drivers work long shifts (~9 h median).
            shift_starts=((6.5, 0.30), (10.0, 0.20), (15.0, 0.35), (20.0, 0.15)),
            shift_median_h=9.0,
            n_customers=60000,
            # ASSUMPTION: an economy car tier, roughly in line with 2025 Pune fares.
            base_fare=40.0, per_km=12.0, per_min=1.5, min_fare=80.0, booking_fee=10.0,
            # one-way outstation rate prices in the empty return (140 km Mumbai run)
            commission=0.22, cancel_fee=50.0, intercity_per_km=18.0,
            tip_prob=0.04, tip_frac_mean=0.08,
            cost_per_km=4.5,               # CNG hatchback: fuel ~3 + wear ~1.5 INR/km
            max_pickup_min=15.0,
        ),
        utm_crs="EPSG:32643",
        currency="INR",
        timezone="Asia/Kolkata",
        hwy_speeds_kph={
            **_COMMON,
            "motorway": 70, "motorway_link": 40,
            "trunk": 45, "trunk_link": 30,
            "primary": 35, "primary_link": 25,
            "secondary": 30, "secondary_link": 22,
            "tertiary": 25, "tertiary_link": 20,
            "residential": 18,
        },
        h3_res=7,
    ),
    "sf": City(
        key="sf",
        name="San Francisco",
        osm_query="San Francisco, California, USA",
        utm_crs="EPSG:32610",
        currency="USD",
        timezone="America/Los_Angeles",
        hwy_speeds_kph={
            **_COMMON,
            "motorway": 90, "motorway_link": 50,
            "trunk": 55, "trunk_link": 40,
            "primary": 45, "primary_link": 35,
            "secondary": 40, "secondary_link": 30,
            "tertiary": 35, "tertiary_link": 30,
            "residential": 30,
        },
        h3_res=8,
        show_boundary=False,
        traffic=TrafficTargets(
            # TomTom Traffic Index 2025, San Francisco: congestion level 49.7% (TTI
            # ~1.50), 29:42 per 10 km. TomTom gives no peak split, so ASSUMPTION:
            # AM peak 8-9h at TTI 1.80, a longer, higher PM peak around 17h at 1.95
            # (TomTom notes evening congestion "builds earlier and lingers longer").
            tti_weekday=(1.03, 1.02, 1.02, 1.02, 1.03, 1.06, 1.18, 1.50, 1.80, 1.72, 1.52, 1.45,
                         1.45, 1.45, 1.52, 1.68, 1.88, 1.95, 1.85, 1.55, 1.32, 1.20, 1.12, 1.06),
            tti_weekend=(1.05, 1.04, 1.02, 1.02, 1.02, 1.03, 1.06, 1.10, 1.18, 1.26, 1.34, 1.40,
                         1.44, 1.45, 1.45, 1.46, 1.48, 1.48, 1.42, 1.32, 1.22, 1.16, 1.10, 1.06),
            am_peak_hour=8.5, pm_peak_hour=17.5,
            gravity_km=4.0,
            # ASSUMPTION: Mediterranean climate, wet season November-March.
            rain_prob_by_month=(0.33, 0.33, 0.28, 0.17, 0.08, 0.03, 0.01, 0.01, 0.03, 0.12, 0.23, 0.33),
            incidents_per_100km_day=2.0,
            # ASSUMPTION: roughly half of SF's jobs are held by in-commuters, and
            # US-101 carries Marin <-> Peninsula through traffic.
            external_share=0.35, through_share=0.15,
            # The county line runs mid-span on both bridges, >1.5 km from where the
            # graph ends at the Golden Gate toll plaza and Yerba Buena Island.
            gateway_max_dist_m=2500.0,
            source="TomTom Traffic Index 2025 (tomtom.com/traffic-index/city/san-francisco-ca)",
        ),
        market=RideMarket(
            # SFCTA "TNCs Today" (2017) counted ~170k intra-SF TNC trips on a typical
            # weekday. ASSUMPTION: this company carries ~7% of that.
            requests_per_day=12000, weekend_factor=1.05,
            # ASSUMPTION: shape after SFCTA's hourly TNC profile: evening peak larger
            # than the morning one, busy late evenings, weekends late-night heavy.
            demand_weekday=(2.5, 1.6, 1.1, 0.6, 0.5, 0.9, 2.2, 4.5, 6.2, 5.6, 4.6, 4.6,
                            4.9, 4.8, 4.8, 5.2, 5.9, 7.0, 7.3, 6.4, 5.4, 4.7, 4.3, 3.4),
            demand_weekend=(4.8, 4.0, 3.2, 1.6, 0.8, 0.6, 0.9, 1.4, 2.4, 3.4, 4.2, 4.8,
                            5.2, 5.4, 5.4, 5.4, 5.6, 5.8, 6.0, 5.8, 5.4, 5.2, 5.2, 5.0),
            ride_gravity_km=3.0,           # SFCTA: mean intra-SF TNC trip ~3 miles
            intercity_share=0.06,
            external=(
                ExternalDest("SFO airport (US-101 S)", (-122.4014, 37.7239), 16.0, 16.0, 0.45),
                ExternalDest("Oakland / East Bay (Bay Bridge)", (-122.3645, 37.8107), 14.0, 18.0, 0.25),
                ExternalDest("Marin (Golden Gate)", (-122.4724, 37.8043), 12.0, 16.0, 0.10),
                ExternalDest("Peninsula (I-280 S)", (-122.4485, 37.7188), 25.0, 24.0, 0.20),
            ),
            n_drivers=1000,
            shift_starts=((6.0, 0.25), (10.0, 0.20), (15.0, 0.30), (20.0, 0.25)),
            shift_median_h=6.0,
            n_customers=80000,
            # ASSUMPTION: UberX-like SF tariff, 2025 order of magnitude.
            base_fare=2.0, per_km=0.85, per_min=0.40, min_fare=8.0, booking_fee=3.0,
            commission=0.25, cancel_fee=5.0, intercity_per_km=1.0,
            tip_prob=0.30, tip_frac_mean=0.15,
            cost_per_km=0.18,              # fuel + maintenance + tyres, USD/km
            max_pickup_min=12.0,
        ),
    ),
}

# ---------------------------------------------------------------- generated cities
#
# The 13 added cities are built from two templates (India = Pune's shapes, US = SF's)
# scaled to each city. Every scaled quantity is an ASSUMPTION except where a TomTom
# 2025 congestion level is given as the source.

def utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _scaled_tti(tpl: tuple, tpl_level: float, level: float) -> tuple:
    """Scale a template's excess delay so its congestion level matches `level`."""
    k = level / tpl_level
    return tuple(round(1 + (v - 1) * k, 3) for v in tpl)


# ASSUMPTION: requests per day per 1,000 residents for one mid-sized platform, and
# drivers per daily request before fleet auto-sizing (scripts/01b_size_fleet.py).
_RATE = {"IN": 2.9, "US": 12.0}
_DRIVERS_PER_REQ = {"IN": 0.14, "US": 0.085}


def _make(key, name, osm_query, country, tier, population, centre, congestion, source,
          circle_km=None, show_boundary=True):
    pune, sf = CITIES["pune"], CITIES["sf"]
    tpl = pune if country == "IN" else sf
    tpl_level = 0.71 if country == "IN" else 0.497
    t = tpl.traffic
    reqs = int(round(_RATE[country] * population / 1000 / 100) * 100)
    reqs = max(reqs, 600)
    gravity = 3.0 if tier == "small" else 4.0 if tier == "medium" else 5.0
    m = replace(
        tpl.market,
        requests_per_day=reqs,
        n_drivers=max(60, int(reqs * _DRIVERS_PER_REQ[country])),
        n_customers=max(4000, int(reqs * 6.5)),
        ride_gravity_km=gravity * (0.7 if country == "US" else 1.0),
        # intercity destinations: the three gateways with the most road capacity
        external=(ExternalDest("Outskirts via main gateway", None, 15.0, 22.0, 0.5, 0),
                  ExternalDest("Outskirts via second gateway", None, 12.0, 20.0, 0.3, 1),
                  ExternalDest("Outskirts via third gateway", None, 10.0, 18.0, 0.2, 2)),
    )
    tr = replace(
        t,
        tti_weekday=_scaled_tti(t.tti_weekday, tpl_level, congestion),
        tti_weekend=_scaled_tti(t.tti_weekend, tpl_level, congestion),
        gravity_km=gravity + 1.0,
        # ASSUMPTION: satellite cities in a metro exchange more traffic with the core
        external_share=0.35 if tier == "metro" else 0.20 if tier == "medium" else 0.15,
        gateway_max_dist_m=1500.0,
        source=source,
    )
    return City(
        key=key, name=name, osm_query=osm_query, utm_crs=utm_epsg(*centre),
        currency=tpl.currency, timezone=tpl.timezone, hwy_speeds_kph=tpl.hwy_speeds_kph,
        h3_res=None, show_boundary=show_boundary, traffic=tr, market=m,
        circle=(centre[0], centre[1], circle_km) if circle_km else None,
        country=country, tier=tier, population=population,
    )


_TT = "TomTom Traffic Index 2025"
_NEW = [
    # key, name, OSM query / relation, country, tier, population, centre, congestion level, source, circle km
    ("navimumbai", "Navi Mumbai", "R13180880", "IN", "metro", 1_120_000, (19.0330, 73.0297),
     0.55, f"ASSUMPTION: below {_TT} Mumbai 63.2% (satellite city)", None),
    ("thane", "Thane", "", "IN", "metro", 1_890_000, (19.2183, 72.9781),
     0.60, f"ASSUMPTION: near {_TT} Mumbai 63.2%", 6.8),
    ("arlington", "Arlington, TX", "R115329", "US", "metro", 400_000, (32.7357, -97.1081),
     0.34, f"ASSUMPTION: between {_TT} Dallas 40.6% and Fort Worth 28.3%", None),
    ("nagpur", "Nagpur", "R12969115", "IN", "medium", 2_500_000, (21.1458, 79.0882),
     0.45, "ASSUMPTION: Indian tier-2 city, not in TomTom", None),
    ("coimbatore", "Coimbatore", "", "IN", "medium", 1_100_000, (11.0168, 76.9558),
     0.45, "ASSUMPTION: Indian tier-2 city, not in TomTom", 9.0),
    ("chandigarh", "Chandigarh", "R1942809", "IN", "medium", 1_100_000, (30.7333, 76.7794),
     0.40, "ASSUMPTION: planned Indian tier-2 city, not in TomTom", None),
    ("pittsburgh", "Pittsburgh", "R188553", "US", "medium", 303_000, (40.4406, -79.9959),
     0.373, f"{_TT} Pittsburgh 37.3%", None),
    ("portland", "Portland, OR", "R186579", "US", "medium", 630_000, (45.5152, -122.6784),
     0.370, f"{_TT} Portland 37.0%", None),
    ("panaji", "Panaji", "", "IN", "small", 115_000, (15.4909, 73.8278),
     0.30, "ASSUMPTION: small Indian city, not in TomTom", 4.0),
    ("udaipur", "Udaipur", "", "IN", "small", 500_000, (24.5854, 73.7125),
     0.30, "ASSUMPTION: small Indian city, not in TomTom", 5.0),
    ("puducherry", "Puducherry", "R10088305", "IN", "small", 250_000, (11.9416, 79.8083),
     0.30, "ASSUMPTION: small Indian city, not in TomTom", None),
    ("boulder", "Boulder, CO", "R112298", "US", "small", 105_000, (40.0150, -105.2705),
     0.22, "ASSUMPTION: small US city, not in TomTom", None),
    ("annarbor", "Ann Arbor, MI", "R135130", "US", "small", 123_000, (42.2808, -83.7430),
     0.22, "ASSUMPTION: small US city, not in TomTom", None),
]
for _row in _NEW:
    _k, _n, _q, _c, _t, _p, _ctr, _lvl, _src, _r = _row
    CITIES[_k] = _make(_k, _n, _q, _c, _t, _p, _ctr, _lvl, _src, circle_km=_r)
CITIES["pune"] = replace(CITIES["pune"], country="IN", tier="core", population=3_500_000)
CITIES["sf"] = replace(CITIES["sf"], country="US", tier="core", population=810_000)

TIERS = {"core": ["sf", "pune"], "metro": ["navimumbai", "thane", "arlington"],
         "medium": ["nagpur", "coimbatore", "chandigarh", "pittsburgh", "portland"],
         "small": ["panaji", "udaipur", "puducherry", "boulder", "annarbor"]}

# Road classes shown in plots, from most to least important.
ROAD_CLASSES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "unclassified", "living_street", "service", "other",
]


def get(key: str) -> City:
    """The city's config, with any fleet size written by the auto-sizing step."""
    try:
        city = CITIES[key]
    except KeyError:
        raise SystemExit(f"unknown city {key!r}; choose from {sorted(CITIES)}")
    f = city.data_dir / "fleet.json"
    if f.exists():
        city = replace(city, market=replace(city.market, n_drivers=json.loads(f.read_text())["n_drivers"]))
    return city
