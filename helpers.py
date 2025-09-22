from __future__ import annotations
import os, re, math, json
from typing import Any, Dict, List, Optional, Tuple
import requests

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from langgraph.graph import StateGraph, END

# ===== iNat iconic taxa per UI class =====
ICONIC_TAXA_MAP: Dict[str, List[str]] = {
    "All": [],
    "Mammals": ["Mammalia"],
    "Birds": ["Aves"],
    "Reptiles": ["Reptilia"],
    "Amphibians": ["Amphibia"],
    "Fish": ["Actinopterygii", "Chondrichthyes"],
    "Insects": ["Insecta"],
    "Arachnids": ["Arachnida"],
    "Molluscs": ["Mollusca"],
}

BROADEN_MAX_KM = 30
INAT_BASE = "https://api.inaturalist.org/v1"
USER_AGENT = "WildScopeLangGraph/1.0"

# ===== Utils =====
def sanitize(s: str) -> str: return re.sub(r"\s+", " ", (s or "").strip())
def km_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def http_get_json(url: str, params: Dict[str, Any] | None = None, timeout: int = 25) -> Optional[Dict]:
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ===== Geocoding =====
_GEOCODER = Nominatim(user_agent="WildScopeLangGraph-geocoder")
_GEOCODE = RateLimiter(_GEOCODER.geocode, min_delay_seconds=1, swallow_exceptions=True)

def geocode_place(text: str) -> Optional[Tuple[str, float, float]]:
    if not text or not text.strip(): return None
    loc = _GEOCODE(text)
    if not loc: return None
    return (loc.address, float(loc.latitude), float(loc.longitude))

# ===== Photo helpers =====
def upscale_inat_url(url: str, target: str = "large") -> str:
    if not url or not isinstance(url, str): return url
    return re.sub(
        r"/(square|small|medium|large|original)\.(jpg|jpeg|png)(\?.*)?$",
        fr"/{target}.\2\3" if r"\3" else fr"/{target}.\2",
        url, flags=re.IGNORECASE
    )

def best_photo_from_observation(photo: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    keys = ["original_url", "large_url", "medium_url", "url", "small_url", "square_url"]
    cand = [photo.get(k) for k in keys if photo.get(k)]
    if not cand: return None, None
    best = upscale_inat_url(cand[0], target="large")
    thumb = upscale_inat_url((cand[1] if len(cand) > 1 else cand[0]), target="large")
    return thumb, best

def photo_from_taxon_default(taxon: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    d = taxon.get("default_photo") if isinstance(taxon, dict) else None
    if not d: return None, None
    keys = ["original_url", "large_url", "medium_url", "url", "square_url"]
    cand = [d.get(k) for k in keys if d.get(k)]
    if not cand: return None, None
    best = upscale_inat_url(cand[0], target="large")
    thumb = upscale_inat_url((cand[1] if len(cand) > 1 else cand[0]), target="large")
    return thumb, best

# ===== Species model =====
class SpeciesRecord:
    def __init__(self, taxon_id: int, scientific_name: str, common_name: str, iconic_class: str):
        self.taxon_id = taxon_id
        self.scientific_name = scientific_name
        self.common_name = common_name
        self.iconic_class = iconic_class
        self.total_observations = 0
        self.nearest_km: Optional[float] = None
        self.thumb_url: Optional[str] = None
        self.best_photo_url: Optional[str] = None
        self.wikipedia_url: Optional[str] = None

# ===== iNaturalist =====
def fetch_observations(lat: float, lon: float, radius_km: int, iconic_taxa: List[str] | None = None, pages: int = 3) -> List[Dict]:
    iconic_taxa = iconic_taxa or []
    out: List[Dict] = []
    per_page = 200
    for page in range(1, pages + 1):
        params = {
            "lat": lat, "lng": lon,
            "radius": max(1, min(radius_km, 200)),
            "verifiable": "true", "photos": "true",
            "order_by": "observed_on", "order": "desc",
            "per_page": per_page, "page": page,
        }
        if iconic_taxa:
            params["iconic_taxa"] = ",".join(iconic_taxa)
        data = http_get_json(f"{INAT_BASE}/observations", params=params)
        if not data or "results" not in data: break
        batch = data["results"]
        if not batch: break
        out.extend(batch)
        if len(batch) < per_page: break
    return out

def observations_to_species(obs: List[Dict], user_lat: float, user_lon: float, limit_species: int, area_label: str) -> List[SpeciesRecord]:
    species_map: Dict[int, SpeciesRecord] = {}
    for item in obs:
        taxon = item.get("taxon") or {}
        t_id = taxon.get("id")
        if not t_id: continue
        sci = sanitize(taxon.get("name")); com = sanitize(taxon.get("preferred_common_name")); iconic = sanitize(taxon.get("iconic_taxon_name"))
        if t_id not in species_map:
            rec = SpeciesRecord(t_id, sci, com, iconic)
            rec.wikipedia_url = (taxon.get("wikipedia_url") or "")[:400]
            t_thumb, t_best = photo_from_taxon_default(taxon)
            if t_best: rec.thumb_url, rec.best_photo_url = t_thumb, t_best
            else:
                photos = item.get("photos") or []
                if photos:
                    o_thumb, o_best = best_photo_from_observation(photos[0])
                    rec.thumb_url, rec.best_photo_url = o_thumb, o_best
            species_map[t_id] = rec

        rec = species_map[t_id]
        rec.total_observations += 1

        # nearest distance among its obs
        latlon = None
        if "geojson" in item and isinstance(item["geojson"], dict):
            coords = item["geojson"].get("coordinates")
            if isinstance(coords, list) and len(coords) == 2:
                latlon = (float(coords[1]), float(coords[0]))
        if not latlon:
            loc_str = item.get("location")
            if isinstance(loc_str, str) and "," in loc_str:
                try:
                    lat_s, lon_s = [p.strip() for p in loc_str.split(",", 1)]
                    latlon = (float(lat_s), float(lon_s))
                except Exception:
                    latlon = None
        if latlon:
            d = km_distance(user_lat, user_lon, latlon[0], latlon[1])
            rec.nearest_km = round(min(rec.nearest_km or d, d), 2)

    species = list(species_map.values())
    species.sort(key=lambda r: (-r.total_observations, r.nearest_km or 9_999.0))
    return species[:limit_species]

# ===== Deterministic plan (locks class; adjusts pages for hotspots) =====
def fallback_plan(ctx: Dict[str, Any]) -> Dict[str, Any]:
    place_l = (ctx["place"] or "").lower()
    in_hotspot = any(h in place_l for h in ("kruger","serengeti","masai","maasai","okavango","south africa","tanzania","kenya","botswana","zimbabwe","namibia","zambia"))
    pages = 5 if (ctx["intent"] == "crowd-pleasers" and in_hotspot) else 3
    radius = max(5, min(int(ctx["ui_radius_km"]), BROADEN_MAX_KM))
    return {"radius_km": radius, "ui_class": ctx["ui_class"], "max_pages": pages}

# ===== Safari hotspot reranker (deterministic) =====
SAFARI_HINTS = (
    "kruger","serengeti","masai","maasai","okavango",
    "south africa","tanzania","kenya","botswana","zimbabwe","namibia","zambia"
)
BIG_FIVE_TOKENS = ("lion","leopard","elephant","rhinoceros","rhino","buffalo")
AFR_MEGA_TOKENS = BIG_FIVE_TOKENS + ("cheetah","giraffe","hippopotamus","hippo","zebra","wild dog","lycaon")

def hotspot_rerank(species: List[SpeciesRecord], place: str, intent: str) -> List[SpeciesRecord]:
    place_l = (place or "").lower()
    if not any(h in place_l for h in SAFARI_HINTS):
        return species

    def score(s: SpeciesRecord):
        name = f"{(s.common_name or '').lower()} {(s.scientific_name or '').lower()}"
        tok = 0
        for t in AFR_MEGA_TOKENS:
            if t in name:
                tok += 1000 if t in BIG_FIVE_TOKENS else 500
        return (tok, s.total_observations, -(s.nearest_km or 9999.0))

    return sorted(species, key=score, reverse=True)

# ===== Robust fetch: broaden radius/pages until coverage is decent =====
def fetch_with_retries(lat: float, lon: float, radius_km: int, iconic: List[str], pages: int,
                       min_species: int = 15, max_radius_km: int = BROADEN_MAX_KM, max_pages: int = 5):
    tries = [
        (radius_km, pages),
        (min(max(radius_km * 2, radius_km + 10), max_radius_km), pages),
        (min(max(radius_km * 2, radius_km + 10), max_radius_km), min(max_pages, pages + 2)),
    ]
    for r, p in tries:
        obs = fetch_observations(lat, lon, r, iconic, p)
        yield r, p, obs
        if obs and len(obs) >= min_species:
            break

# ===== Deterministic, hotspot-aware ranking =====
def robust_rank(species: List[SpeciesRecord], place: str, intent: str) -> List[SpeciesRecord]:
    species = hotspot_rerank(species, place, intent)
    species.sort(
        key=lambda s: (
            -int(s.total_observations or 0),
            float(s.nearest_km or 9_999.0),
            (s.common_name or s.scientific_name or "").lower()
        )
    )
    return species

# ===== Naturalist summary (LLM optional) =====
def summarize_with_llm(place: str, intent: str, species: List[SpeciesRecord]) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
        rows = [{
            "common": s.common_name, "scientific": s.scientific_name,
            "class": s.iconic_class, "observations": s.total_observations,
            "nearest_km": s.nearest_km
        } for s in species[:30]]

        sys = "You are a concise, factual field naturalist. Use ONLY provided rows. 120–160 words. No emojis."
        usr = (
            f"Area: {place}\n"
            f"Intent: {intent}\n"
            f"Rows (JSON):\n{json.dumps(rows, ensure_ascii=False)}\n\n"
            "Write a visitor-friendly overview highlighting 3–5 notable species, behavior/ecology hints, and etiquette (keep distance, no feeding)."
        )
        for m in models:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                    temperature=0.2,
                )
                txt = (resp.choices[0].message.content or "").strip()
                if txt:
                    return txt
            except Exception:
                continue
    except Exception:
        return None
    return None

def fallback_summary(place: str, species: List[SpeciesRecord]) -> str:
    names = [s.common_name or s.scientific_name for s in species[:6] if (s.common_name or s.scientific_name)]
    head = ", ".join(names[:4]) + ("…" if len(names) > 4 else "")
    return (
        f"**Area:** {place}. This list reflects recent iNaturalist observations near you, "
        f"ranked by observation volume and proximity. Highlights: {head}. "
        "Please keep a respectful distance, avoid feeding, and stay on marked routes."
    )

# ===== LangGraph: plan → fetch → rank → summarize =====
def build_graph():
    class State(dict):
        """State dict: {'ctx': {...}, 'trace': [...], 'species': [...], 'summary': str, 'cards': [...]}"""

    def n_plan(state: State) -> State:
        # guard input
        ctx_in = (state or {}).get("ctx", {}) or {}
        trace = state.get("trace", [])

        # 1) geocode
        geo = geocode_place(ctx_in.get("place_query", ""))
        if not geo:
            trace.append(f"Geocode failed for '{ctx_in.get('place_query','')}'.")
            state["trace"] = trace
            state["ctx"] = {}  # ensure exists to avoid KeyError downstream
            state["species"] = []
            state["cards"] = []
            state["summary"] = "Could not geocode that place. Try a broader landmark."
            return state

        display, lat, lon = geo

        # 2) deterministic plan
        ctx = {
            "place": display, "lat": lat, "lon": lon,
            "ui_radius_km": int(ctx_in.get("ui_radius_km", 10)),
            "ui_class": ctx_in.get("ui_class", "All"),
            "intent": ctx_in.get("intent", "crowd-pleasers"),
            "max_species": int(ctx_in.get("max_species", 60)),
        }
        plan = fallback_plan({
            "place": display,
            "ui_radius_km": ctx["ui_radius_km"],
            "ui_class": ctx["ui_class"],
            "intent": ctx["intent"],
        })
        trace.append(f"Plan: radius={plan['radius_km']} km, class={plan['ui_class']}, pages={plan['max_pages']} (locked class)")

        state["ctx"] = {**ctx, **plan}
        state["trace"] = trace
        return state

    def n_fetch(state: State) -> State:
        trace = state.get("trace", [])
        c = state.get("ctx") or {}
        # guard: if ctx missing (e.g., geocode failed), stop here
        if not c or "lat" not in c or "lon" not in c:
            state["trace"] = trace + ["Fetch skipped (no context)."]
            state["species"] = []
            state["cards"] = []
            return state

        iconic = ICONIC_TAXA_MAP.get(c["ui_class"], [])
        best_obs = []; best_r, best_p = c["radius_km"], c["max_pages"]
        for r, p, obs in fetch_with_retries(
            c["lat"], c["lon"], c["radius_km"], iconic, c["max_pages"],
            min_species=15, max_radius_km=BROADEN_MAX_KM, max_pages=5
        ):
            if obs:
                best_obs = obs; best_r, best_p = r, p
                trace.append(f"Fetched preview: {len(obs)} obs @ {r} km / {p} pages")
            else:
                trace.append(f"No data @ {r} km / {p} pages")

        if not best_obs:
            state["trace"] = trace
            state["species"] = []
            state["cards"] = []
            state["summary"] = "No observations found. Try a wider radius or different class."
            return state

        species = observations_to_species(best_obs, c["lat"], c["lon"], int(c.get("max_species", 60)), c["place"])

        # Hard class filter
        if c["ui_class"] != "All":
            wanted = (ICONIC_TAXA_MAP[c["ui_class"]][0] or "").lower()
            before = len(species)
            species = [s for s in species if (s.iconic_class or "").lower() == wanted]
            trace.append(f"Class filter '{c['ui_class']}' → {before} → {len(species)} species")

        state["species"] = species
        state["trace"] = trace
        return state

    def n_rank(state: State) -> State:
        c = state.get("ctx") or {}
        species = state.get("species") or []
        if not c or not species:
            state["cards"] = []
            return state

        species = robust_rank(species, c["place"], c["intent"])
        # prepare cards for UI
        cards = []
        for s in species:
            cards.append({
                "taxon_id": s.taxon_id,
                "name": s.common_name or s.scientific_name or f"taxon {s.taxon_id}",
                "scientific_name": s.scientific_name or "",
                "iconic_class": s.iconic_class or "",
                "img": s.best_photo_url or s.thumb_url or "",
                "meta": f"Obs: {int(s.total_observations)}" + (f" · Nearest: {s.nearest_km} km" if s.nearest_km is not None else "")
            })
        state["species"] = species
        state["cards"] = cards
        return state

    def n_summarize(state: State) -> State:
        c = state.get("ctx") or {}
        species = state.get("species") or []
        if not c:
            state["summary"] = state.get("summary", "No context available.")
            return state
        txt = summarize_with_llm(c["place"], c["intent"], species)
        if not txt:
            txt = fallback_summary(c.get("place",""), species)
        state["summary"] = txt
        return state

    graph = StateGraph(dict)
    graph.add_node("plan", n_plan)
    graph.add_node("fetch", n_fetch)
    graph.add_node("rank", n_rank)
    graph.add_node("summarize", n_summarize)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "fetch")
    graph.add_edge("fetch", "rank")
    graph.add_edge("rank", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()