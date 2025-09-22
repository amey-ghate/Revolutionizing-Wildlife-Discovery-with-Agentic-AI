from __future__ import annotations
import streamlit as st
from dotenv import load_dotenv
from helpers import ICONIC_TAXA_MAP, BROADEN_MAX_KM, build_graph

load_dotenv()
st.set_page_config(page_title="WildScope — LangGraph (Robust)", page_icon="🦉", layout="wide")

CSS = """
<style>
#MainMenu, footer {visibility:hidden}
.block-container { padding-top: .6rem; }
.ws-scroll { max-height: 68vh; overflow-y: auto; padding-right: .25rem; }
.ws-card { display:grid; grid-template-columns:300px 1fr; gap:14px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; margin:.8rem 0; min-height:220px; }
.ws-card .img{ background:#f1f5f9; border-top-left-radius:14px; border-bottom-left-radius:14px; overflow:hidden; position:relative }
.ws-card img{ width:100%; height:100%; object-fit:cover }
.ws-badge{ position:absolute; top:8px; left:8px; background:#000a; color:#fff; padding:.15rem .45rem; border-radius:8px; font-size:.75rem }
.ws-body{ padding:.9rem 1.1rem }
.ws-name{ font-weight:800 }
.ws-sub{ color:#475569 }
.ws-meta{ margin-top:.4rem; color:#0f172a; font-size:.94rem }
.think-chip{ display:inline-block; margin:.15rem .35rem .15rem 0; padding:.18rem .5rem; border:1px solid #cbd5e1; border-radius:999px; font-size:.83rem; background:#f8fafc }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def header():
    st.markdown("<h2 style='text-align:center'>🦉 WildScope — LangGraph (Robust)</h2>", unsafe_allow_html=True)
    st.caption("LangGraph orchestrates: plan → fetch → rank (deterministic) → summarize. Thinking is shown.")

def sidebar():
    st.sidebar.markdown("### Search")
    place = st.sidebar.text_input("📍 Location", "", placeholder="e.g., Kruger National Park")
    radius = st.sidebar.slider("🔭 Radius (km)", 1, BROADEN_MAX_KM, 10)
    animal_class = st.sidebar.selectbox("🧬 Class", list(ICONIC_TAXA_MAP.keys()), index=1)  # default Mammals
    max_species = st.sidebar.slider("🧮 Max species", 10, 200, 60, step=10)
    intent = st.sidebar.selectbox("🎯 Intent", ["crowd-pleasers", "rare-birds", "family-friendly", "pelagic"], 0)
    show_thinking = st.sidebar.toggle("Show thinking", value=True)
    run = st.sidebar.button("Search wildlife", type="primary", use_container_width=True)
    return place, radius, animal_class, max_species, intent, show_thinking, run

def thinking_panel(lines: list[str]):
    if not lines: return
    st.markdown("### Agent thinking")
    for line in lines:
        st.markdown(f"<span class='think-chip'>{line}</span>", unsafe_allow_html=True)

def render_cards(species_cards: list[dict]):
    st.markdown("### Nearby animals (robust-ranked)")
    if not species_cards:
        st.info("No species found. Try increasing the radius or changing the class.")
        return
    st.markdown("<div class='ws-scroll'>", unsafe_allow_html=True)
    for s in species_cards:
        name = s["name"]
        sci = s.get("scientific_name","")
        img = s.get("img","")
        meta = s["meta"]
        st.markdown(
            f"""
<div class="ws-card">
  <div class="img">
    {("<span class='ws-badge'>HD</span><img src='"+img+"'/>") if img else "<div style='height:100%;display:flex;align-items:center;justify-content:center;color:#64748b'>No image</div>"}
  </div>
  <div class="ws-body">
    <div class="ws-name">{name}</div>
    <div class="ws-sub"><i>{sci}</i> — {s.get("iconic_class","")}</div>
    <div class="ws-meta">{meta}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    header()
    place, radius_km, animal_class, max_species, intent, show_thinking, run = sidebar()
    if not run:
        st.info("Enter a location, adjust filters, then click Search.")
        return
    if not place.strip():
        st.warning("Please enter a valid location.")
        return

    graph = build_graph()
    with st.spinner("Thinking & fetching…"):
        result = graph.invoke({
            "ctx": {
                "place_query": place,
                "ui_radius_km": radius_km,
                "ui_class": animal_class,
                "intent": intent,
                "max_species": max_species,
                "show_thinking": show_thinking,
            }
        })

    # Trace / summary / cards are optional keys; guard on missing
    if show_thinking and result.get("trace"):
        thinking_panel(result["trace"])

    if result.get("summary"):
        st.markdown("### Naturalist summary")
        st.markdown(result["summary"])

    render_cards(result.get("cards", []))

    st.markdown("---")
    if st.button("🔁 New search"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()