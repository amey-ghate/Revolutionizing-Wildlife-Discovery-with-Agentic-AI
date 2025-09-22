# WildScope — Agentic AI Wildlife Finder

**WildScope** is a Streamlit app that finds wildlife **near any location on Earth** using open, live data (iNaturalist + Wikipedia + OSM).  
A lightweight **LangGraph** workflow plans the search, **auto-broadens** if needed, applies **deterministic, hotspot-aware ranking** (e.g., Big Five in safari zones when present), and generates a succinct **naturalist summary**.

---

## Why WildScope beats a Google search

- **Live & local vs. evergreen & generic**  
  Google shows “Top animals in Kruger” articles; WildScope pulls **recent observations near *your* coordinates** and ranks by popularity and proximity.

- **Agentic workflow, not manual guesswork**  
  It **plans and adapts**: widens the radius/pages when the immediate area is sparse, then re-ranks—no fiddly search gymnastics.

- **Hotspot-aware relevance**  
  In safari regions it prioritizes lion, elephant, rhino, buffalo (and cheetah/giraffe/hippo/zebra) **when they actually appear** in the data.

- **Photo-smart**  
  Prefers curated taxon images from iNaturalist to avoid bones/skulls/scat thumbnails.

- **Class-locked filters**  
  If you choose **Mammals**, you see **Mammalia**—no plants or unrelated taxa slipping in.

- **LLM optional**  
  The app is fully useful without an LLM. A Groq key simply upgrades the **summary** quality.

---

## Features

- 🌍 Global location search (parks, cities, addresses)  
- 🧭 Agentic flow (LangGraph): **plan → fetch (auto-broaden) → rank → summarize**  
- 🦁 Hotspot-aware, deterministic ranking (stable & trustworthy)  
- 🖼️ Image-first UI with clean, responsive cards  
- 📚 Inline wiki blurb per species (concise)  
- 🔒 Works offline from LLMs; Groq only enhances narrative

---

## Tech Stack

- **Frontend:** Python + Streamlit  
- **Orchestration:** LangGraph  
- **Data:** iNaturalist (observations/photos), Wikipedia REST (summaries), Nominatim/OSM (geocoding)  
- **Optional LLM:** Groq (llama-3.1-70b-versatile or llama-3.1-8b-instant)

---

## Quickstart (Windows / PowerShell)

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
Create .env (optional, for better summaries):
```bash
GROQ_API_KEY=your_groq_key
```
Run
```bash
python -m streamlit run app.py
```
---

## How it works

- Plan — Geocodes your place, locks your chosen class, sets initial radius/pages (extra pages for safari hotspots).

- Fetch — Pulls recent iNaturalist observations; if too few, auto-broadens radius/pages.

- Rank — Deterministic ordering:

    - Safari hotspots: Big Five & iconic safari mammals get a boost when present

    - Then sort by observation volume → distance → name

- Summarize — Optional Groq LLM writes a 120–160-word field-guide note; otherwise, a clean fallback is used.

---
File Structure
--------------

*   app.py: Streamlit UI + calls into the agentic graph

*  helpers.py: iNat/Wiki utils, ranking, auto-broaden fetch, LangGraph build
    
*   requirements.txt: Python dependencies.
    
*   .env: Environment file to store Groq API key.
    
*   README.md: Documentation.
