# 🚀 Ludex: A Hybrid Content + Collaborative Game Recommendation System for Steam

Ludex is a machine learning based game recommendation engine built for the Steam store as part of a 5th-semester B.Tech project at IIIT Pune.  
It aims to replace the “discoverability lottery” with **deeply personalized**, **content-aware**, **CF-enhanced**, and **diversity-rich** recommendations.

Ludex learns from:

- What you **play**
- How long you **play it**
- What the **games actually are** (tags, genres, mechanics, writing style)
- Your **anchor games** (your core taste)
- Playtime similarity across **millions of users** (CF)

> ### 🔷 Status (2025)
>
> ✔ Complete **CBF Pipeline** with TF-IDF embeddings  
> ✔ **Auto-extending game catalogue** (no missing owned games)  
> ✔ **CBF: Global Taste Vector + Anchor Reinforcement + MMR**  
> ✔ Fully working **CF implicit ALS model** (training, updating, recommending)  
> ✔ Automatic **interaction enrichment + conditional retrain**

> ❗ In Progress
>
> - Final Hybrid Blending (CBF + CF unified score)
> - Evaluation Suite (Recall@K, MAP, NDCG)
> - Simple Web UI

---

# 🧠 System Overview

**Pipeline:**

1. Steam crawl → basic appID list
2. Refinement (remove NSFW, CJK/Arabic titles, duplicates)
3. Full metadata scrape (genres/tags/description/developer/publisher)
4. TF-IDF + OHE + weighted feature blocks
5. L2-normalized embeddings (`recommender_matrix.npz`)
6. User CBF profile
7. Anchor-based micro-preference model
8. Diversity layer (MMR)
9. CF implicit ALS model
10. Hybrid-ready scoring

Every run of the recommender **auto-detects missing games**, scrapes them, and **rebuilds the matrix on the fly**.

---

# 🔄 CBF Pipeline (Implemented)

## 1. TF-IDF & Metadata Feature Blocks

Weights are tuned to emphasize genres and tags:

| Block         | Encoder         | Weight |
| ------------- | --------------- | ------ |
| Genres + Tags | TF-IDF (1–2g)   | 0.90   |
| Title         | TF-IDF (1–2g)   | 0.25   |
| Description   | TF-IDF (n-gram) | 0.20   |
| Developers    | OHE             | 0.20   |
| Publishers    | OHE             | 0.10   |

Final embedding per game:
`f_i = Normalize( title ; tags ; description ; developer ; publisher )`

Saved as `recommender_matrix.npz`.

---

# 🌟 Hybrid CBF Engine (Implemented, CF-ready)

Ludex uses a modern multi-stage personalization mechanism.

## 1. Global User Vector

User vector `u` is built from playtime-weighted weighted embeddings:
`score_global[i] = dot(u, f_i)`

This captures long-term preference.

---

## 2. Anchor-Based Reinforcement

Anchor games = top-playtime titles.

For each anchor game `a`:

This captures long-term preference.

---

## 2. Anchor-Based Reinforcement

Anchor games = top-playtime titles.

For each anchor game `a`:
`anchor[a][i] = dot(f_a, f_i)`

Then combine:
`anchor_soft[i] = Σ (w_a * anchor[a][i])`

This boosts micro-tastes (e.g., if you love roguelite platformers, they naturally rise).

---

## 3. Blended CBF Score

`combined_raw[i] = (1 − β) * score_global[i] + β * anchor_soft[i]`

β typically = **0.3**.

---

## 4. MMR Diversity

Maximal Marginal Relevance ensures genre diversity:

`final[i] = λ * combined_raw[i] − (1 − λ) * max_sim_to_selected(i)`

λ ≈ **0.7** gives a healthy mix of comfort picks + diverse exploration.

---

# 🤝 CF (Collaborative Filtering)

Ludex implements **implicit ALS** collaborative filtering:

Components include:

- `CF/cf_model.py`

  - trains / loads the ALS model
  - manages `cf_als_model.pkl` + `cf_als_index.pkl`

- `CF/interactions_update.py`

  - loads `user_game_playtime_top20.csv`
  - auto-adds missing users (via Steam API)
  - grows interaction matrix
  - triggers conditional retrain

- `CF/CF_recommend.py`
  - main CF recommendation engine with:
    - popularity normalization
    - friend-weighted re-ranking
    - fallback logic
    - cold-start strategies

> CF is **fully operational** and used in production.  
> What remains is the full **CBF+CF hybrid score combination**.

---

# 🔮 Planned Hybrid Score

Planned final combination:
`Hybrid(u, i) = α * CF_norm(u, i) + (1 − α) * CBF_norm(u, i)`

- Strong CBF → lower α
- Weak CBF (few games) → higher α

Currently CBF runs standalone; CF also runs standalone.  
Hybrid wiring is trivial and will be added next.

---

# 🧩 Design Principles

- **No missing games** (auto extend catalogue)
- **Explainability** through anchor games
- **Fair genre representation** through MMR
- **Balanced personalization**
- **CF + CBF complementarity**
- **Full modularity**
- **Steam API caching** to minimize API calls

---

# 📅 Roadmap (2025–2026)

### Short-term

- Add hybrid scoring module
- Build evaluation suite (Recall@K, Precision@K, MAP, NDCG)

### Medium-term

- Simple web UI with Steam login
- Real-time recommendation preview

---

# 📁 Data Files

- `data/raw/game_details.csv` — scraped metadata
- `data/raw/user_game_playtime_top20.csv` — interactions for CF
- `data/processed/recommender_matrix.npz` — CBF embeddings
- `data/processed/cf_als_model.pkl` — CF model
- `data/processed/cf_als_index.pkl` — CF mapping (item/user IDs)

---

# ⚡ Quick Start

Follow these steps to run Ludex locally:

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ludex.git
cd ludex
```

## 2. Create Virtual Environment

Ensure that python version = 3.12.xx

```bash
python -m venv myenv
source myenv/bin/activate      # Linux/Mac
myenv\Scripts\activate         # Windows
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Add your Steam Web API Key

Copy .env.example to .env:
Edit .env and insert your Steam API key:

```bash
cp .env.example .env
STEAM_API_KEY=YOUR_KEY_HERESTEAM_API_KEY=YOUR_KEY_HERE
```

## 5. Prepare Raw Data

Run the crawler scripts to populate data/raw/ with game metadata:

```bash
python crawlers/spider.py
python crawlers/spider_refiner.py
python crawlers/spider_details.py
python crawlers/user_topgames.py
```

## 6. Run the Recommender

```bash
python main.py <steamid64>
```

---

# 📄 License

MIT License  
© 2025 Ludex Project Authors
