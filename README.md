# 🚀 **Ludex: A Hybrid Game Recommendation System for Steam**

Ludex is a hybrid game recommendation engine for Steam, built as a 5th-semester B.Tech project at IIIT Pune.  
It moves beyond Steam’s “discoverability lottery” by generating deeply personalized, content-aware, and **genre-diverse recommendations** using:

- What you **play**, and how much you play it  
- What the **games actually are** (tags, genres, mechanics, description, studio identity)  
- Your **anchor games** (high-playtime titles shaping your taste)  
- Global gameplay trends (future CF module)  

> ### 🔷 **Status (2025):**  
> ✔ End-to-end **CBF pipeline** implemented  
> ✔ **Auto-expanding game catalogue**  
> ✔ **Hybrid global + anchor + MMR diversification**  
>  
> ❗ Collaborative Filtering (CF) model planned  
> ❗ Full hybrid scoring (CF + CBF) is architected but CF is not yet implemented  

---

# 📌 Table of Contents

- [Why Ludex?](#why-ludex)
- [System Architecture](#system-architecture)
- [CBF Pipeline](#cbf-pipeline)
- [Hybrid Recommender Engine (NEW)](#hybrid-recommender-engine-new)
  - [Global Vector Scoring](#1-global-vector-scoring)
  - [Anchor-Based Personal Reinforcement](#2-anchor-based-personal-reinforcement)
  - [Blending Formula](#3-blending-formula)
  - [MMR Diversity Layer](#4-mmr-diversity-layer)
  - [Genre Coverage Heuristics](#5-genre-coverage-heuristics-optional)
- [Future: CF + CBF Hybrid](#future-cf--cbf-hybrid)
- [Design Rationale](#design-rationale)
- [Future Work](#future-work)
- [License](#license)

---

# 🎯 **Why Ludex?**

Steam hosts **100k+ games**, yet players typically explore < **15%** of their libraries.  
Most recommendation systems inflate popularity rather than learning what the user *really* likes.

Ludex focuses on:

- High-resolution **game embeddings**  
- A **true user taste vector**  
- Fair exposure for **minor genres** (visual novels, metroidvanias, strategy, indie subcultures)  
- Preventing genre domination (e.g., racing, FPS)  

---

# 🧠 **System Architecture**

            ┌─────────────────────┐
            │ search_basic.csv     │
            └──────────┬──────────┘
                       ▼
       ┌────────────────────────────────┐
       │ 1. NSFW + Language Refiner     │
       └────────────────────────────────┘
                       ▼
            search_basic_clean.csv
                       ▼
       ┌────────────────────────────────┐
       │ 2. Detailed Metadata Scraper   │
       └────────────────────────────────┘
                       ▼
            game_details.csv
                       ▼
       ┌──────────────────────────────────────────────────┐
       │ 3. Auto-Extend Catalogue (NEW)                    │
       │    - Detect missing owned games                   │
       │    - Crawl top 50 missing                         │
       │    - Append & rebuild TF-IDF                      │
       └──────────────────────────────────────────────────┘
                       ▼
       recommender_matrix.npz (L2-normalized embeddings)
                       ▼
       ┌────────────────────────────────┐
       │ Hybrid CBF Engine (NEW)        │
       └────────────────────────────────┘
                       ▼
       Personalized, Diverse Recommendations


---

# 🔄 **CBF Pipeline**

## **1. Initial Steam Crawl**
Collect appids from:
- Top sellers  
- Most played  
- Category pages  
- Search pages  

→ `search_basic.csv`

---

## **2. NSFW + Language Refinement**
Removes:
- NSFW titles  
- CJK, Cyrillic, Arabic, Hangul game titles  

→ `search_basic_clean.csv`

---

## **3. Detailed Metadata Scraper**

Parallel scraper with undetected Chrome retrieves:

- Title  
- Tags  
- Genres  
- Description  
- Developers  
- Publishers  

→ `game_details.csv`

---

## **4. Auto-Extend Catalogue (NEW)**

On every run of `main.py`:

1. Fetch owned games  
2. Compare with existing catalogue  
3. Identify missing appids  
4. Crawl top 50 missing titles  
5. Append & rebuild:

- `game_details.csv`  
- `recommender_matrix.npz`  

➡ **Guarantees no owned game is ever missing**, fixing the classic recommender blind-spot.

---

## **5. Feature Extraction & Model Encoding**

| Feature Block        | Weight |
|----------------------|--------|
| Tags + Genres        | **0.9** |
| Title (1–2 grams)    | 0.25    |
| Description          | 0.20    |
| Developer OHE        | 0.20    |
| Publisher OHE        | 0.10    |

All blocks concatenated → **L2-normalized per game** → final embedding `fᵢ`.

---

# 🌟 **Hybrid Recommender Engine (NEW)**  
A modern, multi-stage recommender similar to Spotify/YouTube/Steam Labs.

## **1. Global Vector Scoring**
Using the user content vector `vᵤ`:

`global_score[i] = vᵤ ⋅ fᵢ`


This captures long-term taste.

---

## **2. Anchor-Based Personal Reinforcement**

High-playtime games (“anchors”) define micro-tastes.

For each anchor game `a`:

`anchor_score[a][i] = fₐ ⋅ fᵢ`


Weighted by playtime importance:

`anchor_soft[i] = Σₐ wₐ ⋅ anchor_score[a][i]`

✔ Helps minority genres rise  
✔ Prevents a single genre from hijacking the model  

---

## **3. Blending Formula**

`combined_raw[i] = (1 − β) · global_score[i] + β · anchor_soft[i]`


---

## **4. MMR Diversity Layer**

Maximal Marginal Relevance:

`score_mmr(i) = λ combined_raw[i] − (1 − λ) max_{j∈S} (fᵢ ⋅ fⱼ)`

- λ ≈ **0.7**  
- Ensures **diverse** top-N recommendations  
- Prevents 15 racing games in your top 20  
- Includes titles from multiple genres, but weighted by your preference  

---

## **5. Genre Coverage Heuristics (Optional)**

To avoid deep genre starvation:

- If user has VN anchors but no VN appears in top 20 → **force-include best VN candidate**
- If user has significant strategy or horror signals → ensure representation  
- Allows minority genres to **compete fairly** without overpowering  

---

# 🔮 **Future: CF + CBF Hybrid**

### Planned CF Model (2025–2026)  
- Steam playtime → implicit feedback matrix    

Result: `CF(u, i) → collaborative affinity score`


### Final Hybrid

`Hybrid(u, i) = α · CF_norm(u, i) + (1 − α) · CBF_norm(u, i)`

Dynamic α:

- Strong CBF profile → CBF dominates  
- Weak CBF profile → CF dominates 

---

# 🧩 Design Rationale

- **Robust embeddings** → every game represented consistently  
- **Playtime-weighted anchors** → authentic personalization  
- **MMR** → protects against monotone lists  
- **Auto-expanding catalogue** → never missing your own games  
- **Genre fairness** → increases serendipity  
- **CF integration** → future-proof hybrid design  

---

# 📅 Future Work

### Short-term
- Add CF embeddings 
- Add evaluation suite (Precision@K, Recall@K, NDCG@K)

### Long-term
- Web UI + login with Steam

---

# 📄 License

MIT License  
© 2025 Ludex Project Authors



