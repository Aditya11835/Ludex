# 🚀 **Ludex: A Hybrid Game Recommendation System for Steam**

Ludex is a hybrid game recommendation system for Steam, built as a 5th-semester B.Tech project at IIIT Pune.
It aims to replace Steam’s “discoverability lottery” with **personalized, content-aware recommendations** built from:

- What you **play**, and how long you play it  
- What the **games actually are**, in terms of theme, genre, mechanics, and metadata  

> 🔷 **Status (2025):**  
> The full **Content-Based Filtering (CBF)** pipeline is implemented end-to-end:  
> **crawl → refine → scrape → feature extract → CBF recommend → CLI output**  
>  
> The **Collaborative Filtering (CF)** and **Hybrid (CF + CBF)** stages are designed but **not yet implemented**.

---

## 📌 Table of Contents

- [Why Ludex?](#why-ludex)  
- [System Architecture](#system-architecture)  
- [CBF Pipeline (Implemented)](#cbf-pipeline-implemented)  
  - [1. Steam Web Crawl](#1-steam-web-crawl)  
  - [2. Search Refiner (NSFW + Language Filter)](#2-search-refiner-nsfw--language-filter)  
  - [3. Detailed Metadata Scraper](#3-detailed-metadata-scraper)  
  - [4. Feature Extraction](#4-feature-extraction)  
  - [5. Content-Based Recommendation](#5-content-based-recommendation)  
- [Planned: Collaborative Filtering (CF)](#planned-collaborative-filtering-cf)  
- [Planned: Hybrid Scoring](#planned-hybrid-scoring)   
- [Design Rationale](#design-rationale)  
- [Future Work](#future-work)  
- [License](#license)

---

## 🎯 **Why Ludex?**

Steam hosts **100,000+ games**, yet most players use less than **10%** of their libraries.  
Recommendations often amplify popularity rather than **true similarity**.

Ludex focuses on:

- Precise **game similarity modeling**  
- A **true user taste profile**  
---

## 🧠 **System Architecture**

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
           ┌────────────────────────────────┐
           │ 3. Auto-Extend Catalogue (NEW) │
           │    - Detect missing appids     │
           │    - Crawl top 50 missing      │
           │    - Append and rebuild        │
           └────────────────────────────────┘
                           ▼
        recommender_matrix.npz (TF-IDF)
                           ▼
           ┌────────────────────────────────┐
           │ User Profile Builder (CBF)     │
           └────────────────────────────────┘
                           ▼
           Personalized Recommendations

---

# 🔄 **CBF Pipeline**

---

## **1. Initial Steam Crawl**

Collects thousands of appids from:

- Top sellers  
- Most played  
- Category pages  
- Search pages  

**Output →** `search_basic.csv`

---

## **2. NSFW + Language Refinement**

Removes:

- NSFW or adult titles  
- Non-Latin languages (CJK, Arabic, Hangul, Cyrillic…)  

**Output →** `search_basic_clean.csv`

---

## **3. Detailed Metadata Scraper**

Parallel scraper using undetected Chrome:

Extracts:

- Title  
- Genres  
- Tags  
- Description  
- Developers  
- Publishers  

**Output →** `game_details.csv`

---

## **4. Auto-Extend Game Catalogue (NEW)**

Triggered when running `main.py`:

1. Fetch user library  
2. Compare appids with existing catalogue  
3. Identify missing titles  
4. Crawl **top 50 most-played missing games** using the same high-quality scraper  
5. Append new rows to `game_details.csv`  
6. Rebuild:
   - `game_details.csv`
   - `recommender_matrix.npz`  

**Ensures no owned game is ever missing from the model.**

---

## **5. Feature Extraction & Training**

Using `recommender/model.py`, TF-IDF blocks:

| Block            | Weight |
|------------------|--------|
| Tags + Genres    | **0.9** |
| Title            | 0.3    |
| Description      | 0.15   |
| Developer (OHE)  | 0.2    |
| Publisher (OHE)  | 0.1    |

All blocks are concatenated → L2-normalized → saved as `recommender_matrix.npz`.

---

## **6. Content-Based Recommender**

### Build a user vector
- Filter games ≥ **MIN_PLAYTIME**  
- Weight by **log(1 + playtime)**  
- Weighted average of feature vectors  
- Normalize → **vᵤ**  

### Score all games
CBF(u, i) = vᵤ ⋅ f_i

Mask owned games → return **top-N recommendations**.

---



# 🔮 Planned: **Collaborative Filtering (CF)**

- Steam playtime → implicit interaction matrix  
- Item-item / user-user similarity  

CF will output: CF(u, i) → predicted preference

---

# 🧪 Planned: **Hybrid Scoring**

Combine CF and CBF per user: `Hybrid(u, i) = α · CF(u, i) + (1 − α) · CBF(u, i)`


Future enhancements:
- Per-user α based on profile strength  

---

# 🧩 Design Rationale

- **One embedding per game** → clean & consistent CBF  
- **One embedding per user** → robust representation  
- **Log-scaled playtime weighting** → prevents whales dominating  
- **Tags + genres TF-IDF** → strongest similarity signal  
- **Descriptions add nuance** without dominating  
- **Dev/pub OHE** → simple studio identity cues  
- **Everything L2-normalized** → cosine = dot product  
- Clean path for hybridization with CF models  

---

# 📅 Future Work

### Short-term
- Implement CF (implicit MF)  
- Create hybrid scoring pipeline  
- Add rank-based evaluation (MAP@K, NDCG@K)  

### Long-term
- Web UI for discovery feed   

---

# 📄 License

MIT License  
© 2025 Ludex Project Authors












