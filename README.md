# 🎮 **Ludex — Hybrid Game Recommendation System (Steam)**

Ludex is a fully offline, hybrid **Content-Based + Collaborative Filtering** recommendation engine designed to provide **high-quality, diverse, and personalized Steam game suggestions**. Built as an academic project, its goal is to outperform Steam’s default discoverability algorithm by combining **metadata**, **player history**, and **diversity-aware ranking**.

---

## 🧩 **Overview**

Ludex works by blending three components:

1. **Content-Based Filtering (CBF)**  
   Uses metadata such as descriptions, tags, genres, and developers.

2. **Collaborative Filtering (CF)**  
   Finds similar users and recommends games based on behavior patterns.

3. **MMR Re-ranking (Maximal Marginal Relevance)**  
   Ensures recommended games are both relevant and diverse.

The project includes a crawling pipeline using the Steam Store API, with a fallback HTML parser based on Requests + BeautifulSoup (no Selenium or undetected-chromedriver required).

---

## 🏗️ **Project Structure**

```
Ludex/
│── CBF/                 
│── CF/                  
│── crawlers/            
│── main.py              
│── requirements.txt
│── LICENSE
│── .env.example
```

---

## 🕸️ **1. Crawlers & Metadata Extraction**

The crawler collects:
- Titles  
- Descriptions  
- Tags, genres  
- Developers, publishers  
- Screenshots & release info  

---

## 🧠 **2. Content-Based Filtering (CBF)**

### 🔹 TF-IDF  
Used on descriptions, tags, genres, and titles.

### 🔹 One-Hot Encoding  
Used for developers & publishers.

### 🔹 Weighted Embedding Blocks  
Combined into one multi-block feature vector for cosine similarity ranking.

---

## 👥 **3. Collaborative Filtering (CF)**

Analyzes:
- User playtime  
- Owned games  
- Behavioral similarity  

This enables discovery of games beyond metadata similarity.

---

## 🔗 **4. Hybrid Recommendation System**

Uses:
- Weighted CBF + CF blending  
- Optional SVD latent factors  
- Score normalization  
- Playtime scaling  

---

## 🎨 **5. MMR — Diversity Enhancement**

MMR ensures diversity by balancing:
- **Relevance**  
- **Variety**  

---

## ▶️ **6. Running the Project**

### Install:
```
pip install -r requirements.txt
```

### Run:
```
python main.py
```

---

# 📁 Data Files

- `data/raw/game_details.csv` — scraped metadata
- `data/raw/user_game_playtime_top20.csv` — interactions for CF
- `data/processed/recommender_matrix.npz` — CBF embeddings
- `data/processed/cf_als_model.pkl` — CF model
- `data/processed/cf_als_index.pkl` — CF mapping (item/user IDs)

---

## 🛠️ **Tech Stack**

- Python  
- scikit-learn  
- pandas  
- NumPy  
- BeautifulSoup  
- undetected-chromedriver  

---

# 📅 Roadmap (2025–2026)

### Short-term

- Add hybrid scoring module
- Build evaluation suite (Recall@K, Precision@K, MAP, NDCG)

### Medium-term

- Simple web UI with Steam login
- Real-time recommendation preview

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
