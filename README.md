# AI/ML Assignment: Movie Audience Segmentation & Hit Prediction

## Problem Context

A movie streaming platform wants to:

1. **Segment movies** based on audience engagement patterns.
2. **Predict** whether a movie will be **"Hit"** or **"Not a Hit"**.

This helps in:

- Content recommendation  
- Marketing spend decisions  
- Acquisition strategy  

You are asked to build a solution using:

- **Unsupervised learning (Clustering)**  
- **Supervised learning (Classification)**  

---

## Dataset

The assignment provides a small dataset (**movies.csv**, 15 rows) with:

| Column           | Description                    |
|------------------|--------------------------------|
| movie_id         | Unique identifier              |
| avg_watch_time   | Average watch time (minutes)   |
| completion_rate  | Share of viewers who finished  |
| ratings_count    | Number of ratings              |
| avg_rating       | Average rating (e.g. 1–5)      |
| **hit**          | **Target:** 1 = Hit, 0 = Not a Hit |

**Sample (first 5 rows):**

```csv
movie_id,avg_watch_time,completion_rate,ratings_count,avg_rating,hit
1,45,0.60,1200,3.8,0
2,110,0.90,8500,4.6,1
3,60,0.65,2000,4.0,0
4,130,0.95,12000,4.8,1
5,40,0.55,900,3.6,0
...
15,150,0.98,18000,5.0,1
```

**Target variable:** `hit` → 1 = Hit, 0 = Not a Hit.

---

## Why I Used TMDB + Synthetic Data

The given dataset has **only 15 movies**. That is too small to:

- Reliably fit clustering (number of clusters, stability).  
- Train classification models without overfitting or unstable metrics.  
- Capture general patterns for recommendation or marketing.

**Approach taken:**

1. **Explored the TMDB dataset (Kaggle)**  
   - Used a larger real-world movie dataset to understand feature distributions and relationships.

2. **Preprocessed TMDB**  
   - Cleaned and aligned TMDB columns to the assignment schema (e.g. watch time, completion, ratings, hit-like target) where possible, and saved preprocessed files for experimentation.

3. **Generated synthetic data**  
   - Built a synthetic dataset (**synthetic_movies.csv**, 1200 rows) that **matches the patterns** of the original 15-row data (ranges, correlations, hit vs non-hit behavior).  
   - Added realistic noise and overlap so the task is not trivially separable and metrics are meaningful.

4. **Trained models on both original and synthetic**  
   - **Original (15 rows):** Full pipeline as per assignment (clustering + classification) for completeness.  
   - **Synthetic (1200 rows):** Main pipeline for stable clustering, robust classification, and realistic evaluation (no data leakage).

So: **small data → explore TMDB → create pattern-matched synthetic data → train and evaluate properly.**

---

## Assignment Tasks (What Was Implemented)

### Part A: Movie Segmentation (Clustering)

**Objective:** Group movies based on **viewer engagement** and **popularity**.

| # | Task | Done in |
|---|------|--------|
| 1 | Load and explore the dataset | All Task 1 notebooks |
| 2 | Select appropriate features for clustering | Engagement + Popularity (composite features) |
| 3 | Apply feature scaling | StandardScaler (and MinMax for composite features) |
| 4 | Apply a clustering algorithm | K-Means |
| 5 | Decide and justify number of clusters | Elbow + Silhouette (original: k=2; synthetic: optimal k) |
| 6 | Assign cluster labels to each movie | Yes |
| 7 | Interpret clusters (e.g. low engagement, cult classics, blockbusters) | Yes – Blockbusters, Cult Classics, Trending, Low Performers |
| 8 | Visualize clusters | Engagement vs Popularity scatter + centroids |

### Part B: Hit Prediction (Classification)

**Objective:** Predict whether a movie will be a Hit.

| # | Task | Done in |
|---|------|--------|
| 1 | Select features for prediction | 4 raw + engagement + popularity (+ cluster) = 6–7 features |
| 2 | Include cluster labels as input feature | Yes (cluster from Part A) |
| 3 | Split into train/test | Stratified, split **before** clustering/scaling (no leakage) |
| 4 | Implement at least one classification model | Logistic Regression, Random Forest (original); RF + XGBoost (synthetic/TMDB) |
| 5 | Evaluate with suitable metrics | Accuracy, confusion matrix, classification report, ROC-AUC |
| 6 | Train another classifier and compare | Yes (LR vs RF; RF vs XGBoost) |
| 7 | Select final model and justify | Based on test accuracy, AUC, CV, no leakage |
| 8 | Predict hit status for a new movie | Yes (example new movie in each notebook) |

---

## Project Structure

```
task assisment/
├── README.md
├── Dataset/
│   ├── original/
│   │   ├── movies.csv              # Assignment 15-row dataset
│   │   └── tmdb_5000_movies.csv     # Kaggle TMDB (exploration)
│   ├── preprocessed/                # TMDB converted/cleaned
│   └── synthetic/
│       ├── synthetic_movies.csv     # 1200 rows, pattern-matched
│       └── ...
├── Task1_Movie_Audience_Segmentation/
│   ├── movie_audience_segmentation.ipynb        # Part A – original 15 rows
│   └── movie_audience_segmentation_synthetic.ipynb  # Part A – synthetic 1200 rows
└── Task2_Part_Hit_Prediction/
    ├── preprocessing.ipynb          # TMDB load, clean, convert
    ├── synthetic_data_generation.ipynb  # Generate synthetic_movies.csv
    ├── hit_prediction_tmdb_dataset.ipynb  # Part B – TMDB-based
    ├── hit_prediction_original_data.ipynb    # Part B – original 15 rows
    └── hit_prediction_synthetic_data.ipynb   # Part B – synthetic 1200 rows
```

---

## Task 1: Movie Audience Segmentation (Clustering) – File Guide

### 1. `movie_audience_segmentation.ipynb` (Original 15 Movies)

**What it does:**

- Loads `Dataset/original/movies.csv` (15 rows).  
- Builds **engagement** (watch time + completion) and **popularity** (ratings count + avg rating, later weighted 0.7/0.3 in Task 2) from normalized features.  
- Scales engagement & popularity, runs **K-Means** (e.g. k=2 for 15 points).  
- Assigns cluster labels, interprets clusters (Blockbusters, Cult Classics, Trending, Low Performers).  
- Plots **Engagement vs Popularity** with clusters and centroids.  
- Outputs final table: movie_id, engagement, popularity, hit, cluster.

**Conclusion (quick read):**

- **Positive:** Pipeline is correct; clustering and interpretation align with assignment (engagement + popularity).  
- **Negative:** With only 15 points, cluster count and stability are limited; results are indicative, not statistically robust. Use this notebook to show the **same methodology** as on larger data.

---

### 2. `movie_audience_segmentation_synthetic.ipynb` (Synthetic 1200 Movies)

**What it does:**

- Loads `Dataset/synthetic/synthetic_movies.csv` (1200 rows).  
- Same feature design: **engagement** and **popularity** from normalized raw features.  
- **Elbow method** and **Silhouette score** used to choose **optimal k**.  
- K-Means with that k; cluster names (Blockbusters, Cult Classics, Trending, Low Performers) from engagement/popularity thresholds.  
- Visualization: Engagement vs Popularity by cluster; final summary table with counts, mean engagement, mean popularity, hit rate per cluster.

**Conclusion (quick read):**

- **Positive:** Clustering is stable; optimal k is data-driven; segment names are interpretable; hit rate by cluster gives actionable insight.  
- **Negative:** None major; this is the preferred Task 1 run for “real” conclusions.

---

## Task 2: Hit Prediction (Classification) – File Guide

### 1. `preprocessing.ipynb`

**What it does:**

- Loads TMDB (e.g. from Kaggle).  
- Handles missing values, basic outlier handling.  
- Converts TMDB columns to assignment-like schema (watch time, completion, ratings, rating, hit).  
- Saves cleaned/converted data under `Dataset/preprocessed/` for use in TMDB hit-prediction and synthetic generation.

**Conclusion (quick read):**

- **Positive:** Needed step to use TMDB and to design synthetic data; outputs are used by downstream notebooks.  
- **Negative:** TMDB does not have true “watch time” or “completion rate”; those are derived/approximated—acceptable for exploration and pattern study only.

---

### 2. `synthetic_data_generation.ipynb`

**What it does:**

- Uses original **movies.csv** (and optionally TMDB/preprocessed) to learn distributions and correlations.  
- Generates **1200 synthetic rows** with similar feature ranges and hit/non-hit behavior, with controlled noise and overlap.  
- Saves `Dataset/synthetic/synthetic_movies.csv` (and related files).  
- Ensures correlations with `hit` are realistic (e.g. in 0.3–0.7 range) so models are not trivially perfect.

**Conclusion (quick read):**

- **Positive:** Synthetic data matches assignment schema and original patterns; enables robust Task 1 and Task 2 without leakage; 50/50 or balanced hit rate is controllable.  
- **Negative:** Still synthetic; real deployment would require real platform data.

---

### 3. `hit_prediction_tmdb_dataset.ipynb`

**What it does:**

- Loads **preprocessed TMDB** data.  
- Builds features (including engagement, popularity, and cluster from clustering step).  
- Train/test split, scaling, and clustering applied without leakage.  
- Trains multiple classifiers (e.g. Random Forest, XGBoost, Voting/Stacking).  
- Compares models and justifies final choice; includes prediction for new movies and optional model save.

**Conclusion (quick read):**

- **Positive:** Shows the full pipeline on a larger real-world-style dataset; good for demonstrating scalability and model comparison.  
- **Negative:** Target and some features are derived from TMDB (e.g. revenue/budget), not true “hit” or watch metrics; use for methodology demonstration, not as final production setup.

---

### 4. `hit_prediction_original_data.ipynb` (Original 15 Movies)

**What it does:**

- Loads `movies.csv` (15 rows).  
- Creates engagement & popularity (weighted: popularity = 0.7×ratings_count + 0.3×avg_rating).  
- Uses **all 6 features** (avg_watch_time, completion_rate, ratings_count, avg_rating, engagement, popularity) + **cluster** (7 inputs).  
- **Split first**, then fit scaler and K-Means on **training only**; transform test; then scale for classification (no leakage).  
- Trains Logistic Regression and Random Forest; cross-validation via **Pipeline** (scaler + model).  
- GridSearch for hyperparameters; final evaluation and **new movie prediction** with 6 features + cluster.

**Conclusion (quick read):**

- **Positive:** Implements assignment exactly on given data; pipeline is correct (split → cluster → scale → train); deliverables satisfied.  
- **Negative:** With 15 rows, test set is tiny (e.g. 3 samples); metrics like 100% accuracy are not meaningful—expected for such small data. Use this to show **correct methodology**, not to claim model quality.

---

### 5. `hit_prediction_synthetic_data.ipynb` (Synthetic 1200 Movies)

**What it does:**

- Loads `Dataset/synthetic/synthetic_movies.csv`.  
- Same feature set: 6 features + cluster (7 inputs); popularity = 0.7×ratings_count + 0.3×avg_rating.  
- **Split first**; clustering and scaling fitted on **training only** (no leakage).  
- **Optimal k** from Elbow + Silhouette on training data.  
- Trains Random Forest and XGBoost; compares CV and test accuracy, ROC-AUC.  
- Selects best model (e.g. RF or XGBoost); feature importance; **new movie prediction**; optional evaluation on original 15-row data as a separate test set.

**Conclusion (quick read):**

- **Positive:** Realistic metrics (e.g. ~89% test accuracy, high AUC); no data leakage; proper CV and model selection; best notebook to show **actual performance** and justification.  
- **Negative:** None major; this is the main Task 2 implementation for conclusions.

---

## Summary Table: Where to Read Conclusions Without Opening Every File

| File | Task | Conclusion (short) |
|------|------|--------------------|
| **Task 1** | | |
| `movie_audience_segmentation.ipynb` | Clustering (original) | Methodology correct; 15 points → limited robustness. **Indicative only.** |
| `movie_audience_segmentation_synthetic.ipynb` | Clustering (synthetic) | Stable, optimal k, clear segments. **Use for clustering conclusions.** |
| **Task 2** | | |
| `preprocessing.ipynb` | TMDB clean/convert | Needed for TMDB + synthetic; derived schema. **Supporting.** |
| `synthetic_data_generation.ipynb` | Synthetic data | Pattern-matched, realistic; enables robust ML. **Positive.** |
| `hit_prediction_tmdb_dataset.ipynb` | Classification (TMDB) | Full pipeline on large data; methodology demo. **Positive; target is derived.** |
| `hit_prediction_original_data.ipynb` | Classification (15 rows) | Correct pipeline, no leakage; tiny test set → metrics not meaningful. **Methodology only.** |
| `hit_prediction_synthetic_data.ipynb` | Classification (1200 rows) | Realistic accuracy/AUC, no leakage, proper CV. **Use for classification conclusions.** |

---

## Deliverables Checklist

| Deliverable | Location / Notes |
|-------------|------------------|
| **Source code** | This repo: Task1_* and Task2_* notebooks; clean, commented. |
| **Dataset** | `Dataset/original/movies.csv` (assignment); `Dataset/synthetic/synthetic_movies.csv` (generated). |
| **Outputs** | Logs, metrics, and plots are produced when running each notebook. |
| **Short report (1–2 pages)** | Use this README for: problem understanding, clustering approach and insights, classification approach and evaluation; expand in a separate PDF if required. |

---

## How to Run

1. **Environment:** Python 3.x with pandas, numpy, matplotlib, seaborn, scikit-learn; for Task 2 synthetic/TMDB: xgboost (and optionally sdv/torch for synthetic generation).  
2. **Paths:** Run notebooks from project root or adjust paths to `Dataset/` and `../Dataset/` as in the notebooks.  
3. **Order (optional):**  
   - Task 1: `movie_audience_segmentation.ipynb` then `movie_audience_segmentation_synthetic.ipynb`.  
   - Task 2: Ensure `synthetic_movies.csv` exists (run `synthetic_data_generation.ipynb` if needed). Then run `hit_prediction_original_data.ipynb` and `hit_prediction_synthetic_data.ipynb`. TMDB notebooks need preprocessed data from `preprocessing.ipynb`.

---

## References

- Assignment handout: Movie Audience Segmentation & Hit Prediction (Part A clustering, Part B classification).  
- TMDB dataset: Kaggle (e.g. “TMDB 5000 Movie Dataset” or similar).  
- Synthetic data: Pattern-matched to `movies.csv`; generated in `Task2_Part_Hit_Prediction/synthetic_data_generation.ipynb`.
