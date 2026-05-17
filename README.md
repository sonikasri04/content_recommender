# Content Recommender 
A personalized news recommendation system trained on 2.7M user interactions from the MIND dataset. Replaces popularity-based ranking with a two-stage ML pipeline A/B tested with statistically significant CTR lift.

## Results
 
| Model | CTR@5 | Lift | p-value |
|---|---|---|---|
| Popularity Baseline | 0.1029 | — | — |
| Two-Tower Retrieval | 0.0951 | +6.4% | < 0.05 |
| **LightGBM Re-ranker** | **0.1428** | **+38.8%** | **< 0.000001** |
 
A/B tested on 47,000 users. Statistically significant at p < 0.000001.

## Architecture
 
```
User click history
       │
       ▼
┌─────────────────┐
│  Two-Tower Net  │  ← PyTorch, 64-dim embeddings
│  (Retrieval)    │    Top-100 candidates per user
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LightGBM       │  ← 7 features: tt_score, user_ctr,
│  Re-ranker      │    news_ctr, popularity, category...
└────────┬────────┘
         │
         ▼
    Top-K Results
```
 
**Stage 1 — Two-Tower Retrieval:** User and news encoders trained with BCEWithLogitsLoss on click data. Retrieves top-100 candidates via dot-product similarity.
 
**Stage 2 — LightGBM Re-ranker:** Re-ranks candidates using behavioral features (user CTR history, article popularity, category, two-tower score). This is where the big lift comes from.
 
**Stage 3 — FastAPI Serving:** REST API returning top-K recommendations in <100ms.

## Dataset
 
[MIND (Microsoft News Dataset)](https://msnews.github.io/) — MINDsmall split
- 50,000 users, 22,771 articles
- 5.8M training impressions, 2.7M dev impressions

## Stack
 
PyTorch · LightGBM · FastAPI · scikit-learn · pandas · scipy








