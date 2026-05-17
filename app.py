from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import lightgbm as lgb
import pickle, torch
import pandas as pd
import numpy as np
import torch.nn as nn

ARTIFACTS = './artifacts'

# ── Load artifacts ────────────────────────────────────────
user_enc   = pickle.load(open(f'{ARTIFACTS}/user_enc.pkl',  'rb'))
news_enc   = pickle.load(open(f'{ARTIFACTS}/news_enc.pkl',  'rb'))
cat_enc    = pickle.load(open(f'{ARTIFACTS}/cat_enc.pkl',   'rb'))
user_act   = pd.read_parquet(f'{ARTIFACTS}/user_activity.parquet')
news_stats = pd.read_parquet(f'{ARTIFACTS}/news_stats.parquet')
news_cat   = pd.read_parquet(f'{ARTIFACTS}/news_cat.parquet')
popularity = pd.read_parquet(f'{ARTIFACTS}/popularity.parquet')
config     = pickle.load(open(f'{ARTIFACTS}/model_config.pkl', 'rb'))
reranker   = lgb.Booster(model_file=f'{ARTIFACTS}/reranker.lgb')

user_lookup = {v: i for i, v in enumerate(user_enc.classes_)}
news_lookup = {v: i for i, v in enumerate(news_enc.classes_)}
top_news    = popularity.nlargest(500, 'pop_score')['news_id'].tolist()

# ── Two-tower model ───────────────────────────────────────
class TwoTower(nn.Module):
    def __init__(self, n_users, n_news, emb_dim=64):
        super().__init__()
        self.user_emb   = nn.Embedding(n_users, emb_dim)
        self.news_emb   = nn.Embedding(n_news,  emb_dim)
        self.user_tower = nn.Sequential(nn.Linear(emb_dim,128), nn.ReLU(), nn.Linear(128,64))
        self.news_tower = nn.Sequential(nn.Linear(emb_dim,128), nn.ReLU(), nn.Linear(128,64))
    def forward(self, u, n):
        return (self.user_tower(self.user_emb(u)) * self.news_tower(self.news_emb(n))).sum(dim=1)

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
tt_model = TwoTower(config['n_users'], config['n_news']).to(device)
tt_model.load_state_dict(torch.load(f'{ARTIFACTS}/two_tower.pt', map_location=device))
tt_model.eval()

# ── App ───────────────────────────────────────────────────
app = FastAPI()

@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_k: int = 5):
    candidates = top_news[:100]
    u_idx = user_lookup.get(user_id, 0)
    n_idx = [news_lookup.get(n, 0) for n in candidates]

    with torch.no_grad():
        u = torch.tensor([u_idx]*len(candidates), dtype=torch.long).to(device)
        n = torch.tensor(n_idx, dtype=torch.long).to(device)
        tt_scores = tt_model(u, n).cpu().numpy()

    u_row   = user_act[user_act['user_id'] == user_id]
    n_rows  = news_stats[news_stats['news_id'].isin(candidates)].set_index('news_id')
    nc_rows = news_cat[news_cat['news_id'].isin(candidates)].set_index('news_id')
    pop_row = popularity.set_index('news_id')

    rows = []
    for i, nid in enumerate(candidates):
        rows.append({
            'tt_score':          tt_scores[i],
            'pop_score':         pop_row.loc[nid,'pop_score'] if nid in pop_row.index else 0,
            'user_ctr':          float(u_row['user_ctr'].values[0]) if len(u_row) else 0,
            'user_total_clicks': float(u_row['user_total_clicks'].values[0]) if len(u_row) else 0,
            'news_ctr':          float(n_rows.loc[nid,'news_ctr']) if nid in n_rows.index else 0,
            'news_clicks':       float(n_rows.loc[nid,'news_clicks']) if nid in n_rows.index else 0,
            'news_impressions':  float(n_rows.loc[nid,'news_impressions']) if nid in n_rows.index else 0,
            'category_idx':      float(nc_rows.loc[nid,'category_idx']) if nid in nc_rows.index else 0,
        })

    features = pd.DataFrame(rows)[['tt_score','pop_score','user_ctr','user_total_clicks',
                                    'news_ctr','news_clicks','news_impressions','category_idx']]
    scores  = reranker.predict(features)
    top_idx = np.argsort(scores)[::-1][:top_k]

    return {
        "user_id": user_id,
        "recommendations": [candidates[i] for i in top_idx],
        "scores": [round(float(scores[i]), 4) for i in top_idx]
    }

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; }
  h1 { font-size: 22px; font-weight: 500; }
  p  { color: #666; font-size: 14px; margin-bottom: 24px; }
  input  { width: 100%; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 15px; box-sizing: border-box; margin-top: 8px; }
  button { margin-top: 10px; width: 100%; padding: 10px; background: #1a1a1a; color: white; border: none; border-radius: 8px; font-size: 15px; cursor: pointer; }
  button:hover { background: #333; }
  .card  { background: white; border: 1px solid #eee; border-radius: 10px; padding: 14px 18px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center; }
  .rank  { font-size: 12px; color: #999; }
  .nid   { font-size: 15px; font-weight: 500; }
  .score { font-size: 13px; color: #666; background: #f0f0f0; padding: 3px 10px; border-radius: 20px; }
  .label { font-size: 12px; color: #999; margin: 20px 0 6px; }
</style>
</head>
<body>
  <h1>Content Recommender</h1>
  <p>A personalized news recommendation system trained on 2.7M user interactions from the MIND dataset.
It uses a <b>two-tower neural network</b> to retrieve candidates and a <b>LightGBM re-ranker</b> to personalize results.
A/B tested against a popularity baseline — achieved <b>+38.8% CTR lift</b> (p&lt;0.000001) on 47k users.</p>

<div style="background:#f0f0f0;border-radius:8px;padding:12px 16px;font-size:13px;color:#444;margin-bottom:16px">
  <b>How to use:</b> Enter any User ID from the MIND dataset (e.g. U13740, U80234, U60458) and click Get Recommendations.
  The model returns the top-K news articles ranked by predicted click probability.
</div>
  <div class="label">User ID</div>
  <input id="userId" value="U13740" />
  <div class="label">Top K</div>
  <input id="topK" value="5" />
  <button onclick="recommend()">Get Recommendations</button>
  <div id="results"></div>
<script>
async function recommend() {
  const uid  = document.getElementById('userId').value.trim();
  const topk = document.getElementById('topK').value.trim() || 5;
  const out  = document.getElementById('results');
  out.innerHTML = '<p style="color:#999;margin-top:16px">Loading...</p>';
  const res  = await fetch(`/recommend/${uid}?top_k=${topk}`);
  const data = await res.json();
  out.innerHTML = `<div class="label">${data.recommendations.length} recommendations for <b>${data.user_id}</b></div>`;
  data.recommendations.forEach((nid,i) => {
    out.innerHTML += `<div class="card"><div><div class="rank">#${i+1}</div><div class="nid">${nid}</div></div><div class="score">${data.scores[i]}</div></div>`;
  });
}
</script>
</body>
</html>
"""