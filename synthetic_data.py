#!/usr/bin/env python3
"""Generate a compact synthetic EB-NeRD-like dataset for testing gate_a_c.py."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

SEED = 20260919

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="synthetic_ebnerd")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    categories = ["news", "sport", "entertainment", "crime", "business"]
    topics_by_cat = {
        "news": ["politics", "world", "society"],
        "sport": ["football", "cycling", "tennis"],
        "entertainment": ["music", "tv", "celebrity"],
        "crime": ["police", "court", "investigation"],
        "business": ["markets", "companies", "economy"],
    }

    n_articles = 420
    article_ids = np.arange(100000, 100000 + n_articles)
    start = pd.Timestamp("2023-05-20T00:00:00Z")
    published = start + pd.to_timedelta(rng.uniform(0, 7 * 24, n_articles), unit="h")
    cats = rng.choice(categories, n_articles, p=[.25, .2, .2, .2, .15])
    topics = [[rng.choice(topics_by_cat[c])] for c in cats]
    premium = rng.random(n_articles) < 0.2

    articles = pd.DataFrame({
        "article_id": article_ids,
        "published_time": published,
        "premium": premium,
        "category": cats,
        "topics": topics,
        "title": [f"Synthetic article {i}" for i in article_ids],
    }).sort_values("published_time")

    # Category-structured embeddings so Gate C should beat random.
    dim = 12
    centroids = {c: rng.normal(size=dim) for c in categories}
    X = np.vstack([centroids[c] + rng.normal(scale=.25, size=dim) for c in cats])
    embeddings = pd.DataFrame({
        "article_id": article_ids,
        "embedding": [row.astype(float).tolist() for row in X],
    })

    n_users = 120
    user_ids = np.arange(1, n_users + 1)
    rows = []
    impression_id = 1

    article_pub_map = articles.set_index("article_id")["published_time"]
    article_cat_map = articles.set_index("article_id")["category"]

    for user in user_ids:
        device = rng.choice(["mobile", "desktop"], p=[.7, .3])
        subscriber = bool(rng.random() < .25)
        preferred = rng.choice(categories)
        for day in range(3):
            session = f"{user}-{day}"
            n_imp = int(rng.integers(2, 5))
            base_t = pd.Timestamp("2023-05-24T08:00:00Z") + pd.Timedelta(days=day) + pd.Timedelta(minutes=int(rng.integers(0, 600)))
            scroll = float(np.clip(rng.normal(60 if device == "desktop" else 45, 18), 0, 100))
            for pos in range(n_imp):
                t = base_t + pd.Timedelta(minutes=pos * int(rng.integers(4, 15)))
                eligible = articles.loc[(articles["published_time"] < t) & (articles["published_time"] >= t - pd.Timedelta("24h"))]
                if len(eligible) < 12:
                    eligible = articles.loc[articles["published_time"] < t].tail(80)

                # Simulate a publisher-side slate with recency plus mild category preference.
                scores = np.ones(len(eligible))
                scores += (eligible["category"].to_numpy() == preferred) * 1.5
                age_h = (t - eligible["published_time"]).dt.total_seconds().to_numpy() / 3600
                scores *= np.exp(-age_h / 36)
                scores = np.clip(scores, 1e-8, None)
                probs = scores / scores.sum()
                k = int(rng.integers(8, 14))
                chosen = rng.choice(eligible["article_id"].to_numpy(), size=min(k, len(eligible)), replace=False, p=probs)
                clicked = []
                familiar = [aid for aid in chosen if article_cat_map.loc[aid] == preferred]
                if familiar and rng.random() < .75:
                    clicked = [int(rng.choice(familiar))]
                elif len(chosen) and rng.random() < .3:
                    clicked = [int(rng.choice(chosen))]

                # Every 5th impression is an article-page context so the filter is tested.
                context_article = None if impression_id % 5 else int(rng.choice(chosen))

                rows.append({
                    "impression_id": impression_id,
                    "user_id": user,
                    "impression_time": t,
                    "article_id": context_article,
                    "session_id": session,
                    "device_type": device,
                    "scroll_percentage": scroll,
                    "article_ids_inview": [int(x) for x in chosen],
                    "article_ids_clicked": clicked,
                    "is_subscriber": subscriber,
                })
                impression_id += 1

    behaviors = pd.DataFrame(rows)

    if args.format == "parquet":
        articles.to_parquet(out / "articles.parquet", index=False)
        behaviors.to_parquet(out / "behaviors.parquet", index=False)
        embeddings.to_parquet(out / "embeddings.parquet", index=False)
    else:
        articles.to_csv(out / "articles.csv", index=False)
        behaviors.to_csv(out / "behaviors.csv", index=False)
        embeddings.to_csv(out / "embeddings.csv", index=False)

    print(f"Wrote synthetic {args.format} data to {out}")

if __name__ == "__main__":
    main()
