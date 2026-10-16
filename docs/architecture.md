# Architecture

```mermaid
flowchart LR
  subgraph offline [Offline pipeline]
    A[Kaggle ratings] --> B[Clean and split by user]
    B --> C[(Parquet files)]
    C --> D[SVD training]
    D --> E[Item factors]
    D -.-> M[MLflow runs]
    C --> F[Popularity scores]
    S[Anime synopses] --> H[Text embeddings]
    H --> I[Nearest neighbour index]
  end

  subgraph serving [Serving]
    E --> R[Hybrid ranker]
    F --> R
    I --> R
    R --> API[FastAPI]
    API --> UI[Streamlit app]
    AL[AniList] -->|if unavailable| JK[Jikan]
    JK -->|if unavailable| LD[Local dataset]
    AL --> UI
    JK --> UI
    LD --> UI
    UI --> LOG[(SQLite event log)]
    LOG --> UI
  end
```

Left side runs once per retrain (`scripts/train_models.py`). Right side runs
on every request. Live sources only supply posters, scores and browse lists,
the ranking itself comes only from the trained signals.
