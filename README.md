# Models

A single home for every model/algorithm implemented across my Master's in Computer Science (Winter 2024 – Fall 2025), organized by ML paradigm rather than by course. Each subfolder holds the original assignment/project code, ported as-is from the course materials.

See [ROADMAP.md](ROADMAP.md) for the full model-to-course index and suggested next steps for each area.

## Structure

- `01-classical-ml/` — KNN, linear classifiers (SVM, Softmax, Perceptron, Logistic Regression), Naive Bayes, MIRA, Decision Trees, Regression (Linear/LASSO/Logistic)
- `02-deep-learning-cv/` — CNN architecture zoo (14 architectures), image classification pipeline, YOLO object detection, RNN for time series
- `03-graph-network/` — Girvan-Newman community detection, web mining/PageRank-style project
- `04-probabilistic/` — EM algorithm for missing data imputation, GMM estimation, hyperparameter tuning + dimensionality reduction
- `05-unsupervised-dimred/` — PCA, NMF, Topic Modeling
- `06-nlp-llm-apps/` — PDF RAG chatbot (LLM-based document QA)

## Courses with no ML/model content

CS501 (IoT/embedded), CS557 (Database Design), CS567 (Algorithms theory), CS569 (Ethical Hacking), CS571 (Graph Theory proofs) — these were theory/systems courses with no models to port. Noted here for completeness of the "all courses" map.

## Related repos

Several of these projects also exist as standalone polished repos with full writeups: `cs596-deep-learning-rnn-project`, `cs509-missing-data-imputation-em`, `cs536-web-mining-structure-analysis`, `cs566-advanced-ai-classification`, `cs505-datamining-mountains-vs-beaches`, `cs503-data-visualization-bank-churn`. This repo is the cross-cutting map; those are the deep-dive versions.
