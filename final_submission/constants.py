from datetime import date

REFERENCE_DATE = date(2026, 7, 1)

POSITIVE_TITLES = {"ai", "ml", "machine learning", "data scientist", "nlp",
    "search", "retrieval", "ranking", "recommendation", "deep learning",
    "computer vision", "research engineer", "applied scientist",
    "backend engineer", "software engineer", "platform engineer"}

NEGATIVE_TITLES = {"marketing", "sales", "hr", "human resource", "support",
    "admin", "finance", "accountant", "civil", "mechanical", "graphic",
    "content writer", "project manager", "operations", "business analyst"}

RELEVANT_SKILLS = {"python", "pytorch", "tensorflow", "embeddings", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "nlp", "bert", "transformers",
    "sentence-transformers", "information retrieval", "ranking", "recommendation",
    "xgboost", "lightgbm", "deep learning", "machine learning", "mlops",
    "elasticsearch", "opensearch", "vector database", "langchain", "llm",
    "fine-tuning", "rag", "huggingface", "scikit-learn", "keras", "spacy",
    "opencv", "spark", "airflow", "docker", "kubernetes"}

STRONG_DESC_KEYWORDS = {"shipped", "deployed", "production", "ranking", "retrieval",
    "embeddings", "recommendation", "search", "vector", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "xgboost", "lightgbm", "ndcg", "mrr",
    "a/b test", "fine-tun", "transformer", "bert", "sentence-transformer",
    "pytorch", "tensorflow", "inference", "latency", "throughput", "scale",
    "pipeline", "feature engineering", "model training", "evaluation"}

SERVICES_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "mindtree", "ltimindtree",
    "persistent", "hexaware", "cyient", "zensar", "l&t infotech"}

BIG_TECH = {"microsoft", "google", "meta", "facebook", "amazon", "apple",
    "netflix", "uber", "oracle", "ibm", "intel", "nvidia", "salesforce", "adobe"}
