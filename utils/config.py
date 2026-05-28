CATALOG = "workspace"
SCHEMA = "financial_sentiment"
TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"

EXPERIMENT_NAME = "/Users/jezapataf@eafit.edu.co/financial_sentiment_mlops"

REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.financial_sentiment_classifier"

VALID_LABELS = ["negative", "neutral", "positive"]
VALID_SPLITS = ["train", "validation", "test"]

MIN_MACRO_F1_TO_REGISTER = 0.55
MIN_DELTA_TO_PROMOTE = 0.001