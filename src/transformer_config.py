# src/transformer_config.py

TRANSFORMER_MODELS = {
    "distilbert": {
        "name": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_length": 256,
        "epochs": 3
    },
    "bert": {
        "name": "bert-base-uncased",
        "learning_rate": 3e-5,
        "batch_size": 8,
        "max_length": 256,
        "epochs": 3
    },
    "roberta": {
        "name": "roberta-base",
        "learning_rate": 2e-5,
        "batch_size": 8,
        "max_length": 256,
        "epochs": 3
    },
}
