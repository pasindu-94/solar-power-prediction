"""Configuration for solar power prediction models."""
MODEL_CONFIG = {
    "random_forest": {"n_estimators": 200, "max_depth": 15, "random_state": 42},
    "gradient_boosting": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "random_state": 42},
}
DATA_CONFIG = {"train_split": 0.8, "validation_split": 0.1, "test_split": 0.1, "random_seed": 42}
FEATURE_COLUMNS = ["ambient_temperature", "module_temperature", "irradiation", "dc_power", "ac_power"]
