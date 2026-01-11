def get_config():
    return {
        "SEED": 86,
        "TRAIN_DATA_DIR_PATH": "./data/train",
        "TEST_DATA_DIR_PATH": "./data/test",
        "RESULTS_DIR_PATH": "./output",
        "BATCH_SIZE": 64,
        "NUM_WORKERS": 0,
        "N_EPOCHS": 30,
        "LEARNING_RATE": 1e-4,
        "LABEL_SMOOTHING": 0.05,
        "DEVICE": "cuda",
    }