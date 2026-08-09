from typing import Dict


DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "sst2": {
        "hf_name": "stanfordnlp/sst2",
        "eval_split": "validation",
        "text_column": "sentence",
        "label_column": "label",
        "label_map": {0: "negative", 1: "positive"},
        "prompt_template": (
            "You are a helpful sentiment classifier. "
            "Classify the sentiment of the following sentence as negative or positive.\n\n"
            "Sentence: {text}\n\nSentiment:"
        ),
        "default_target_label": 1,
    },
    "agnews": {
        "hf_name": "ag_news",
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "prompt_template": (
            "You are a helpful news classifier. "
            "Classify the following news article into one of four categories: World, Sports, Business, Sci/Tech.\n\n"
            "News: {text}\n\nLabel:"
        ),
        "default_target_label": 1,
    },
}
