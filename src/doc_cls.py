import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_INPUT_CSV = "./examples/default_example.csv"
DEFAULT_OUTPUT_CSV = "./examples/default_result.csv"
DEFAULT_MODEL_PATH = "./models/model_20210315_150752/model.bin"
DEFAULT_TOKENIZER_PATH = "./models/tokenizer"
DEFAULT_DOC_COL = "promed_news_text"

import os

from google_drive_downloader import GoogleDriveDownloader as gdd


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(df_examples, max_seq_length, tokenizer, column="docs"):

    features = []
    for (ex_index, example) in df_examples.iterrows():
        # Replacing new lines with [SEP] tokens
        text_a = example[column].replace("\\n", "[SEP]")
        tokens_a = tokenizer.tokenize(text_a)

        tokens_b = None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=None,
            )
        )
    return features


def get_predict(df_examples, model, tokenizer, column="docs"):
    eval_features = convert_examples_to_features(df_examples, 512, tokenizer, column)
    unpadded_input_ids = [f.input_ids for f in eval_features]
    unpadded_input_mask = [f.input_mask for f in eval_features]
    unpadded_segment_ids = [f.segment_ids for f in eval_features]

    padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
    padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
    padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)

    eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=8)

    model.eval()

    predicted_labels, target_labels = list(), list()
    scores = []

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Predicting"):
        input_ids = input_ids.to(model.device)
        input_mask = input_mask.to(model.device)
        segment_ids = segment_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, input_mask, segment_ids)
            pooled_output = outputs[1]
            pooled_output = model.dropout(pooled_output)
            logits = model.classifier(pooled_output)
        scores.extend(F.sigmoid(logits).cpu().detach().numpy())
        predicted_labels.extend(F.sigmoid(logits).round().long().cpu().detach().numpy())
    scores = [f"{(pos / (pos + neg)):.4f}" for pos, neg in scores]
    predicted_labels = [np.argmin(i) for i in predicted_labels]
    predicted_labels = [True if i else False for i in predicted_labels]
    return predicted_labels, scores


class DocCls:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        if not os.path.exists(DEFAULT_MODEL_PATH):
            print("Downloading pre-trained models...")
            os.makedirs("models", exist_ok=True)
            gdd.download_file_from_google_drive(
                file_id="1CRHtXiTK1SIbSFeRuEgdp7T1fEUpAcS8",
                dest_path=DEFAULT_MODEL_PATH,
            )

        print(f"Loading model from: {model_path}")
        self.model = torch.load(model_path, map_location=torch.device("cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH)

    def predict_csv(self, input_csv, output_csv, doc_col_name=DEFAULT_DOC_COL):
        print(f"Reading documents from {input_csv}")
        df = pd.read_csv(input_csv)
        predictions, relevancies = get_predict(
            df, self.model, self.tokenizer, column=doc_col_name
        )
        df["predictions"] = predictions
        df["relevancies"] = relevancies
        df.to_csv(output_csv)
        print(f"The result file is saved to: {output_csv}")

    def predict_text(self, text):
        if not isinstance(text, list):
            text = [text]
        df = pd.DataFrame({"docs": text})
        predictions, relevancies = get_predict(
            df, self.model, self.tokenizer, column="docs"
        )
        return predictions, relevancies

