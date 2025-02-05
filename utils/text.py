import sys
import os
import torchtext
import pandas as pd
import numpy as np
import torch
import argparse
from typing import Union, Literal
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import numericalize_tokens_from_iterator
from utils.datacreator import CaptionGenderDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Text Processor Class
class CaptionProcessor:
    def __init__(
        self,
        gender_words,
        obj_words,
        glove_path=None,
        gender_token="gender",
        obj_token="obj",
        stopwords=[".", ",", " "],
        tokenizer="basic_english",
        lang="en",
    ) -> None:
        if tokenizer == "nltk":
            from nltk.tokenize import NLTKWordTokenizer

            self.tokenizer = NLTKWordTokenizer().tokenize
        else:
            self.tokenizer = get_tokenizer(tokenizer, lang)
        self.stopwords = stopwords
        self.gender_words = gender_words
        self.gender_token = gender_token
        self.object_words = obj_words
        self.object_token = obj_token
        self.glove_model = self.load_glove_model(glove_path) if glove_path else None

    @staticmethod
    def load_glove_model(glove_path):
        return KeyedVectors.load_word2vec_format(glove_path, binary=False)

    def apply_tokenizer(self, text_obj: Union[list[str], pd.Series]):
        if isinstance(text_obj, pd.Series):
            return text_obj.apply(self.tokenize)
        return [self.tokenize(text) for text in text_obj]

    def build_vocab(self, text_obj: Union[list[str], pd.Series]):
        vocab = build_vocab_from_iterator(self.apply_tokenizer(text_obj))
        return vocab

    def tokenize(self, text: str) -> list[str]:
        tokens = self.tokenizer(text)
        return [token for token in tokens if token not in self.stopwords]

    def tokens_to_numbers(
        self, vocab, text_obj: Union[list[str], pd.Series], pad_value: int = 0
    ):
        sequence = numericalize_tokens_from_iterator(
            vocab, self.apply_tokenizer(text_obj)
        )
        token_ids = [list(next(sequence)) for _ in range(len(text_obj))]
        return pad_sequence(
            [torch.tensor(x) for x in token_ids],
            batch_first=True,
            padding_value=pad_value,
        )

    def maskWords(
        self, token_list, mode="gender", object_presence_df=None, img_id=None
    ):
        if mode not in ["gender", "object"]:
            raise ValueError("Expected mode to be 'gender' or 'object'")
        masked_tokens = []
        for token in token_list:
            if mode == "gender" and token in self.gender_words:
                masked_tokens.append(self.gender_token)
            elif mode == "object" and token in self.object_words:
                if object_presence_df is not None and img_id is not None:
                    masked_tokens.append(
                        self.object_token
                        if object_presence_df.loc[img_id, token] == 1
                        else token
                    )
                else:
                    masked_tokens.append(token)
            else:
                masked_tokens.append(token)
        return masked_tokens

    def equalize_vocab(self, human_captions, model_captions, similarity_threshold=0.5):
        """
        Equalize vocabularies by substituting with GloVe embeddings where possible
        and replacing unmatched tokens with 'unk'.
        """

        def substitute_token(token, machine_corpus):
            token = token.lower()  # Ensure consistent casing
            if token in machine_corpus:
                return token  # Keep if it exists in the machine corpus
            if not self.glove_model or token not in self.glove_model:
                return "unk"
            token_vec = torch.tensor(self.glove_model[token])

            corpus_tokens = list(machine_corpus)
            corpus_embeddings = np.array(
                [self.glove_model[t] for t in corpus_tokens if t in self.glove_model]
            )  # Skip tokens not in GloVe
            if len(corpus_embeddings) == 0:  # If no embeddings are found
                return "unk"

            corpus_embeddings = torch.tensor(corpus_embeddings)

            # Compute cosine similarity and find best match
            token_vec = token_vec.unsqueeze(0)  # Add batch dimension
            similarities = torch.nn.functional.cosine_similarity(
                token_vec, corpus_embeddings, dim=1
            )
            max_similarity, best_match_idx = torch.max(similarities, dim=0)
            if max_similarity >= similarity_threshold:
                return corpus_tokens[best_match_idx.item()]
            else:
                return "unk"

        human_tokens = [self.tokenize(caption) for caption in human_captions]
        model_tokens = [self.tokenize(caption) for caption in model_captions]

        # Perform substitution while maintaining structure
        machine_corpus = set([token for tokens in model_tokens for token in tokens])

        equalized_human_captions = [
            " ".join([substitute_token(token, machine_corpus) for token in tokens])
            for tokens in tqdm(human_tokens, desc="Equalizing Human Vocab")
        ]
        equalized_model_captions = [" ".join(tokens) for tokens in model_tokens]

        return equalized_human_captions, equalized_model_captions


# CLI
def get_parser():
    parser = argparse.ArgumentParser(description="CaptionProcessor CLI")
    parser.add_argument("--tokenizer", default="nltk", choices=["nltk", "spacy"])
    parser.add_argument("--mode", default="gender", choices=["gender", "object"])
    parser.add_argument("--glove_path", required=True)
    parser.add_argument("--output_folder", default="output")
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    HUMAN_ANN_PATH = "./bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl"
    MODEL_ANN_PATH = "./bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl"
    data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
    human_ann, model_ann = data_obj.getData()

    processor = CaptionProcessor(
        gender_words=[],
        obj_words=[],
        glove_path=args.glove_path,
        tokenizer=args.tokenizer,
    )
    vocab_human = processor.build_vocab(human_ann.caption)
    vocab_model = processor.build_vocab(model_ann.caption)

    equalized_human_vocab, equalized_model_vocab = processor.equalize_vocab(
        vocab_human, vocab_model, similarity_threshold=args.similarity_threshold
    )

    print("Equalized Vocabulary:")
    print(equalized_human_vocab[:10], equalized_model_vocab[:10])
