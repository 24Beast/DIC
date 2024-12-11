import sys
import os
import torchtext
torchtext.disable_torchtext_deprecation_warning()

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
            self.tokenizer = NLTKWordTokenizer().tokenize  # Set to NLTK's word_tokenize
        else:
            self.tokenizer = get_tokenizer(tokenizer, lang)
        self.stopwords = stopwords
        self.gender_words = gender_words
        self.gender_token = gender_token
        self.object_words = obj_words
        self.object_token = obj_token
        if glove_path:
            self.glove_model = self.load_glove_model(glove_path)
        else:
            self.glove_model = None

    @staticmethod
    def load_glove_model(glove_path):
        print("Loading GloVe embeddings...")
        return KeyedVectors.load_word2vec_format(glove_path, binary=False)

    def apply_tokenizer(
        self, text_obj: Union[list[str], pd.Series]
    ) -> Union[list[list[str]] | pd.Series]:
        if type(text_obj) == pd.Series:
            return text_obj.apply(self.tokenize)
        else:
            sentence_tokens = [self.tokenize(text) for text in text_obj]
            return sentence_tokens

    def build_vocab(
        self, text_obj: Union[list[str], pd.Series], special_tokens: list[str] = []
    ):
        vocab = build_vocab_from_iterator(
            iter(self.apply_tokenizer(text_obj)), specials=special_tokens
        )
        return vocab

    def tokenize(self, text: str) -> list["str"]:
        tokens = self.tokenizer(text)
        tokens = [token for token in tokens if token not in self.stopwords]
        return tokens

    def tokens_to_numbers(
        self, vocab, text_obj: Union[list[str], pd.Series], pad_value: int = 0
    ):
        sequence = numericalize_tokens_from_iterator(
            vocab, self.apply_tokenizer(text_obj)
        )
        token_ids = []
        for i in range(len(text_obj)):
            x = list(next(sequence))
            token_ids.append(x)
        padded_text = pad_sequence(
            [torch.tensor(x) for x in token_ids],
            batch_first=True,
            padding_value=pad_value,
        )
        return padded_text

    def maskWords(
        self,
        token_list: Union[list[str], pd.Series],
        mode: Literal["gender", "object"] = "gender",
        object_presence_df: pd.DataFrame = None,
        img_id: int = None,
    ) -> Union[list[str], pd.Series]:
        """
        Mask words based on the specified mode:
        - "gender": Masks gender words with self.gender_token.
        - "object": Masks object words with self.object_token if present in object_presence_df.
        """        
        masked_tokens = []
        for token in token_list:
            if mode == "gender" and token in self.gender_words:
                masked_tokens.append(self.gender_token)
            elif mode == "object" and token in self.object_words:
                # Check if the object is present in the object_presence_df for this img_id
                if object_presence_df is not None and img_id is not None:
                    if object_presence_df.loc[img_id, token] == 1:
                        masked_tokens.append(self.object_token)
                    else:
                        masked_tokens.append(token)
                else:
                    masked_tokens.append(token)
            else:
                raise ValueError("Expected only 'gender' or 'object' for mode")
        return masked_tokens

    def equalize_vocab(
        self, vocab_human, vocab_model, glove_model, similarity_threshold=0.5
):
        """
        Equalize vocabularies by substituting with GloVe embeddings where possible
        and replacing unmatched tokens with 'unk'.
        """
        def substitute_token(token, machine_corpus):
            if token in machine_corpus:
                return token  # Keep if it exists in the machine corpus
            if not self.glove_model or token not in glove_model:
                return 'unk'
            token_vec = torch.tensor(glove_model[token])

            corpus_tokens = list(machine_corpus)
            corpus_embeddings = np.array([glove_model[t] for t in corpus_tokens])
            corpus_embeddings = torch.tensor(corpus_embeddings)

            # Compute cosine similarity and find best match
            token_vec = token_vec.unsqueeze(0)  # Add batch dimension
            similarities = torch.nn.functional.cosine_similarity(token_vec, corpus_embeddings, dim=1)
            max_similarity, best_match_idx = torch.max(similarities, dim=0)
            if max_similarity >= similarity_threshold:
                return corpus_tokens[best_match_idx.item()]
            else:
                return 'unk'

        # Applying substitution to human and model vocabularies
        equalized_vocab_human = [
            substitute_token(token, vocab_model) for token in tqdm(vocab_human, desc="Equalizing Human Vocab")
        ]
        equalized_vocab_model = [
            substitute_token(token, vocab_human) for token in tqdm(vocab_model, desc="Equalizing Model Vocab")
        ]

        return equalized_vocab_human, equalized_vocab_model

# Command-line argument parser to choose tokenizer and substitution mode
def get_parser():
    parser = argparse.ArgumentParser(description="CaptionProcessor CLI")
    parser.add_argument("--tokenizer", default="nltk", choices=["nltk", "spacy"], help="Choose tokenizer: 'nltk' or 'spacy'")
    parser.add_argument("--mode", default="gender", choices=["gender", "object"], help="Choose masking mode: 'gender' or 'object'")
    parser.add_argument("--glove_path", required=True, help="Path to GloVe embeddings in word2vec format")
    parser.add_argument("--output_folder", default='output', help="Folder to save processed outputs")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Cosine similarity threshold for GloVe substitution")
    return parser

# Test
if __name__ == "__main__":
    args = get_parser().parse_args()

    HUMAN_ANN_PATH = "gender_obj_cap_mw_entries.pkl"
    MODEL_ANN_PATH = "gender_val_transformer_cap_mw_entries.pkl"
    GENDER_WORDS = []
    OBJ_WORDS = []
    GENDER_TOKEN = "GENDER"
    OBJ_TOKEN = "OBJ"

    data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
    human_ann, model_ann = data_obj.getData()
    object_presence_df = data_obj.get_object_presence_df()

    processor = CaptionProcessor(
        GENDER_WORDS, 
        OBJ_WORDS, 
        glove_path=args.glove_path, 
        tokenizer=args.tokenizer
    )

    vocab_human = processor.build_vocab(human_ann.caption)
    vocab_model = processor.build_vocab(model_ann.caption)

    #If you want to print sample equalized vocabulary
    equalized_human_vocab, equalized_model_vocab = processor.equalize_vocab(
        vocab_human, vocab_model, glove_model=None, similarity_threshold=args.similarity_threshold
    )

    print("Equalized Vocabulary:")
    print(equalized_human_vocab[:10], equalized_model_vocab[:10])
