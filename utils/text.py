import nltk
import torch
import pandas as pd
from typing import Union, Literal
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import numericalize_tokens_from_iterator


# Text Processor Class
class CaptionProcessor:

    def __init__(
        self,
        gender_words,
        obj_words,
        gender_token="<gender>",
        obj_token="<obj>",
        stopwords=[".", ",", " "],
        tokenizer="spacy",
        lang="en_core_web_sm",
    ) -> None:
        self.tokenizer = get_tokenizer(tokenizer, lang)
        self.stopwords = stopwords
        self.gender_words = gender_words
        self.gender_token = gender_token
        self.object_words = obj_words
        self.object_token = obj_token

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

    def replaceWords(
        self, token_list: list[str], mode: Literal["gender", "object"] = "gender"
    ) -> list[str]:
        if mode == "gender":
            return [
                token if not (token in self.gender_words) else self.gender_token
                for token in token_list
            ]
        elif mode == "object":
            return [
                token if not (token in self.object_words) else self.object_token
                for token in token_list
            ]
        else:
            raise ValueError("Expected only 'gender' or 'object' for mode")

    def equalize_vocab(self, model_captions, human_captions):
        human_vocab = self.build_vocab(human_captions)
        model_vocab = self.build_vocab(model_captions)

        pass


# Test
if __name__ == "__main__":
    from utils.datacreator import CaptionGenderDataset

    HUMAN_ANN_PATH = "bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl"
    MODEL_ANN_PATH = "bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl"
    GENDER_WORDS = []
    OBJ_WORDS = []
    GENDER_TOKEN = "GENDER"
    OBJ_TOKEN = "OBJ"

    data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
    human_ann, model_ann = data_obj.getData()

    processor = CaptionProcessor(GENDER_WORDS, OBJ_WORDS)

    vocab_human = processor.build_vocab(human_ann.caption)
    vocab_model = processor.build_vocab(model_ann.caption)
