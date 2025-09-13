import sys
import os
import pandas as pd
import numpy as np
import faiss
import torch
import argparse
from typing import Union, Literal
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import numericalize_tokens_from_iterator
from utils.datacreator import CaptionGenderDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# torchtext.disable_torchtext_deprecation_warning()

count_total = 0
count_context = 0


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
        model_type="glove",
        bert_model="bert-base-uncased",  # for BERT
        device="cpu",
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
        self.model_type = model_type
        self.device = device
        if model_type == "glove" or model_type == "fasttext":
            self.glove_model = self.load_glove_model(glove_path) if glove_path else None
            self.bert_tokenizer = None
            self.bert_model = None
        elif model_type == "bert":
            print(f"Loading BERT model: {bert_model}...")
            try:
                from transformers import BertTokenizer, BertModel

                self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
                self.bert_model = BertModel.from_pretrained(bert_model).to(device)
            except ImportError:
                from transformers import AutoTokenizer, AutoModel

                self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
                self.bert_model = AutoModel.from_pretrained(bert_model).to(device)
            self.bert_model.eval()
            self.glove_model = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @staticmethod
    def load_glove_model(glove_path):
        return KeyedVectors.load_word2vec_format(glove_path, binary=False)

    def apply_tokenizer(
        self, text_obj: Union[list[str], pd.Series]
    ) -> Union[list[list[str]] | pd.Series]:
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
        self,
        string_list: Union[list[str], pd.Series],
        mode: Literal["gender", "object"] = "gender",
        object_presence_df: pd.DataFrame = None,
        img_id: int = None,
    ) -> Union[list[str], pd.Series]:
        """
        Mask words based on the specified mode:
        - "gender": Masks gender words with self.gender_token.
        - "object": Masks object words with self.object_token if present in object_presence_df.
        """
        if mode not in ["gender", "object"]:
            raise ValueError("Expected mode to be 'gender' or 'object'")
        masked_strings = []
        for string in string_list:
            masked_tokens = []
            token_list = self.tokenize(string)
            for token in token_list:
                if mode == "gender" and token in self.gender_words:
                    masked_tokens.append(self.gender_token)
                elif mode == "object" and token in self.object_words:
                    """
                    TODO : Previous logic, needs to be updated
                    if object_presence_df is not None and img_id is not None:
                        masked_tokens.append(
                            self.object_token
                            if object_presence_df.loc[img_id, token] == 1
                            else token
                        )
                    else:
                        masked_tokens.append(token)
                    """
                    masked_tokens.append(self.object_token)
                else:
                    masked_tokens.append(token)
            masked_strings.append(" ".join(masked_tokens))
        return masked_strings

    def get_all_token_embeddings_with_map(self, captions, batch_size=64):
        """
        Returns:
          embeddings: torch.Tensor (N_tokens, hidden_dim) on CPU
          mapping: list of dicts {caption_id, token_id, token}
        The tokens are the tokenizer.convert_ids_to_tokens results per caption (up to actual length).
        """
        assert self.model_type == "bert", "This method requires model_type == 'bert'"
        model = self.bert_model
        tokenizer = self.bert_tokenizer
        device = self.device

        model.eval()
        all_hidden = []
        mapping = []

        # Use DataLoader-like batching using indices to preserve order
        n = len(captions)
        for start in tqdm(range(0, n, batch_size), desc="Embedding captions (batched)"):
            end = min(n, start + batch_size)
            batch_texts = captions[start:end]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)  # (batch, seq_len, hidden_dim)

            hidden = outputs.last_hidden_state  # keep on GPU briefly
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # For each example in the batch, slice by attention_mask.sum and collect token embeddings
            for b_idx in range(hidden.size(0)):
                real_len = int(
                    attention_mask[b_idx].sum().item()
                )  # includes special tokens if present
                token_embeds = hidden[
                    b_idx, :real_len, :
                ].cpu()  # (seq_len, hidden_dim) -> move to CPU to save VRAM
                token_ids = input_ids[b_idx, :real_len].cpu().tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)

                all_hidden.append(token_embeds)  # list of (seq_len, hidden_dim) tensors

                # mapping contains captions' original index (start + b_idx), token position within caption, token string
                mapping.extend(
                    [
                        {"caption_id": start + b_idx, "token_id": t_idx, "token": tok}
                        for t_idx, tok in enumerate(tokens)
                    ]
                )

            # free GPU memory for this batch
            del inputs, outputs, hidden
            torch.cuda.empty_cache() if device.type.startswith("cuda") else None

        # concat along token dimension -> (total_tokens, hidden_dim)
        if len(all_hidden) == 0:
            return torch.empty((0, self.bert_model.config.hidden_size)), []
        embeddings = torch.cat(all_hidden, dim=0)  # on CPU
        return embeddings, mapping

    def build_faiss_index(self, embeddings, use_gpu_if_available=False):
        """
        embeddings: torch.Tensor (N, d) on CPU, dtype=float32 or convertible
        returns: index, is_gpu_bool, faiss_res (None if cpu)
        """
        if isinstance(embeddings, torch.Tensor):
            vecs = embeddings.numpy().astype("float32")
        else:
            vecs = np.asarray(embeddings, dtype="float32")
        d = vecs.shape[1]

        # Normalize for cosine similarity (we will use inner product on normalized vectors)
        faiss.normalize_L2(vecs)

        # CPU index
        cpu_index = faiss.IndexFlatIP(d)

        use_gpu = use_gpu_if_available and faiss.get_num_gpus() > 0
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                gpu_index.add(vecs)
                return gpu_index, True, res
            except Exception as e:
                # fallback to CPU index
                print("FAISS GPU index creation failed, falling back to CPU. Error:", e)
                cpu_index.add(vecs)
                return cpu_index, False, None
        else:
            cpu_index.add(vecs)
            return cpu_index, False, None

    # -------------------------
    # Replacement using index + delta threshold
    # -------------------------
    def replace_tokens_with_index(
        self,
        captions,
        index,
        token_map,
        batch_size=64,
        delta=0.5,
        desc="Replacing tokens",
    ):
        """
        Optimized version with batched BERT inference and vectorized FAISS operations.
        Expected 15-30x speedup over original implementation.
        """
        replaced_captions = []

        # Process in batches for maximum efficiency
        for start in tqdm(range(0, len(captions), batch_size), desc=desc):
            end = min(len(captions), start + batch_size)
            batch_captions = captions[start:end]

            # Batch tokenization (much faster than individual calls)
            inputs = self.bert_tokenizer(
                batch_captions, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            # Batched BERT inference
            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            # Move to CPU once for entire batch
            hidden_states = (
                outputs.last_hidden_state.cpu().numpy()
            )  # (batch, seq_len, hidden_dim)
            attention_masks = inputs["attention_mask"].cpu().numpy()
            input_ids = inputs["input_ids"].cpu().numpy()

            # Collect all embeddings for vectorized FAISS search
            all_embeddings = []
            caption_token_maps = []  # Track which embeddings belong to which caption

            for i, (hidden, attn_mask, ids) in enumerate(
                zip(hidden_states, attention_masks, input_ids)
            ):
                actual_len = int(attn_mask.sum())
                hidden_seq = hidden[:actual_len]  # (actual_len, hidden_dim)

                # Normalize embeddings
                faiss.normalize_L2(hidden_seq)
                all_embeddings.append(hidden_seq)

                # Store mapping info
                tokens = self.bert_tokenizer.convert_ids_to_tokens(ids[:actual_len])
                caption_token_maps.append(
                    {
                        "tokens": tokens,
                        "start_idx": len(all_embeddings) - 1,
                        "length": actual_len,
                    }
                )

            # Vectorized FAISS search (major speedup)
            if all_embeddings:
                all_embeddings_concat = np.vstack(all_embeddings)
                D, I = index.search(
                    all_embeddings_concat, 1
                )  # Single search for entire batch

                # Process results for each caption
                emb_idx = 0
                for cap_info in caption_token_maps:
                    tokens = cap_info["tokens"]
                    length = cap_info["length"]

                    # Get similarities and indices for this caption
                    cap_similarities = D[emb_idx : emb_idx + length, 0]
                    cap_indices = I[emb_idx : emb_idx + length, 0]

                    replaced_tokens = []
                    for tok, sim, idx in zip(tokens, cap_similarities, cap_indices):
                        # Skip special tokens
                        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                            continue

                        if sim >= delta:
                            replaced_tokens.append(token_map[idx]["token"])
                        else:
                            replaced_tokens.append("unk")

                    replaced_captions.append(" ".join(replaced_tokens))
                    emb_idx += length

        return replaced_captions

    def equalize_vocab(
        self,
        human_captions,
        model_captions,
        similarity_threshold=0.5,
        maskType="contextual",
        bidirectional=False,
    ):
        """
        Equalize captions using embeddings (GloVe or BERT).
        Preserves structure of tokenized captions.
        """

        human_tokens = [self.tokenize(caption) for caption in human_captions]
        model_tokens = [self.tokenize(caption) for caption in model_captions]

        # Flatten corpora into sets
        machine_corpus = set([token for tokens in model_tokens for token in tokens])
        human_corpus = set([token for tokens in human_tokens for token in tokens])

        global count_context
        global count_total

        count_total = 0
        count_context = 0

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
            if (max_similarity >= similarity_threshold) and maskType == "contextual":
                return corpus_tokens[best_match_idx.item()]
            else:
                return "unk"

        if self.model_type in ["glove", "fasttext"]:
            # Equalize human captions
            equalized_human = [
                " ".join(
                    [
                        substitute_token(tok, machine_corpus, " ".join(cap))
                        for tok in cap
                    ]
                )
                for cap in tqdm(human_tokens, desc="Equalizing Human Captions")
            ]

            print(f"{count_context=}/{count_total=} = {count_context/count_total}")
            # Equalize model captions if bidirectional
            if bidirectional:
                equalized_model = [
                    " ".join(
                        [
                            substitute_token(tok, human_corpus, " ".join(cap))
                            for tok in cap
                        ]
                    )
                    for cap in tqdm(model_tokens, desc="Equalizing Model Captions")
                ]
            else:
                equalized_model = [" ".join(cap) for cap in model_tokens]
        else:
            model_embs, model_map = self.get_all_token_embeddings_with_map(
                model_captions
            )

            if model_embs.numel() == 0:
                # nothing to index
                return human_captions, model_captions

            # 2) Build FAISS index (try GPU if requested)
            index, is_gpu, faiss_res = self.build_faiss_index(model_embs)

            # 3) Equalize model captions by replacing tokens
            equalized_human = self.replace_tokens_with_index(
                human_captions,
                index,
                model_map,
                delta=similarity_threshold,
                desc="Replacing Human Tokens",
            )

            # For human equalized (if bidirectional desired) we can also replace human tokens against model tokens:
            if bidirectional:
                # Build model token embeddings and index them, then replace human tokens similarly
                human_embs, human_map = self.get_all_token_embeddings_with_map(
                    human_captions
                )
                if human_embs.numel() == 0:
                    equalized_model = model_captions
                else:
                    h_index, h_is_gpu, _ = self.build_faiss_index(human_embs)
                    equalized_model = self.replace_tokens_with_index(
                        model_captions,
                        h_index,
                        human_map,
                        delta=similarity_threshold,
                        desc="Replacing Model Tokens",
                    )
            else:
                equalized_model = model_captions

        return equalized_human, equalized_model


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


def cmpVocab(vocab1, vocab2):
    set1 = set(vocab1.stoi.keys())
    set2 = set(vocab2.stoi.keys())

    common_tokens = set1 & set2
    only_in_vocab1 = set1 - set2
    only_in_vocab2 = set2 - set1
    print(
        f"Common_tokens : {len(common_tokens)}, vocab_1_exc: {len(only_in_vocab1)}, vocab_2_exc: {len(only_in_vocab2)}"
    )
