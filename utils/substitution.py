import os
import pickle
import nltk 
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import argparse
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Preload gender word lists
masculine = ['man','men','male','father','gentleman','boy','uncle','husband','actor','prince','waiter','he','his','him']
feminine = ['woman','women','female','mother','lady','girl','aunt','wife','actress','princess','waitress','she','her','hers']
gender_words = masculine + feminine

# Substitutions for gender words
genderword_substitutions = ['gender']

# Function to load .pkl file
def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Function to create machine_list from machine captions
def create_machine_dict(machine_pkl_path):
    machine_captions = load_pkl_file(machine_pkl_path)
    machine_captions = sorted(machine_captions, key=lambda x: x['img_id'])
    machine_dict = {entry['img_id']: entry['pred'] for entry in machine_captions}
    return machine_dict

# Function to create human_list from human captions
def create_human_list(human_pkl_path):
    human_captions = load_pkl_file(human_pkl_path)
    return [{'img_id': entry['img_id'], 'caption_list': entry['caption_list']} for entry in human_captions]

# Prepare machine corpus for substitution
def create_machine_corpus(machine_dict):
    machine_corpus = set()
    for caption in machine_dict.values():
        tokens = word_tokenize(caption.lower())
        machine_corpus.update(tokens)
    print(len(machine_corpus))
    return machine_corpus

# Load pre-trained GloVe embeddings (Word2Vec format)
def load_glove_model(glove_path):
    model = KeyedVectors.load_word2vec_format(glove_path, binary=False)
    return model

# Function to calculate cosine similarity between two words using GloVe
def word_similarity(word1, word2, glove_model):
    if word1 in glove_model and word2 in glove_model:
        word1_vec = glove_model[word1].reshape(1, -1)
        word2_vec = glove_model[word2].reshape(1, -1)
        return cosine_similarity(word1_vec, word2_vec)[0][0]
    return 0.0

# Ignore punctuation at the end of the sentence
def should_ignore_token(token):
    return token in ['.', ',', ' ']

# Contextual substitution function using GloVe embeddings for all words not found in the machine corpus
def contextual_substitute(human_list, machine_dict, machine_corpus, glove_model, mask_gender=True, similarity_threshold=0.5):
    modified_captions_grouped = []

    for entry in human_list:
        img_id = entry['img_id']
        modified_captions_for_img = []
        if img_id in machine_dict:
            for human_caption in entry['caption_list']:
                tokens = word_tokenize(human_caption.lower())
                new_caption = []
                
                for i, token in enumerate(tokens):
                    if should_ignore_token(token) and i == len(tokens) - 1:
                        new_caption.append(token)
                    elif token in gender_words and mask_gender:
                        new_caption.append('gender')
                    elif token in machine_corpus:
                        new_caption.append(token)
                    else:
                        # Perform GloVe substitution for all other words
                        substituted_token = substitute_with_glove(token, machine_corpus, glove_model, similarity_threshold)
                        new_caption.append(substituted_token)
                
                modified_captions_for_img.append(' '.join(new_caption))
        
        modified_captions_grouped.append({
            'img_id': img_id,
            'modified_caption_list': modified_captions_for_img
        })
    return modified_captions_grouped

# Function to substitute words using GloVe similarity
def substitute_with_glove(human_token, machine_corpus, glove_model, threshold=0.5):
    best_match = 'unk'
    max_similarity = 0.0

    for machine_token in machine_corpus:
        similarity = word_similarity(human_token, machine_token, glove_model)
        if similarity > max_similarity and similarity >= threshold:
            best_match = machine_token
            max_similarity = similarity

    return best_match

# Constant substitution function
def constant_substitute(human_list, machine_dict, machine_corpus, mask_gender=True):
    modified_captions_grouped = []

    for entry in human_list:
        img_id = entry['img_id']
        modified_captions_for_img = []
        if img_id in machine_dict:
            for human_caption in entry['caption_list']:
                tokens = word_tokenize(human_caption.lower())
                new_caption = []
                
                for i, token in enumerate(tokens):
                    if should_ignore_token(token) and i == len(tokens) - 1:
                        new_caption.append(token)
                    elif token in gender_words and mask_gender:
                        new_caption.append('gender')
                    elif token in machine_corpus:
                        new_caption.append(token)
                    else:
                        new_caption.append('unk')
                
                modified_captions_for_img.append(' '.join(new_caption))
        
        modified_captions_grouped.append({
            'img_id': img_id,
            'modified_caption_list': modified_captions_for_img
        })
    
    return modified_captions_grouped

# Common function for masking machine captions
def mask_machine_captions(machine_dict, mask_gender=True):
    masked_machine_dict = {}
    
    for img_id, caption in machine_dict.items():
        tokens = word_tokenize(caption.lower())
        new_caption = []
        
        for token in tokens:
            if token in gender_words and mask_gender:
                new_caption.append('gender')
            else:
                new_caption.append(token)
        
        masked_machine_dict[img_id] = ' '.join(new_caption)
    
    return masked_machine_dict

# Function to save modified human captions to a specified folder
def save_modified_captions_to_file(modified_human_captions_grouped, output_folder, output_file):
    output_path = os.path.join(output_folder, output_file)
    os.makedirs(output_folder, exist_ok=True)

    with open(output_path, 'w') as f:
        for entry in modified_human_captions_grouped:
            img_id = entry['img_id']
            for caption in entry['modified_caption_list']:
                f.write(f"{img_id}: {caption}\n")
    print(f"Modified human captions saved to {output_path}")

# Main function to handle both contextual and constant substitutions
def process_corpora_by_img_id(human_list, machine_dict, glove_model=None, similarity_threshold=0.5, mode="contextual"):
    machine_corpus = create_machine_corpus(machine_dict)

    if mode == "contextual":
        modified_human_captions_grouped = contextual_substitute(human_list, machine_dict, machine_corpus, glove_model, similarity_threshold=similarity_threshold)
    elif mode == "constant":
        modified_human_captions_grouped = constant_substitute(human_list, machine_dict, machine_corpus)
    
    masked_machine_dict = mask_machine_captions(machine_dict)
    return modified_human_captions_grouped, masked_machine_dict

# Command-line argument parser
#Example run: python substitution.py --mode contextual --output_folder /folder/path/ --output_file file/path/ 
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)
    parser.add_argument("--mode", default='contextual', choices=['contextual', 'constant'], help='Choose substitution mode: "contextual" or "constant".')
    parser.add_argument("--output_folder", default='output', type=str, help='Folder to save modified human captions')
    parser.add_argument("--output_file", default='modified_human_captions.txt', type=str, help='File to save modified human captions')
    return parser

def main(args):
    machine_pkl_path = 'gender_val_fc_cap_mw_entries.pkl'
    human_pkl_path = 'gender_obj_cap_mw_entries.pkl'
    glove_path = 'glove.6B.100d.word2vec.txt'

    machine_dict = create_machine_dict(machine_pkl_path)
    human_list = create_human_list(human_pkl_path)
    glove_model = load_glove_model(glove_path) if args.mode == "contextual" else None

    modified_human_list_grouped, masked_machine_dict = process_corpora_by_img_id(
        human_list, machine_dict, glove_model, similarity_threshold=0.5, mode=args.mode
    )

    # Save the modified human captions to a file in the specified folder
    save_modified_captions_to_file(modified_human_list_grouped, args.output_folder, args.output_file)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)