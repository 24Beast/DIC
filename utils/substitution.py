import pickle
import nltk
import random
from nltk import word_tokenize
nltk.download('punkt')

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

# Preload gender word lists
masculine = ['man','men','male','father','gentleman','boy','uncle','husband','actor','prince','waiter','he','his','him']
feminine = ['woman','women','female','mother','lady','girl','aunt','wife','actress','princess','waitress','she','her','hers']
gender_words = masculine + feminine
genderword_substitutions = ['genderword1', 'genderword2', 'genderword3', 'genderword4', 'genderword5', 'genderword6', 'genderword7']

# Function to determine if a token should be substituted based on its presence in the machine caption
def should_substitute(token, machine_caption_tokens):
    return token in machine_caption_tokens

# Mask and substitute function for human annotations using machine-generated caption for the same img_id
def mask_and_substitute(human_list, machine_dict, mask_gender=True):

    modified_captions_grouped = []

    for entry in human_list:
        img_id = entry['img_id']
        modified_captions_for_img = []
        if img_id in machine_dict:
            machine_caption_tokens = word_tokenize(machine_dict[img_id].lower())

            for human_caption in entry['caption_list']:
                tokens = word_tokenize(human_caption.lower())
                new_caption = []
                
                for token in tokens:
                    if token in gender_words and mask_gender:
                        new_caption.append(random.choice(genderword_substitutions))

                    elif should_substitute(token, machine_caption_tokens):
                        new_caption.append(token)
                        
                    else:
                        new_caption.append(token)
                
                modified_captions_for_img.append(' '.join(new_caption))
        
        modified_captions_grouped.append({
            'img_id': img_id,
            'modified_caption_list': modified_captions_for_img
        })
    
    return modified_captions_grouped

# Mask function for machine captions (after substitution)
def mask_machine_captions(machine_dict, mask_gender=True):
    masked_machine_dict = {}
    
    for img_id, caption in machine_dict.items():
        tokens = word_tokenize(caption.lower())
        new_caption = []
        
        for token in tokens:
            if token in gender_words and mask_gender:
                new_caption.append('genderword')
            else:
                new_caption.append(token)
        
        masked_machine_dict[img_id] = ' '.join(new_caption)
    
    return masked_machine_dict

# Main function to process corpora using img_id-based matching
def process_corpora_by_img_id(human_list, machine_dict):
    modified_human_captions_grouped = mask_and_substitute(human_list, machine_dict)
    masked_machine_dict = mask_machine_captions(machine_dict)

    return modified_human_captions_grouped, masked_machine_dict

# Load the lists from pkl files (replace with your actual file paths)
machine_pkl_path = 'gender_val_att2in_cap_mw_entries.pkl'
human_pkl_path = 'gender_obj_cap_mw_entries.pkl'

# Create the machine caption dictionary and human caption list
machine_dict = create_machine_dict(machine_pkl_path)
human_list = create_human_list(human_pkl_path)

# Process corpora by img_id with token-based substitution based on machine captions
modified_human_list_grouped, masked_machine_dict = process_corpora_by_img_id(human_list, machine_dict)

# # Print the first 5 original and modified captions
for original, modified_group in zip(human_list[:5], modified_human_list_grouped[:5]):
    print(f"Original (img_id {original['img_id']}): {original['caption_list'][0]}")
    print(f"Modified (img_id {modified_group['img_id']}): {modified_group['modified_caption_list'][0]}")
    print()

# Example showing the first 5 masked machine captions
for img_id, masked_caption in list(masked_machine_dict.items())[:5]:
    print(f"img_id {img_id}: {masked_caption}")