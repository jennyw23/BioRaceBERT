import pandas as pd
import mapply
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

# ######################################################################################
# # This file is used to remove specific named entities for our ablation study.
# # 
# # This document provides my step by step analysis of the race classifier's performance
# # as different categories are redacted or kept in their original form.
# # https://docs.google.com/document/d/1Tu0F2XWNG_bL57rZW6wIhjvmDaMIbhnursTWd5UFQwI/edit
# #
# ######################################################################################

################################### MAIN FUNCTIONS ####################################

def clean_column_to_18_specific_entities(in_col: str, entities: set, reason: str=""):
    '''
    Removes NER tags specified by `entities` to redact labels
    '''
    if reason:
        print(f"TASK: Ablate flair entities {entities}.\nREASON: {reason}")

    out_col = in_col.mapply(lambda x: clean_specific_entities(x, entities))
    
    return out_col

def clean_to_18():
    '''
    2 ALL ENTITIES (18 CATEGORY MODEL)
    '''
    df = pd.read_csv(f"{main_dir}/data/test_sample_metadata.csv")
    df["ner_bio"] = df["mini_bio"].mapply(clean_all_entities)
    
    out_file = f"{main_dir}/data/test_sample_metadata_with_ner18.csv"
    df.to_csv(out_file, index=False)

def clean_to_4():
    '''
    1 ALL ENTITIES
    '''
    df = pd.read_csv(f"{main_dir}/data/test_sample_metadata.csv")

    tqdm.pandas(desc="progress bar!")
    df["ner_bio"] = df["mini_bio"].progress_apply(clean_all_entities)
    
    out_file = f"{main_dir}/data/test_sample_metadata_with_ner.csv"
    df.to_csv(out_file, index=False)

################################### HELPER FUNCTIONS ####################################

def clean_specific_entities(string, entities: set = {}):
    '''
    Removes specified named entities by looping through bio text in reverse order
    '''
    # initialize flair 'Sentence'
    sentence = Sentence(string)
    # predict NER tags
    tagger.predict(sentence)
    # convert flair Sentence to tokenized string
    tokens = sentence.to_tokenized_string().split(" ")

    # final clean string
    # clean_string = [substitute_named_entities(tokens, label) for label in reversed(sentence.get_labels()) if label.value in entities][-1] # fails when no entities

    for label in reversed(sentence.get_labels()):
        if label.value in entities:
            tokens = substitute_named_entities(tokens, label)    

    return " ".join(tokens)

def clean_all_entities(string):
    '''
    Removes all named entities (either all 4- or all 18-) by looping through bio text in reverse order, 
    substituting named entities for labels. 
    '''
    # initialize flair 'Sentence'
    sentence = Sentence(string)
    # predict NER tags
    tagger.predict(sentence)
    # convert flair Sentence to tokenized string
    tokens = sentence.to_tokenized_string().split(" ")
    # final clean string
    clean_string = [substitute_named_entities(tokens, label) for label in reversed(sentence.get_labels())][-1] # select final str
    return " ".join(clean_string)

def substitute_named_entities(tokens, label):
    '''
    Replaces each named entity with its corresponding label

    ['Hortensia', 'Santoveña'] at position [0:2] is replaced with "PER"
    ['Two', 'Mules', 'for', 'Sister', 'Sara'] at position [23:28] is replaced with "MISC"
    '''
    entity = label.value
    start, end = get_span_position(label)
    del tokens[start:end]
    tokens.insert(start,entity)
    return tokens

def get_span_position(label):
    '''
    "Span[0:2]: "Hortensia Santoveña" → PER (0.9905)" --> 0, 2
    "Span[64:65]: "Mexico" → LOC (0.9998)" --> 64, 65
    '''
    span = str(label).split(": ")[0][5:-1]
    start, end = span.split(":")
    start, end = int(start), int(end)

    return start, end

############################################# DO HERE ###################################################

### SET UP 
main_dir = ".."
mapply.init(n_workers=-1, chunk_size=1, max_chunks_per_worker=10, progressbar=True)

# tagger = SequenceTagger.load("flair/ner-english") # tagger used for 1 ALL ENTITIES
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast") # tagger used for 2 ALL ENTITIES (18 CATEGORY)

### Create initial `test_sample_metadata_with_ner` and `test_sample_metadata_with_ner18` ##########
# 1 All ENTITIES
# clean_to_4()

# 2 ALL ENTITIES (18 CATEGORY MODEL)
# clean_to_18()

### Cleans specific entities and combinations of entities
file = "test_sample_metadata_with_ner18"
entities = {
    "ethnicity": {'NORP', 'LANGUAGE'},
    "location": {'GPE', 'LOC'},
    "people": {'PERSON'},
    "ethnicity+location": {'NORP', 'LANGUAGE', 'GPE', 'LOC'},
    "ethnicity+people": {'NORP', 'LANGUAGE', 'PERSON'},
    "location+people": {'GPE', 'LOC', 'PERSON'},
    "ethnicity+location+people": {'NORP', 'LANGUAGE', 'GPE', 'LOC', 'PERSON'},
}

df = pd.read_csv(f"{main_dir}/data/{file}.csv")

# 3 NON-ETHNICITY ENTITIES
df["ner_no_ethn_bio"] = clean_column_to_18_specific_entities(in_col="mini_bio", entities=entities["ethnicity"], reason="remove specific ethnicity labels")
df.to_csv(f"{main_dir}/data/{file}.csv", index=False)

# 4 NON-LOCATION ENTITIES
df["ner_no_loc_bio"] = clean_column_to_18_specific_entities(in_col=df["mini_bio"], entities=entities["location"], reason="remove information about cities, states, and countries")
df.to_csv(f"{main_dir}/data/{file}.csv", index=False)

# 5 NON-PERSON ENTITIES
df["ner_no_ppl_bio"] = clean_column_to_18_specific_entities(in_col=df["mini_bio"], entities=entities["people"], reason="remove person names")
df.to_csv(f"{main_dir}/data/{file}.csv", index=False)

# 6 NO ETHNICITY AND NO PERSON ENTITIES
df["ner_no_ethn+ppl_bio"] = clean_column_to_18_specific_entities(in_col=df["mini_bio"], entities=entities["ethnicity+people"], reason="remove person and ethnicity names")
df.to_csv(f"{main_dir}/data/{file}.csv", index=False)

# 7 NO ETHNICITY AND NO LOCATION ENTITIES
df["ner_no_ethn+loc_bio"] = clean_column_to_18_specific_entities(in_col=df["mini_bio"], entities=entities["ethnicity+location"], reason="remove ethnicity and location names")
df.to_csv(f"{main_dir}/data/{file}.csv", index=False)

# 8 NO LOCATION AND NO PERSON ENTITIES
df["ner_no_loc+ppl_bio"] = clean_column_to_18_specific_entities(in_col=df["mini_bio"], entities=entities["location+people"], reason="remove person and location names")
df.to_csv(f"{main_dir}/data/{file}.csv", index=False)

# 9 NO ETHNICITY AND NO LOCATION AND NO PERSON ENTITIES
df["ner_no_ppl+ethn+loc_bio"] = clean_column_to_18_specific_entities(in_col=df["mini_bio"], entities=entities["ethnicity+location+people"], reason="remove person,ethnicity, and location names")
df.to_csv(f"{main_dir}/data/{file}.csv", index=False)
