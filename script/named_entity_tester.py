import pandas as pd
import mapply
import numpy as np
import sys
sys.path.append("../script")
import text_preprocessing
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
from sklearn.metrics import classification_report
from tqdm import tqdm
######################################################################################
# This file is used to test the performance of our DistilBERT race classifier
# on our test set (781 people). Each csv file is created using a function in 
# named-entity-cleaner.py and contains "redacted" biography text.
#
# This document provides my step by step analysis of the race classifier's performance
# as different categories are redacted or kept in their original form.
# https://docs.google.com/document/d/1Tu0F2XWNG_bL57rZW6wIhjvmDaMIbhnursTWd5UFQwI/edit
# 
######################################################################################

###################################### TEST CSV FILES ####################################
def test_original_bio():
    '''
    0 NO ENTITIES
    '''
    in_file = "test_sample_metadata"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="mini_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_outfile.csv", index=None)

def test_ner_bio():
    '''
    1 ALL ENTITIES
    '''
    in_file = "test_sample_metadata_with_ner"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_outfile.csv", index=None)

def test_ner18_bio():
    '''
    2 ALL ENTITIES with 18 categories
    '''
    in_file = "test_sample_metadata_with_ner18"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_outfile.csv", index=None)

def test_ner_no_ethn_bio():
    '''
    3 NON-ETHNICITY ENTITY with 18 categories
    '''
    in_file = "test_sample_metadata_with_ner18"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_no_ethnicity_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_ethn_outfile.csv", index=None)

def test_ner_no_loc_bio():
    '''
    4 NON-LOCATION ENTITIES
    '''
    in_file = "test_sample_metadata_with_ner18"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_no_loc_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_loc_outfile.csv", index=None)

def test_ner_no_ppl_bio():
    '''
    5 NON-PERSON ENTITIES
    '''
    in_file = "test_sample_metadata_with_ner18"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_no_ppl_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_ppl_outfile.csv", index=None)

def test_ner_no_ethn_ppl_bio():
    '''
    6 NO ETHNICITY+PERSON ENTITIES
    '''
    in_file = "test_sample_metadata_with_ner18"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_no_ethn+ppl_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_ethn+ppl_outfile.csv", index=None)

def test_ner_no_ethn_loc_bio():
    '''
    7 NO ETHNICITY+LOCATION ENTITIES
    '''
    in_file = "test_sample_metadata_with_ner18"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_no_ethn+loc_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_ethn+loc_outfile.csv", index=None)

def test_ner_no_loc_ppl_bio():
    '''
    8 NO LOCATION+PEOPLE ENTITIES
    '''
    in_file = "test_sample_metadata_with_ner18"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_no_loc+ppl_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_loc+ppl_outfile.csv", index=None)

def test_ner_no_ppl_ethn_loc_bio():
    '''
    9 NO PEOPLE+ETHNICITY+LOCATION ENTITIES
    '''
    in_file = "test_sample_metadata_with_ner18"
    pred_df = predict_race_for(file_path=f"{main_dir}/data/{in_file}.csv", col_name="ner_no_ppl+ethn+loc_bio")
    pred_df.to_csv(f"{main_dir}/data/{in_file}_ppl+ethn+loc_outfile.csv", index=None)


############################################ HELPER FUNCTIONS ###########################################
def predict_one(raw_text):
    '''
    Params:
        raw_text: unprocessed biography text 
    Returns:
        prediction_value: one of {0, 1, 2, 3} which maps to {Asian, Black, Hispanic, White}
    '''
    text = ' '.join(text_preprocessing.preprocess(raw_text, lemmatization=True))
    predict_input = loaded_tokenizer.encode(text,
                                  truncation=True,
                                  padding=True,
                                  return_tensors="tf")

    output = loaded_model(predict_input)[0]
    prediction_value = tf.argmax(output, axis=1).numpy()[0]
    return prediction_value

def predict_race_for(file_path, col_name, should_eval=True):
    '''
    Predicts race label using `predict` function
    Returns:
        test_predictions: list of race label predictions
    '''
    test_df = pd.read_csv(file_path)
    test_df = test_df.replace(np.nan, "", regex=True)
    tqdm.pandas(desc="progress bar!")
    test_df["pred"] = test_df[col_name].progress_apply(predict_one)

    pred_df = test_df[["name", "href", col_name, "label", "pred"]]

    # test_predictions = [predict_one(text) for text in tqdm(test_df[col_name])]
    # pred_df = pd.DataFrame({
    #                     "name": test_df["name"],
    #                     "href": test_df["href"],
    #                     "text": test_df[col_name],
    #                     "label": test_df["label"],
    #                     "pred": test_predictions
    #                                             })
    if should_eval:
        print(classification_report(pred_df["label"], pred_df["pred"]))
    return pred_df

############################################# DO HERE ###################################################

main_dir = ".."
mapply.init(n_workers=-1, chunk_size=1, max_chunks_per_worker=10, progressbar=True)

loaded_tokenizer = DistilBertTokenizerFast.from_pretrained(f"{main_dir}/model/distilbert")
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(f"{main_dir}/model/distilbert")

# 0 ORIGINAL
# test_original_bio()

## 1 All ENTITIES
# test_ner_bio()

## 2 ALL ENTITIES (18 CATEGORY MODEL)
# test_ner18_bio()

## 3 NON-ETHNICITY ENTITIES
# test_ner_no_ethn_bio()

## 4 NON-LOCATION ENTITIES
# test_ner_no_loc_bio()

# # 5 NON-PERSON ENTITIES
# test_ner_no_ppl_bio()

# # 6 NO ETHNICITY+PERSON ENTITIES
# test_ner_no_ethn_ppl_bio()

# # 7 NO ETHNICITY+LOCATION ENTITIES
# test_ner_no_ethn_loc_bio()

# # 8 NO LOCATION+PEOPLE ENTITIES
# test_ner_no_loc_ppl_bio()

# # 9 NO PEOPLE+ETHNICITY+LOCATION ENTITIES
# test_ner_no_ppl_ethn_loc_bio()