# An Examination of the Racial Differences in the Performance of Race Classifiers: Evidence from Television
Jenny Wang, Kyung Park, Eni Mustafaraj


# Race Classification - Ground Truth

**Data Files Used**
RaceEthnicityGroundTruth.csv

CleanRaceData.ipynb --> Goes through 

# Race Classification - Biography

**Data Files Used**
- final_sample_metadata.csv: raw data scraped from IMDb /bio pages
- final_sample_preprocessed_new.csv: cleaned bio text, preprocessed bio text

**Notebooks for BioRaceBERT**
BioRaceClassification_DataPreprocessing --> Preprocesses biography text and creates a csv file containing the original biography text (w/ newlines and formatting cleaned) and preprocessed biography text.

BioRaceClassification_Training.ipynb (GPU needed) --> Builds Distilbert-based BioRaceBERT classifier that takes in biography text and saves the model (trained using CoLab & Google Drive)

BioRaceClassification_Testing.ipynb (GPU recommended) --> Tests BioRaceBERT models

**Notebooks for Ablation Analysis**
BioAblationAnalysis --> Creates files of relabeled named entities, *uses `BioAblationTesting` to get test results (because models were hosted on Google Drive/CoLab*, then returns to this notebook to conduct analysis

BioAblationTesting (GPU recommended)--> Tests files created in BioAblationAnalysis using BioRaceBERT model

BioAblationBreakdown --> Looks at how many bios contained name, location, ethnicity information. figure out if viz is helpful to paper

BioExplainability.ipynb --> Creates graphs of P(race|race) vs. famousness and word count

ErrorAnalysis --> `FailureAnalysis.csv` 


