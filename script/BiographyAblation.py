import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
import spacy
import re

#################################################################################
# Module to support BioAblationAnalysis notebook
# ###############################################################################


class FlairHelper:
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")

    entities = {
        "ethnicity": {'NORP', 'LANGUAGE'},
        "location": {'GPE', 'LOC'},
        "people": {'PERSON'},
        "ethnicity+location": {'NORP', 'LANGUAGE', 'GPE', 'LOC'},
        "ethnicity+people": {'NORP', 'LANGUAGE', 'PERSON'},
        "location+people": {'GPE', 'LOC', 'PERSON'},
        "ethnicity+location+people": {'NORP', 'LANGUAGE', 'GPE', 'LOC', 'PERSON'},
    }

    def get_name_set(self, name):
        '''
        e.g. Ang Lee --> [Ang Lee, Ang, Lee]
        Brian M. Metcalf --> [Brian M. Metcalf, Brian, M., Metcalf]
        '''
        name_set = [name]
        try:
            components = name.split()
            for i in components:
                name_set.append(i)
            return name_set

        except:
            return name_set

    def label_name_as_keyword(self, string, name, keyword):
        '''
        Removes ONLY SPECIFIED ENTITY by replacing variations of entity in bio text with keyword

        e.g. keyword=PERSON, name_variations={"Ang Lee", "Ang", "Lee"}
        '''
        # replace entity with specified keyword
        name_variations = self.get_name_set(name)
        for var in name_variations:
            string = string.replace(var, keyword)

        # remove duplicates
        string = re.sub(r'\b(\w+)\s+\1\b', r'\1', string)

        return string

    def label_specific_entities(self, string, entities: set = {}):
        '''
        Replaces specified named entities by looping through bio text in reverse order
        '''
        # initialize flair 'Sentence'
        sentence = Sentence(string)
        # predict NER tags
        self.tagger.predict(sentence)
        # convert flair Sentence to tokenized string
        tokens = sentence.to_tokenized_string().split(" ")

        for label in reversed(sentence.get_labels()):
            if label.value in entities:
                tokens = self.substitute_named_entities(tokens, label)

        return " ".join(tokens)

    def substitute_named_entities(self, tokens, label):
        '''
        Replaces each named entity with its corresponding label

        ['Hortensia', 'Santoveña'] at position [0:2] is replaced with "PER"
        ['Two', 'Mules', 'for', 'Sister', 'Sara'] at position [23:28] is replaced with "MISC"
        '''
        entity = label.value
        start, end = self.get_span_position(label)
        del tokens[start:end]
        tokens.insert(start, entity)
        return tokens

    def get_span_position(self, label):
        '''
        "Span[0:2]: "Hortensia Santoveña" → PER (0.9905)" --> 0, 2
        "Span[64:65]: "Mexico" → LOC (0.9998)" --> 64, 65
        '''
        span = str(label).split(": ")[0][5:-1]
        start, end = span.split(":")
        start, end = int(start), int(end)

        return start, end

    ###### Spacy ############


class SpacyHelper:
    nlp = spacy.load('en_core_web_sm')

    entities = {
        "ethnicity": {'NORP', 'LANGUAGE'},
        "location": {'GPE', 'LOC'},
        "people": {'PERSON'},
        "ethnicity+location": {'NORP', 'LANGUAGE', 'GPE', 'LOC'},
        "ethnicity+people": {'NORP', 'LANGUAGE', 'PERSON'},
        "location+people": {'GPE', 'LOC', 'PERSON'},
        "ethnicity+location+people": {'NORP', 'LANGUAGE', 'GPE', 'LOC', 'PERSON'},
    }

    def label_specific_entities(self, string, entities: set = {}):
        '''
        Relabels specified named entities by looping through bio text in reverse order
        '''
        doc = self.nlp(string)
        modified_text = ""
        start_pos = 0

        for token in doc:
            if token.ent_type_:
                # Append the modified text with the generic category
                modified_text += doc.text[start_pos:token.idx] + \
                    token.ent_type_
                # Update the start position
                start_pos = token.idx + len(token.text)

        # # Append any remaining text after the last entity
        modified_text += doc.text[start_pos:]

        # Print the modified text
        return modified_text

    def label_entity_as_keyword(self, string, keyword, entity_variations: set):
        '''
        Removes ONLY SPECIFIED ENTITY by replacing variations of entity in bio text with keyword

        e.g. keyword=PERSON_NAME, entity_variations={"Ang Lee", "Ang", "Lee"}
        '''
        # replace entity with specified keyword
        for var in entity_variations:
            string = string.replace(var, keyword)

        # remove duplicates
        string = re.sub(r'\b(\w+)\s+\1\b', r'\1', string)

        return string