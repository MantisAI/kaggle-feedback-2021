from ast import literal_eval
import os

from spacy.tokens import DocBin, Doc, Span
from spacy.util import get_words_and_spaces
import pandas as pd
import spacy
import typer


def get_spans_from_iob(iob_entities, tokens):
    spans = []
    start_char = None
    char_i = 0
    for i, ent in enumerate(iob_entities):
        ent_iob = ent[0]

        if ent_iob in ["B", "0"] and start_char:
            end_char = char_i
            span = (start_char, end_char, label)
            spans.append(span)

        if ent_iob == "B":
            ent_iob, ent_type = ent.split("-")
            start_char = char_i
            label = ent_type
        
        char_i += len(tokens[i]) + 1 # space 

    if start_char:
        end_char = char_i
        span = (start_char, end_char, label)
        spans.append(span)
    return spans

def convert_data_to_spacy(kaggle_data_path, spacy_data_path):
    data = pd.read_csv(os.path.expanduser(kaggle_data_path))
    data.entities = data.entities.apply(lambda x: literal_eval(x))

    nlp = spacy.blank("en")

    docbin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    for _, example in data.iterrows():
        #words = example.text.split()
        #doc = Doc(nlp.vocab, words=words, ents=example.entities)
        #doc = Doc(nlp.vocab, words=words)
        words, spaces = get_words_and_spaces(example.text.split(), example.text)
        doc = Doc(nlp.vocab, words=words, spaces=spaces)        
        spans = get_spans_from_iob(example.entities, example.text.split())

        #doc.ents = spans # for NER
        doc.ents = [doc.char_span(start_char, end_char, label=label) for (start_char, end_char, label) in spans[:1]]
        #doc.spans["ents"] = spans # for SpanCategorizer

        docbin.add(doc)
    docbin.to_disk(spacy_data_path)

if __name__ == "__main__":
    typer.run(convert_data_to_spacy)
