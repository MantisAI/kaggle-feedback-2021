from ast import literal_eval
import os

from spacy.tokens import DocBin, Doc
import pandas as pd
import spacy
import typer


def convert_data_to_spacy(kaggle_data_path, spacy_data_path):
    data = pd.read_csv(os.path.expanduser(kaggle_data_path))
    data.entities = data.entities.apply(lambda x: literal_eval(x))

    nlp = spacy.blank("en")

    docbin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    for _, example in data.iterrows():
        doc = Doc(nlp.vocab, words=example.text.split(), ents=example.entities)
        docbin.add(doc)
    docbin.to_disk(spacy_data_path)

if __name__ == "__main__":
    typer.run(convert_data_to_spacy)
