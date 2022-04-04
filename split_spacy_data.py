import random

from spacy.tokens import DocBin
import spacy
import typer


def load_data(data_path):
    nlp = spacy.blank("en")
    docbin = DocBin().from_disk(data_path)
    data = list(docbin.get_docs(nlp.vocab))
    return data

def write_data(data, data_path):
    docbin = DocBin()
    for doc in data:
        docbin.add(doc)
    docbin.to_disk(data_path)

def split_spacy_data(data_path, train_data_path, test_data_path, test_size:float=0.2):
    data = load_data(data_path)

    random.shuffle(data)
    
    train_size = int(len(data)*(1-test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]

    write_data(train_data, train_data_path)
    write_data(test_data, test_data_path)

if __name__ == "__main__":
    typer.run(split_spacy_data)    
