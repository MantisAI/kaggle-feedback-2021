# kaggle-feedback-2021

## Notebooks
[The main simple longformer training notebook](https://colab.research.google.com/drive/1DB7CETQ6xMy6O_4ftVq3Vz_szC_vKW1R#scrollTo=LTNAmqPDSWLT).

[Notebook that generates more data by back-translation](https://colab.research.google.com/drive/1Qe5L9KhrOI329bRer2VbewT3T7nll-Nv#scrollTo=XE0PMZSkftkf)


### EDA on the data
More clean one: https://www.kaggle.com/dinowun/eda-simplified-feedback-prize
More comprehensive one: https://www.kaggle.com/erikbruin/nlp-on-student-writing-eda

## Local run

Create virtualenv
```
python -m venv venv # or your way of creating it
source venv/bin/activate
pip install --upgrade pip # cause why not
```

Install requirements
```
pip install -r requirements.txt
```

Get model
```
bash get_model.sh
```

Run train
```
python train.py
```
