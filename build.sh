#!/bin/bash

echo "Installing NLTK and spaCy models..."
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet
python -m spacy download en_core_web_sm