#!/bin/bash

echo "Downloading NLTK data..."
python -m nltk.downloader stopwords

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm
