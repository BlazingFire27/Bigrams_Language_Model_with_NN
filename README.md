# Implementation of a BiGram Language Model using Neural Networks
This repository contains a LLM project focused on a neural network predicting the next best word given a previous word of context_size = 1. 
The project leverages the PyTroch framework to create a BiGram Language model from scratch. 

## Data Source
The project utilizes the "Brown Dataset", which is publicly available at following URL: [https://www.nltk.org/nltk_data/](https://www.nltk.org/nltk_data/)
Alternatively, you can download it directly to the directory using below code:
```
import nltk
nltk.download('brown', download_dir='/path/to/nltk_data') #download_dir parameter is optional
```

## Project Overview
The core of the project is to practice how to implement NGrams Language Model from Scratch.
The Jupyter Notebook included in this repository provides a detailed walkthrough of the model implementation.

## Procedure to create the model
### Data Processing
The Brown dataset consists of 503 files with hundreds of words. To simplify, we only use sentences containing 10 words or fewer. These sentences are preprocessed by:

    Lowercasing the words

    Removing special characters using regular expressions

    Generating bi-grams (word pairs)

After cleaning and creating bi-grams, input-target pairs are generated for training the model.

### Dataset DataLoader part
A custom dataset is created to handle the processed input-output pairs.
The DataLoader is used to batch the data into 64-word sequences for training the model.

### Model Creation
The model is a simple feedforward neural network with:

    An Embedding Layer for encoding the input words

    A Fully Connected Layer (FC) to predict the target word

    A Softmax Activation function to convert logits to probabilities

### Training Loop
The training loop uses Cross-Entropy Loss to train the model.
The accuracy and loss are plotted over the course of 7 epochs.

## Results
Below are the plots for **Accuracy** and **Loss** over 7 training epochs:
### 📈 Accuracy Plot
![Accuracy](acc.png)

### 📉 Loss Plot
![Loss](loss.png)

## Note: 
The project code assumes that the required data files are present in the same directory as the Jupyter Notebook. Please follow the instructions above to acquire and prepare the data accordingly.
