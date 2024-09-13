# 🎥 IMDB Sentiment Analysis

This repository contains the implementation of a sentiment analysis model using various Recurrent Neural Networks (RNN, LSTM, GRU) for the IMDB dataset. The project includes features like data preprocessing, model training, evaluation, visualization, and logging with TensorBoard.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Script](#running-the-script)
  - [Visualizing Data](#visualizing-data)
  - [Predicting Sentiment](#predicting-sentiment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to build and compare different neural network models for sentiment analysis on the IMDB dataset. The models are designed to classify movie reviews as either positive or negative.

## Features
- 🧠 **Multiple Model Architectures:** Supports RNN, LSTM, and GRU models.
- 🔠 **Pretrained Embeddings:** Utilizes GloVe pretrained embeddings.
- 🧹 **Data Preprocessing:** Includes text cleaning, tokenization, and vocabulary building.
- 📊 **Visualization:** Generates word clouds for positive and negative reviews.
- 📈 **TensorBoard Integration:** Logs training and evaluation metrics for visualization in TensorBoard.

## Installation
To set up the project, clone the repository and install the required packages:

```bash
git clone https://github.com/NimaVahdat/IMDB_Sentiment_Analysis.git
cd imdb-sentiment-analysis
```

Ensure you have the following packages installed:

* torch
* torchtext
* pandas
* matplotlib
* wordcloud
* tqdm
* tensorboard

## Usage

### Configuration
Create configuration files for your models (RNN, LSTM, GRU). Example configuration files are provided in the `config` directory. You can modify these files according to your needs.

### Running the Script
To train, evaluate, and predict sentiment using the models, run the `main.py` script:

```bash
python main.py
```

This script will:

* Load configurations for RNN, LSTM, and GRU models.
* Initialize the models.
* Visualize word clouds for positive and negative reviews.
* Count and print the number of parameters in each model.
* Train and evaluate each model.
* Test each model on the test dataset.
* Predict the sentiment of example reviews using each model.

### Visualizing Data
The visualize method in the script generates word clouds for positive and negative reviews in the training data:

```python
# Visualize word clouds
imdb_rnn.visualize()
```
<img src="https://raw.githubusercontent.com/visual/positive_word_cloud.png" alt="Positive Reviews Word Cloud" width="45%"/> <img src="https://raw.githubusercontent.com/NimaVahdat/visual/negative_word_cloud.png" width="45%"/>

This helps in understanding the most common words in positive and negative reviews.

### Predicting Sentiment
The script includes sentiment prediction for example reviews:

```python
# Example reviews for sentiment prediction
review1 = "It's a good movie..."
review2 = "Wow that was a painful 90 minutes..."

# Predict sentiment for the example reviews using all models
imdb_rnn.predict_sentiment(review1)
imdb_lstm.predict_sentiment(review1)
imdb_gru.predict_sentiment(review1)

imdb_rnn.predict_sentiment(review2)
imdb_lstm.predict_sentiment(review2)
imdb_gru.predict_sentiment(review2)
```

This will output the predicted sentiment class and probability for the provided custom reviews.


## Results

### Model Accuracy

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------------|---------------|
| RNN   | 0.893             | 0.832               | 0.825         |
| LSTM  | 0.896             | 0.883               | 0.878         |
| GRU   | 0.913             | 0.893               | 0.882         |

### Model Parameters

| Model | Number of Parameters      |
|-------|---------------------------|
| RNN   | 7,729,202                 |
| LSTM  | 12,580,802                |
| GRU   | 14,359,502                |

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.


