# 🎥 IMDB Sentiment Analysis

This repository contains the implementation of a sentiment analysis model using various Recurrent Neural Networks (RNN, LSTM, GRU) for the IMDB dataset. The project includes features like data preprocessing, model training, evaluation, visualization, and logging with TensorBoard.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
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
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
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

## Results
The results of the model training and evaluation, including accuracy and loss metrics, can be visualized using TensorBoard. Additionally, word clouds offer a visual representation of common terms in positive and negative reviews.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.


