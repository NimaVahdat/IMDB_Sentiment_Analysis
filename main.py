import torch
import numpy as np
from imdb_sentiment import IMDBSentimentAnalyzer
from utils import load_config_file

# Set the random seeds for reproducibility
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Load configurations for different models
rnn_config = load_config_file("./config/RNN_config.yaml")
lstm_config = load_config_file("./config/LSTM_config.yaml")
gru_config = load_config_file("./config/GRU_config.yaml")

# Initialize models
imdb_rnn = IMDBSentimentAnalyzer(rnn_config)
imdb_lstm = IMDBSentimentAnalyzer(lstm_config)
imdb_gru = IMDBSentimentAnalyzer(gru_config)

# Visualize word clouds for positive and negative reviews in the training data
imdb_rnn.visualize()

# Count the number of parameters in each model
rnn_num_params = imdb_rnn.count_parameters()
lstm_num_params = imdb_lstm.count_parameters()
gru_num_params = imdb_gru.count_parameters()

# Print the number of parameters for each model
print(f"Number of parameters in RNN model: {rnn_num_params:,}")
print(f"Number of parameters in LSTM model: {lstm_num_params:,}")
print(f"Number of parameters in GRU model: {gru_num_params:,}")

# Train and evaluate RNN model
print("Training and evaluating RNN model...")
imdb_rnn.train_and_evaluate()
imdb_rnn.test_model()

# Train and evaluate LSTM model
print("Training and evaluating LSTM model...")
imdb_lstm.train_and_evaluate()
imdb_lstm.test_model()

# Train and evaluate GRU model
print("Training and evaluating GRU model...")
imdb_gru.train_and_evaluate()
imdb_gru.test_model()

# Example reviews for sentiment prediction
review1 = (
    "It's a good movie (certainly Garfield's best) that shouldn't be taken seriously. "
    "You just have to have a good time with your family and enjoy it. There are some jokes "
    "that are more childish, but what can you do to it like that is the original children's series. "
    "There are also the odd jokes that the older ones can enjoy. The film leaves you with a beautiful message "
    "that many of us should take into account. It's not a great adaptation, because it takes several elements "
    "that aren't the same as the original series but that's okay, because you replace them with other elements "
    "that make the movie feel refreshing and different. Like any movie, not everything is good and this is no exception "
    "since there are some characters that I didn't like so much, such as the villain, which in my opinion is the weakest "
    "part of the film. It's a good movie to have a good time with the family."
)

review2 = (
    "Wow that was a painful 90 minutes. It felt a lot a longer. Even my son was bored to tears and he can turn an empty "
    "cardboard box into an adventure. Technically the cat looked like Garfield but the Garfield character was decidedly absent, "
    "and the entire domestic premise of the Garfield series. Jon was demoted to a side character, no Liz, no Nermal, pizza seemed "
    "to take precedence over lasagne! Read the comics, watch the 90s cartoons, maybe even the Bill Murray films. Well, not the Tail "
    "of Two Kitties. That was awful. Nearly as awful as this. 33 characters to go: It was poo, poop, poopy poo. Done."
)

# Predict sentiment for the example reviews using all models
print("\nPredicting sentiment for review1...")
print("RNN model prediction:")
imdb_rnn.predict_sentiment(review1)
print("LSTM model prediction:")
imdb_lstm.predict_sentiment(review1)
print("GRU model prediction:")
imdb_gru.predict_sentiment(review1)

print("\nPredicting sentiment for review2...")
print("RNN model prediction:")
imdb_rnn.predict_sentiment(review2)
print("LSTM model prediction:")
imdb_lstm.predict_sentiment(review2)
print("GRU model prediction:")
imdb_gru.predict_sentiment(review2)
