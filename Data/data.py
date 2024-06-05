import torch
import torchtext
import datasets
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

class IMDB_data:
    def __init__(self, test_size: float = 0.25, max_length: int = 256, min_freq: int = 5):
        """
        IMDB_data class for loading and preprocessing the IMDB dataset.

        Parameters:
        - test_size: float, optional (default=0.25)
            Proportion of the dataset to include in the test split.
        - max_length: int, optional (default=256)
            Maximum length of the review to be processed. Reviews longer than this will be truncated.
        - min_freq: int, optional (default=5)
            The minimum frequency needed to include a token in the vocabulary. Tokens with a frequency
            lower than this will be ignored.
        """
        self.test_size = test_size
        self.max_length = max_length
        self.min_freq = min_freq

        # Load the IMDB dataset
        self.train_data, self.test_data = datasets.load_dataset('imdb', split=['train', 'test'])
        # Use the basic English tokenizer from torchtext
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

    def _clean_text(self, text):
        """
        Cleans the input text by performing the following operations:
        - Converts text to lowercase
        - Removes HTML tags
        - Removes URLs
        - Removes special characters, punctuation, and numbers
        - Removes extra whitespace
        """
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove special characters, punctuation, and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _clean_and_tokenize_input(self, x):
        """
        Cleans and tokenizes the input text.
        """
        cleaned_text = self._clean_text(x['text'])
        tokens = self.tokenizer(cleaned_text)[:self.max_length]
        length = len(tokens)
        return {'tokens': tokens, 'length': length}
    
    def _get_vocab(self, train_data):
        """
        Builds a vocabulary from the training data.
        """
        special_tokens = ["<unk>", "<pad>"]
        vocab = torchtext.vocab.build_vocab_from_iterator(
            train_data["tokens"],
            min_freq=self.min_freq,
            specials=special_tokens,
        )

        self.unk_index = vocab["<unk>"]
        self.pad_index = vocab["<pad>"]

        vocab.set_default_index(self.unk_index)
        return vocab
    
    def _numericalize_input(self, x, vocab):
        """
        Converts tokens to their respective indices using the vocabulary.
        """
        ids = vocab.lookup_indices(x["tokens"])
        return {"ids": ids}

    def get_data(self):
        """
        Processes the dataset by tokenizing, building the vocabulary, and numericalizing the tokens.
        
        Returns:
        - train_data: torch.utils.data.Dataset
            The processed training dataset.
        - valid_data: torch.utils.data.Dataset
            The processed validation dataset.
        - test_data: torch.utils.data.Dataset
            The processed test dataset.
        - vocab: torchtext.vocab.Vocab
            The produced vocab.
        """
        # Tokenize the data
        train_data = self.train_data.map(self._clean_and_tokenize_input)
        test_data = self.test_data.map(self._clean_and_tokenize_input)

        # Split the training data into training and validation sets
        train_valid_data = train_data.train_test_split(test_size=self.test_size)
        train_data = train_valid_data["train"]
        valid_data = train_valid_data["test"]

        # Build the vocabulary using the training data
        vocab = self._get_vocab(train_data=train_data)

        # Numericalize the tokenized text
        train_data = train_data.map(self._numericalize_input, fn_kwargs={"vocab": vocab})
        valid_data = valid_data.map(self._numericalize_input, fn_kwargs={"vocab": vocab})
        test_data = test_data.map(self._numericalize_input, fn_kwargs={"vocab": vocab})

        # Convert datasets to torch format with specified columns
        train_data = train_data.with_format(type="torch", columns=["ids", "label", "length"])
        valid_data = valid_data.with_format(type="torch", columns=["ids", "label", "length"])
        test_data = test_data.with_format(type="torch", columns=["ids", "label", "length"])

        return train_data, valid_data, test_data, vocab
    
    def visualize(self):
        """
        Visualizes word clouds for positive and negative reviews in the training data.
        """

        # Create a DataFrame from the training data
        data = {
            'text': self.train_data['text'],
            'label': self.train_data['label']
        }
        df = pd.DataFrame(data)

        # Clean the text
        df['text'] = df['text'].apply(self._clean_text)

        # Generate word clouds for positive and negative reviews
        for sentiment in [0, 1]:
            sentiment_text = df.loc[df['label'] == sentiment, 'text']
            text = ' '.join(sentiment_text.astype(str))
            title = 'Wordcloud for Negative Reviews' if sentiment == 0 else 'Wordcloud for Positive Reviews'
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

            plt.figure(figsize=(10, 5))
            plt.title(title)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
