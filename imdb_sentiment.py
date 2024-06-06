import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import collections
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from Data import IMDB_data, get_data_loader
from Networks.networks import RNN, LSTM, GRU


class IMDBSentimentAnalyzer:
    def __init__(self, config):
        """
        Initialize the IMDBSentimentAnalyzer with the given configuration.

        Parameters:
        - config (dict): Configuration dictionary containing model, data, and training parameters.
        """
        self.config = config
        self._setup_model()
        self._setup_data()
        self._initialize_weights()
        self._setup_pretrained_embeddings()
        self._setup_training_tools()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def _setup_model(self):
        """Set up the model based on the configuration."""
        model_name = self.config["model"]
        model_config = self.config["model_config"]
        if model_name == "RNN":
            self.model = RNN(**model_config)
        elif model_name == "LSTM":
            self.model = LSTM(**model_config)
        elif model_name == "GRU":
            self.model = GRU(**model_config)
        else:
            raise ValueError("Model should be either 'RNN', 'LSTM', or 'GRU'!")

    def _setup_data(self):
        """Set up data loaders based on the configuration."""
        data_config = self.config["data_config"]
        batch_size = self.config["batch_size"]
        data = IMDB_data(**data_config)
        self.train_data, self.valid_data, self.test_data, self.vocab = data.get_data()
        pad_index = data.pad_index
        self.train_loader = get_data_loader(
            self.train_data, batch_size, pad_index, shuffle=True
        )
        self.valid_loader = get_data_loader(self.valid_data, batch_size, pad_index)
        self.test_loader = get_data_loader(self.test_data, batch_size, pad_index)

    def _initialize_weights(self):
        """Initialize model weights if specified in the configuration."""
        if self.config.get("initialize_weights", False):
            self.model.apply(self._init_weights)

    def _setup_pretrained_embeddings(self):
        """Set up pretrained embeddings if specified in the configuration."""
        vectors = torchtext.vocab.GloVe()
        pretrained_embeddings = vectors.get_vecs_by_tokens(self.vocab.get_itos())
        self.model.embedding.weight.data = pretrained_embeddings

    def _setup_training_tools(self):
        """Set up optimizer and loss criterion for training."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.criterion = nn.CrossEntropyLoss()

    def _init_weights(self, m):
        """Initialize weights of the model."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    def train_epoch(self, dataloader):
        """Train the model for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_accs = []
        for batch in tqdm.tqdm(dataloader, desc="Training..."):
            ids = batch["ids"].to(self.device)
            lengths = batch["length"]
            labels = batch["label"].to(self.device)
            predictions = self.model(ids, lengths)
            loss = self.criterion(predictions, labels)
            accuracy = self._get_accuracy(predictions, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
        return torch.mean(torch.tensor(epoch_losses)), torch.mean(
            torch.tensor(epoch_accs)
        )

    def evaluate_epoch(self, dataloader):
        """Evaluate the model for one epoch."""
        self.model.eval()
        epoch_losses = []
        epoch_accs = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Evaluating..."):
                ids = batch["ids"].to(self.device)
                lengths = batch["length"]
                labels = batch["label"].to(self.device)
                predictions = self.model(ids, lengths)
                loss = self.criterion(predictions, labels)
                accuracy = self._get_accuracy(predictions, labels)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())
        return torch.mean(torch.tensor(epoch_losses)), torch.mean(
            torch.tensor(epoch_accs)
        )

    def train_and_evaluate(self):
        """Train and evaluate the model for a specified number of epochs."""
        num_epochs = self.config["num_epochs"]
        best_valid_loss = float("inf")
        metrics = collections.defaultdict(list)
        model_name = self.config["model"]
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(self.train_loader)
            valid_loss, valid_acc = self.evaluate_epoch(self.valid_loader)
            metrics["train_losses"].append(train_loss)
            metrics["train_accs"].append(train_acc)
            metrics["valid_losses"].append(valid_loss)
            metrics["valid_accs"].append(valid_acc)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f"{model_name}.pt")
            print(f"Epoch: {epoch + 1}")
            print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")
            print(f"Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}")

    @staticmethod
    def _get_accuracy(predictions, labels):
        """Calculate accuracy of the predictions."""
        batch_size = predictions.shape[0]
        predicted_classes = predictions.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(labels).sum().item()
        accuracy = correct_predictions / batch_size
        return accuracy

    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )

    def visualize(self):
        """Visualize word clouds for positive and negative reviews in the training data."""
        data = {"text": self.train_data["text"], "label": self.train_data["label"]}
        df = pd.DataFrame(data)
        df["text"] = df["text"].apply(self._clean_text)

        # Positive reviews word cloud
        positive_reviews = df[df["label"] == 1]["text"]
        self._generate_wordcloud(positive_reviews, "Wordcloud for Positive Reviews")

        # Negative reviews word cloud
        negative_reviews = df[df["label"] == 0]["text"]
        self._generate_wordcloud(negative_reviews, "Wordcloud for Negative Reviews")

    @staticmethod
    def _clean_text(text):
        """Clean text by removing HTML tags, URLs, special characters, punctuation, and extra whitespace."""
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _generate_wordcloud(text_series, title):
        """Generate and display a word cloud from a series of texts."""
        text = " ".join(text_series.astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
