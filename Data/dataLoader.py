import torch
import torch.nn as nn


def create_collate_fn(pad_index):
    """
    Creates a collate function for padding sequences in a batch.
    """

    def collate_fn(batch):
        # Extract input sequences, lengths, and labels from the batch
        batch_ids = [sample["ids"] for sample in batch]
        batch_lengths = [sample["length"] for sample in batch]
        batch_labels = [sample["label"] for sample in batch]

        # Pad input sequences to the maximum length in the batch
        batch_ids_padded = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )

        # Convert batch lengths and labels to tensors
        batch_lengths_tensor = torch.stack(batch_lengths)
        batch_labels_tensor = torch.stack(batch_labels)

        # Construct the padded batch
        padded_batch = {
            "ids": batch_ids_padded,
            "length": batch_lengths_tensor,
            "label": batch_labels_tensor,
        }

        return padded_batch

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset: The dataset to load.
        batch_size (int): The batch size.
        pad_index (int): The padding index.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: The DataLoader.
    """
    # Create collate function
    collate_fn = create_collate_fn(pad_index)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
    )

    return data_loader
