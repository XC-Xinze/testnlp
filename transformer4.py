import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import argparse
from nltk.corpus import wordnet
import matplotlib.pyplot as plt

# Load processed dataset
def load_dataset():
    dataset = np.load("filtered_dataset.npz")
    return dataset['user_inputs'], dataset['keywords']

# Dataset definition
class KeywordDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# Compute weights for imbalanced training data
def compute_sample_weights(labels):
    class_counts = np.sum(labels, axis=0)
    total_samples = len(labels)
    class_weights = {cls: total_samples / count for cls, count in enumerate(class_counts) if count > 0}
    sample_weights = [np.mean([class_weights[idx] for idx, value in enumerate(label) if value > 0]) for label in labels]
    return sample_weights

# Transformer model definition
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_keywords, num_heads, num_encoder_layers, dropout_rate, max_sequence_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_sequence_length, embedding_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads, dropout=dropout_rate
            ),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(embedding_dim, num_keywords)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        encoded = self.encoder(embedded.permute(1, 0, 2))
        x = self.fc(encoded.mean(dim=0))
        return x  # No Sigmoid here, BCEWithLogitsLoss will apply it internally

# Preprocess input for testing
def preprocess_input(input_text, word_to_index, max_sequence_length):
    unk_index = word_to_index.get("<UNK>", 0)
    tokens = input_text.lower().split()
    sequence = [word_to_index.get(word, unk_index) for word in tokens]
    if len(sequence) < max_sequence_length:
        sequence += [0] * (max_sequence_length - len(sequence))
    else:
        sequence = sequence[:max_sequence_length]
    return torch.tensor([sequence], dtype=torch.long)

# Predict keywords
def predict_keywords(model, input_text, word_to_index, mlb, max_sequence_length, threshold=0.5):
    input_sequence = preprocess_input(input_text, word_to_index, max_sequence_length)
    with torch.no_grad():
        logits = model(input_sequence).squeeze(0).numpy()
        print(f"Prediction Logits: {logits}")
        binary_predictions = (logits > threshold).astype(int).reshape(1, -1)
        predicted_keywords = mlb.inverse_transform(binary_predictions)
    return predicted_keywords

# Train the model
def train_model():
    # Load dataset
    padded_user_inputs, binary_keywords = load_dataset()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        padded_user_inputs, binary_keywords, test_size=0.2, random_state=42
    )

    # Compute weights and create sampler
    train_sample_weights = compute_sample_weights(y_train)
    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

    # Create DataLoader
    batch_size = 32
    train_dataset = KeywordDataset(X_train, y_train)
    val_dataset = KeywordDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    vocab_size = padded_user_inputs.max() + 1
    num_keywords = binary_keywords.shape[1]
    max_sequence_length = padded_user_inputs.shape[1]

    model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=128,  # Reduced embedding_dim
        num_keywords=num_keywords,
        num_heads=4,  # Reduced number of heads
        num_encoder_layers=4,  # Reduced layers
        dropout_rate=0.3,
        max_sequence_length=max_sequence_length
    )

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Training loop
    num_epochs = 20
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Track the losses
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    # Plot the loss curves
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save model
    torch.save(model.state_dict(), "optimized_transformer_model.pth")
    print("Model saved as 'optimized_transformer_model.pth'")

# Test the model
def test_model(input_text):
    # Load dataset and model
    dataset = np.load("filtered_dataset.npz")
    padded_user_inputs = dataset['user_inputs']
    binary_keywords = dataset['keywords']
    with open("tokenizer.pkl", "rb") as f:
        word_to_index = pickle.load(f)
    with open("mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    # Model initialization
    vocab_size = padded_user_inputs.max() + 1
    num_keywords = binary_keywords.shape[1]
    max_sequence_length = padded_user_inputs.shape[1]

    model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=128,  # Same settings as in training
        num_keywords=num_keywords,
        num_heads=4,
        num_encoder_layers=4,
        dropout_rate=0.3,
        max_sequence_length=max_sequence_length
    )

    # Load model weights
    model.load_state_dict(torch.load("optimized_transformer_model.pth"))
    model.eval()

    # Predict keywords
    predicted_keywords = predict_keywords(model, input_text, word_to_index, mlb, max_sequence_length)
    print(f"Input Text: {input_text}")
    print(f"Predicted Keywords: {predicted_keywords}")

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", type=str, help="Test the model with an input text")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.test:
        test_model(args.test)
    else:
        print("Please specify --train or --test <input_text>")
