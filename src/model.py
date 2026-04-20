import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(df, target_col='Outcome'):
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    print(f"✅ Train : {X_train.shape[0]} lignes")
    print(f"✅ Test  : {X_test.shape[0]} lignes")

    return X_train, X_test, y_train, y_test, scaler

class MLModel(nn.Module):
    def __init__(self, input_size):
        super(MLModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def train_model(X_train, y_train, epochs=150):
    input_size = X_train.shape[1]
    model = MLModel(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n=== Entraînement du modèle ===")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        print(f"\n=== Résultats ===")
        print(f"✅ Précision : {accuracy.item() * 100:.2f}%")

def save_model(model, path="models/model.pt"):
    torch.save(model, path)
    print(f"✅ Modèle sauvegardé dans {path}")