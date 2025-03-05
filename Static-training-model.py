import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class HandSignModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandSignModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HandSignDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna()

        try:
            self.y = torch.tensor(df.iloc[:, 0].astype(int).values, dtype=torch.long) 
            self.X = torch.tensor(df.iloc[:, 1:].astype(float).values, dtype=torch.float32) 
        except ValueError as e:
            print("Error in data conversion:", e)
            print("Check if all values are numeric.")
            exit(1)
        
        print(df)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    csv_path = 'C:\\Users\\shilp\\Downloads\\Gesture_Data.csv' 
    dataset = HandSignDataset(csv_path)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_size = dataset.X.shape[1] 
    num_classes = len(torch.unique(dataset.y)) 
    
    model = HandSignModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, criterion, optimizer)
    
    torch.save(model.state_dict(), "hand_sign_model.pth")
