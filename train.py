import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load Data
try:
    df = pd.read_csv("cv_data.csv")
    df.dropna(inplace=True)
    print(f"Loaded {len(df)} records")
except FileNotFoundError:
    print("❌ cv_data.csv file not found. Please check the path")
    exit()

# Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Encode specialization as numbers
specialization_encoder = LabelEncoder()
df["specialization_encoded"] = specialization_encoder.fit_transform(df["specialization"])

# Split data
train_texts, test_texts, train_ages, test_ages, train_exps, test_exps, train_specs, test_specs, train_labels, test_labels = train_test_split(
    df["cv_text"].tolist(),
    df["age"].tolist(),
    df["experience_years"].tolist(),
    df["specialization_encoded"].tolist(),
    df["label_encoded"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label_encoded"]
)

print(f"Training data size: {len(train_texts)}")
print(f"Test data size: {len(test_texts)}")

# Normalize numerical features (age + experience)
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(list(zip(train_ages, train_exps)))
test_features_scaled = scaler.transform(list(zip(test_ages, test_exps)))

#  Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset
class CV_Dataset(Dataset):
    def __init__(self, texts, features, specs, labels):
        self.texts = texts
        self.features = features
        self.specs = specs
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=256,  
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "features": torch.tensor(self.features[idx], dtype=torch.float),
            "spec": torch.tensor(self.specs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = CV_Dataset(train_texts, train_features_scaled, train_specs, train_labels)
test_dataset = CV_Dataset(test_texts, test_features_scaled, test_specs, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model
class CVScoringModel(nn.Module):
    def __init__(self, num_specs):
        super(CVScoringModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.spec_embedding = nn.Embedding(num_embeddings=num_specs, embedding_dim=16)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size + 2 + 16, 3)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, features, spec):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        spec_embed = self.spec_embedding(spec)
        combined = torch.cat((pooled_output, features, spec_embed), dim=1)
        logits = self.fc(self.dropout(combined))
        return logits

# Training
model = CVScoringModel(num_specs=len(specialization_encoder.classes_)).to(device)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label_encoded"]),
    y=df["label_encoded"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=2e-5,
    weight_decay=0.01
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

epochs = 6
best_f1 = 0
train_losses = []
val_accuracies = []
val_f1_scores = []

print("Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch["features"].to(device)
        spec = batch["spec"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, features, spec)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    scheduler.step()

    # Validation
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            spec = batch["spec"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, features, spec)
            _, predictions = torch.max(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    val_accuracies.append(accuracy)
    val_f1_scores.append(f1)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "cv_scoring_classifier.pt")
        print(f"Saved best model with F1: {f1:.4f}")

#Final Evaluation
print("\n" + "="*50)
print("Final evaluation on test data")
print("="*50)

model.load_state_dict(torch.load("cv_scoring_classifier.pt"))
model.eval()
all_predictions, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch["features"].to(device)
        spec = batch["spec"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, features, spec)
        _, predictions = torch.max(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Final Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")
print(f"F1 Score: {f1_score(all_labels, all_predictions, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, 
                            target_names=label_encoder.classes_))

# Save model and resources
torch.save(model.state_dict(), "cv_scoring_classifier.pt")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(specialization_encoder, "specialization_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and encoders saved successfully")
print(" Model info:")
print(f"- Number of classes: {len(label_encoder.classes_)}")
print(f"- Number of specializations: {len(specialization_encoder.classes_)}")
print(f"- Best F1 Score: {best_f1:.4f}")
