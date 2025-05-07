from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__, static_folder='static', template_folder='templates')

GENRE_COLUMNS = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
    'Horror', 'Documentary', 'Animation', 'Music', 'Crime'
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅  Using device: {DEVICE}")

# Load tokenizer and embedding matrix
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizer from JSON
with open('models/tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(f.read())

embedding_matrix = np.load('models/embedding_matrix.npy')

# Safe tokenization wrapper
def tokenize_plot(text, tokenizer, max_len=200):
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        seq = [0] * max_len
    padded = np.zeros((1, max_len), dtype=np.int64)
    padded[0, :len(seq)] = seq[:max_len]
    return torch.tensor(padded, dtype=torch.long, device=DEVICE)

# Define LSTM model
class GenreLSTM(nn.Module):
    def __init__(self, emb, hid=128, drop=0.3):
        super().__init__()
        vocab_size, emb_dim = emb.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(emb, dtype=torch.float32),
            requires_grad=False
        )
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hid * 2, len(GENRE_COLUMNS))

    def forward(self, x):
        lstm_out, _ = self.lstm(self.embedding(x))
        pooled = lstm_out.mean(dim=1)
        dropped = self.dropout(pooled)
        return self.fc(dropped)

# Load text model
text_model = GenreLSTM(embedding_matrix).to(DEVICE)
text_model.load_state_dict(torch.load('models/genre_classifier.pth', map_location=DEVICE))
text_model.eval()

# Load ResNet model for poster classification
poster_model = models.resnet34(pretrained=True)
poster_model.fc = nn.Linear(poster_model.fc.in_features, len(GENRE_COLUMNS))
poster_model.load_state_dict(torch.load('models/poster_genre_classifier.pth', map_location=DEVICE), strict=False)
poster_model.to(DEVICE).eval()

IMG_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.get_json(silent=True) or {}
    plot = (data.get("plot") or data.get("text") or "").strip()
    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    tensor = tokenize_plot(plot, tokenizer)
    with torch.no_grad():
        probs = torch.sigmoid(text_model(tensor))[0].cpu().numpy()
    top3 = probs.argsort()[-3:][::-1]
    return jsonify({"genres": [GENRE_COLUMNS[i] for i in top3]})

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if 'poster' not in request.files:
        return jsonify({"error": "No poster uploaded"}), 400
    try:
        img = Image.open(request.files['poster'].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    tensor = IMG_TF(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(poster_model(tensor))[0].cpu().numpy()
    top3 = probs.argsort()[-3:][::-1]
    return jsonify({"genres": [GENRE_COLUMNS[i] for i in top3]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
