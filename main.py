import json

import streamlit as st
import torch
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from torch import nn

# Load the saved model and tokenizer
model_path = 'lstm_model.pth'
tokenizer_path = 'tokenizer.json'

# Load tokenizer
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)

# Load model parameters
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

class PoetryLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super(PoetryLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden


vocab_size = checkpoint['vocab_size']
embed_size = checkpoint['embed_size']
hidden_size = checkpoint['hidden_size']
num_layers = checkpoint['num_layers']

model = PoetryLSTM(vocab_size, embed_size, hidden_size, num_layers)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Utility function for poetry generation
def generate_poetry(model, input_text, tokenizer, max_length=50, temperature=0.7):
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]
    input_tensor = torch.LongTensor(input_sequence).unsqueeze(0)

    generated_sequence = input_sequence.copy()
    hidden = None
    recent_tokens = set()
    repetition_window = 5

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            output = output[:, -1, :] / temperature
            probabilities = torch.softmax(output, dim=-1)
            for token in recent_tokens:
                probabilities[0][token] *= 0.1
            top_k = 10
            top_probs, top_indices = torch.topk(probabilities, top_k)
            predicted_token = top_indices[0][torch.multinomial(torch.softmax(top_probs, dim=-1), 1)].item()

            if len(recent_tokens) >= repetition_window:
                recent_tokens.pop()
            recent_tokens.add(predicted_token)

            generated_sequence.append(predicted_token)
            input_tensor = torch.LongTensor([[predicted_token]])

            if predicted_token == tokenizer.word_index.get('<END>', 0):
                break

    generated_words = []
    for idx in generated_sequence:
        word = next((word for word, index in tokenizer.word_index.items()
                    if index == idx), '')
        if word and word not in ['<START>', '<END>', '<PAD>']:
            generated_words.append(word)

    return ' '.join(generated_words)


# Streamlit UI
st.title("Poetry Generation with LSTM")
st.write("Enter a prompt, adjust the sliders, and generate poetry!")

# User inputs
input_text = st.text_input("Input Text", value="aisā hai ki")
temperature = st.slider("Temperature (controls creativity)", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
max_length = st.slider("Max Length", min_value=10, max_value=100, value=50, step=10)

# Generate button
if st.button("Generate Poetry"):
    if input_text.strip():
        predicted_text = generate_poetry(model, input_text, tokenizer, max_length=max_length, temperature=temperature)
        st.subheader("Generated Poetry")
        st.write(predicted_text)
    else:
        st.warning("Please enter some input text to generate poetry.")
