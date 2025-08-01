import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout

# === 1. Load and Prepare Data ===
df = pd.read_csv("augmented_data.csv", encoding="utf-8")
texts = df['text'].astype(str).tolist()
labels = df['label'].replace({-1: 0, 1: 1}).tolist()
labels = np.array(labels)

# === 2. K-Fold Cross-Validation Setup ===
max_vocab_size = 10000
max_length = 100
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
all_metrics = []

# === 3. K-Fold Training ===
for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
    X_train, X_val = np.array(texts)[train_idx], np.array(texts)[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # New vectorizer per fold to prevent leakage
    vectorizer = TextVectorization(
        max_tokens=max_vocab_size,
        output_sequence_length=max_length,
        output_mode='int'
    )
    text_ds_train = tf.data.Dataset.from_tensor_slices(X_train).batch(32)
    vectorizer.adapt(text_ds_train)

    # === 4. Build Model ===
    model = Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        Embedding(input_dim=max_vocab_size, output_dim=64),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # === 5. Train Model ===
    print(f"\nTraining Fold {fold}...")
    model.fit(X_train, y_train, epochs=5, batch_size=32,
              validation_data=(X_val, y_val), verbose=1)

    # === 6. Evaluate ===
    y_pred_probs = model.predict(X_val)
    y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')

    print(f"✅ Fold {fold} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    all_metrics.append({
        'fold': fold,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# === 7. Final Metrics ===
results_df = pd.DataFrame(all_metrics)
print("\n📊 Average Metrics Across Folds:")
print(results_df.mean(numeric_only=True).round(4))

# === 8. Save Final Model ===
keras.models.save_model(model, "model.keras")

# === 9. Inference Loop ===
print("\n🔍 Sentiment Prediction (type 'exit' to quit):")

def predict_sentiment(text_input):
    prob = model.predict([text_input])[0][0]
    label = "Positive" if prob >= 0.5 else "Negative"
    return label, float(prob)

while True:
    user_input = input("📰 Enter a headline: ").strip()
    if user_input.lower() == "exit":
        print("👋 Exiting.")
        break
    label, prob = predict_sentiment(user_input)
    print(f"🔎 Prediction: {label} ({prob:.2f})\n")
