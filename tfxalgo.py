import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout

# === 1. Load and Prepare Data ===
df = pd.read_csv("augmented_data.csv", encoding="utf-8")
texts = df['text'].astype(str).tolist()
labels = df['label'].replace({-1: 0, 1: 1}).tolist()
labels = np.array(labels)

# === 2. Text Vectorization Layer ===
max_vocab_size = 10000
max_length = 100

vectorizer = TextVectorization(
    max_tokens=max_vocab_size,
    output_sequence_length=max_length,
    output_mode='int'
)
text_ds = tf.data.Dataset.from_tensor_slices(texts).batch(32)
vectorizer.adapt(text_ds)

# === 3. Vectorize All Data for CV ===
padded = np.array(vectorizer(np.array(texts)))
labels = np.array(labels)

# === 4. K-Fold Cross-Validation ===
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
all_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(padded, labels), 1):
    X_train, X_val = padded[train_idx], padded[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # === 5. Build Model with Embedded Vectorizer ===
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

    # === 6. Training ===
    print(f"\n🧠 Training Fold {fold}...")
    model.fit(np.array(texts)[train_idx], y_train, epochs=5, batch_size=32,
              validation_data=(np.array(texts)[val_idx], y_val), verbose=1)

    y_pred_probs = model.predict(np.array(texts)[val_idx])
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

# === 7. Final Results ===
results_df = pd.DataFrame(all_metrics)
print("\n📊 Average Metrics Across Folds:")
print(results_df.mean(numeric_only=True).round(4))

# === 8. Save Full Model as .keras ===
model.save("model.keras", save_format="keras")



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

