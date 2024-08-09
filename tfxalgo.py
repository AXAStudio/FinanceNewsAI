import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class SentimentModel: 
    def __init__(self):
        self.df = pd.read_csv('shuffled_main.csv').dropna()
        self.df = self.df[self.df['sentiment'].isin([-1, 1])]  # Keep only -1 and 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['message'], self.df['sentiment'], test_size=0.1, random_state=42)

        # Convert labels to binary (0 for -1 and 1 for 1)
        self.y_train = (self.y_train == 1).astype(int)
        self.y_test = (self.y_test == 1).astype(int)

        VOCAB_SIZE = 10000  # Increased vocab size
        self.text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        train_messages = self.X_train.values
        self.text_vectorizer.adapt(train_messages)

        self._model = tf.keras.Sequential()
        self._hist_obj = None

    def build(self):
        self._model.add(self.text_vectorizer)
        self._model.add(tf.keras.layers.Embedding(len(self.text_vectorizer.get_vocabulary()), 128, mask_zero=True))
        self._model.add(tf.keras.layers.SpatialDropout1D(0.2))
        self._model.add(tf.keras.layers.GRU(256, return_sequences=True))
        self._model.add(tf.keras.layers.GRU(128))
        self._model.add(tf.keras.layers.Dense(64, activation='relu'))
        self._model.add(tf.keras.layers.Dropout(0.5))
        self._model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Single output neuron for binary classification
        
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self._model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                            optimizer=opt,
                            metrics=[
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                            ])

    def train(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test)).batch(32)

        self._hist_obj = self._model.fit(
            train_dataset,
            epochs=20,
            validation_data=val_dataset,
            verbose=1,
            shuffle=True,
            callbacks=[early_stopping, reduce_lr]
        )

    def evaluate(self):
        test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test)).batch(32)
        results = self._model.evaluate(test_dataset)
        print('Test Loss:', results[0])
        print('Test Accuracy:', results[1])
        print('Test Precision:', results[2])
        print('Test Recall:', results[3])

    def save(self):
        self._model.save('sentiment_model')  # Save the model

    def load(self):
        self._model = tf.keras.models.load_model('sentiment_model', custom_objects={'TextVectorization': self.text_vectorizer})

    def print_label_percentages(self):
        test_dataset = tf.data.Dataset.from_tensor_slices(self.X_test).batch(32)
        predictions = self._model.predict(test_dataset)
        predicted_labels = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels
        unique, counts = np.unique(predicted_labels, return_counts=True)
        percentages = dict(zip(unique, counts / len(predicted_labels) * 100))

        for label, percentage in percentages.items():
            print(f"Label {label}: {percentage:.2f}%")

    def predict_sentiment(self, user_input):
        # Ensure user_input is a single string
        if not isinstance(user_input, str):
            raise ValueError("Input must be a valid string")

        # Create a dataset for prediction
        model_input = tf.data.Dataset.from_tensor_slices([user_input]).batch(1)

        # Make the prediction
        prediction = self._model.predict(model_input)

        # Convert the prediction to sentiment (-1 to 1)
        sentiment = prediction[0] if prediction[0] > 0.5 else -1 * (1 - prediction[0])
        return sentiment
