from tensorflow.keras.models import load_model

model = load_model("model.keras")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({type(layer).__name__})")
