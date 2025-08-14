import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
from src.model import build_model
import os

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


if __name__ == "__main__":
    from src.data_loader import load_training_data, load_class_names

    # Step 1: Load all training images and labels
    X, y = load_training_data(image_size=(32, 32))

    # Step 2: Filter selected classes
    selected_classes = [1, 14, 17, 38]  # Speed limit (30), Stop, No entry, Keep right
    filtered_indices = [i for i, label in enumerate(y) if label in selected_classes]
    X = X[filtered_indices]
    y = y[filtered_indices]

    # Step 3: Re-map labels to 0 to N-1
    label_map = {original: new for new, original in enumerate(selected_classes)}
    y = np.array([label_map[label] for label in y])

    # Step 4: Normalize image data
    X = X.astype('float32') / 255.0

    # Step 5: One-hot encode the labels
    num_classes = len(selected_classes)
    y = to_categorical(y, num_classes)

    # Step 6: Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("Training samples:", X_train.shape[0])
    print("Validation samples:", X_val.shape[0])

    # Step 7: Load class names
     # Load full class name mapping
    all_class_names = load_class_names()

    # Fix: reverse label_map to go from remapped → original
    reverse_label_map = {new: old for old, new in label_map.items()}

    # Now map remapped labels (0–3) to correct names
    class_names = {new: all_class_names[original] for new, original in reverse_label_map.items()}


    # Step 8: Visualize random samples
    num_samples = 10
    indices = random.sample(range(X_train.shape[0]), num_samples)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        image = X_train[idx]
        label = y_train[idx].argmax()
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(f"{class_names[int(label)]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


    # Create a folder to store the files
    if not os.path.exists("data/processed/X_train.npy"):
        np.save("data/processed/X_train.npy", X_train)
        np.save("data/processed/X_val.npy", X_val)
        np.save("data/processed/y_train.npy", y_train)
        np.save("data/processed/y_val.npy", y_val)
        print("Preprocessed data saved.")
    else:
        print("ℹPreprocessed data already exists. Skipping save.")

    # Load preprocessed training and validation data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    # Build the model
    model = build_model(input_shape=(32, 32, 3), num_classes=4)

    # Print model summary
    model.summary()

    # Create a directory to save models if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Define callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath='models/best_cnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    import matplotlib.pyplot as plt

    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


    # Step 1: Predict on validation set
    y_pred = model.predict(X_val)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true_classes = y_val.argmax(axis=1)

    # Step 2: Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Step 3: Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[i] for i in range(4)],
                yticklabels=[class_names[i] for i in range(4)])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Step 4: Classification report
    print("\nClassification Report:\n")
    print(classification_report(y_true_classes, y_pred_classes, target_names=[class_names[i] for i in range(4)]))
