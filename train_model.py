import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class EmotionModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_fer2013(self, csv_path):
        """Load and preprocess FER2013 dataset"""
        df = pd.read_csv('fer2013.csv')
        
        # Convert string pixels to numpy arrays
        pixels = df['pixels'].apply(lambda x: np.array(x.split(' '), dtype='float32'))
        
        # Reshape pixels to 48x48 images
        X = np.array([pixel.reshape(48, 48, 1) for pixel in pixels])
        
        # Normalize pixel values
        X = X / 255.0
        
        # Get labels
        y = pd.get_dummies(df['emotion']).values
        
        return X, y
    
    def create_model(self):
        """Create CNN model for emotion recognition"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')  # 7 emotions
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def train(self, X, y, epochs=50, batch_size=64):
        """Train the model"""
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and compile model
        self.model = self.create_model()
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
            ]
        )
        
        # Save the model
        self.model.save('emotion_model.h5')
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

if __name__ == "__main__":
    trainer = EmotionModelTrainer()
    
    # Load and preprocess data
    print("Loading FER2013 dataset...")
    X, y = trainer.load_fer2013('fer2013.csv')
    
    # Train model
    print("Training model...")
    history = trainer.train(X, y)
    
    # Plot training history
    print("Plotting training history...")
    trainer.plot_training_history()
    
    print("Training complete! Model saved as 'emotion_model.h5'")  