import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

class EmotionDetector:
    def __init__(self):
        # Initialize MTCNN for face detection
        self.detector = MTCNN()
        
        # Load the trained emotion recognition model
        self.model = self.load_model()
        
        # Define emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
    def load_model(self):
        """Load the pre-trained emotion recognition model"""
        try:
            model = tf.keras.models.load_model('emotion_model.h5')
            print("Model loaded successfully")
            return model
        except:
            print("No pre-trained model found. Please train the model first.")
            return None
    
    def preprocess_face(self, face_img):
        """Preprocess the face image for emotion prediction"""
        # Resize to 48x48 (FER2013 input size)
        face_img = cv2.resize(face_img, (48, 48))
        # Convert to grayscale
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Normalize pixel values
        face_img = face_img / 255.0
        # Reshape for model input
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        return face_img
    
    def detect_emotions(self):
        """Main function to detect emotions in real-time"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert frame to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.detector.detect_faces(rgb_frame)
            
            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']
                
                # Only process faces with high confidence
                if confidence > 0.9:
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Preprocess face
                    processed_face = self.preprocess_face(face_img)
                    
                    # Predict emotion
                    if self.model is not None:
                        predictions = self.model.predict(processed_face)
                        emotion_idx = np.argmax(predictions)
                        emotion = self.emotion_labels[emotion_idx]
                        confidence = predictions[0][emotion_idx]
                        
                        # Draw bounding box and emotion label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{emotion}: {confidence:.2f}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.9, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Emotion Detection', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.detect_emotions() 