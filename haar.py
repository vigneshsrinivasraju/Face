import cv2

class HaarFaceDetector:
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            # Use the default cascade file from OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, image):
        """
        Detect faces in the given image
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            List of detected faces as (x, y, w, h) rectangles
        """
        if image is None:
            return []
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces