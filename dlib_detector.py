import dlib
import cv2

class DlibFaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dlib_faces = self.detector(gray)
        faces = [(d.left(), d.top(), d.width(), d.height()) for d in dlib_faces]
        return faces
