import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from datetime import datetime

from detector.haar import HaarFaceDetector
from detector.mediapipe_detector import MediapipeFaceDetector

# App state
current_detector = HaarFaceDetector()
face_id = 0
save_folder = "saved_faces"
os.makedirs(save_folder, exist_ok=True)

# Switch detector
def change_detector(event):
    global current_detector
    selected = detector_choice.get()
    if selected == "haar":
        current_detector = HaarFaceDetector()
    elif selected == "mediapipe":
        current_detector = MediapipeFaceDetector()

# Save face
def save_face(face_img):
    global face_id
    filename = os.path.join(save_folder, f"face_{face_id}_{datetime.now().strftime('%H%M%S')}.jpg")
    cv2.imwrite(filename, face_img)
    face_id += 1

# Update video frame
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    faces = current_detector.detect_faces(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Save face for preview
        global last_face
        last_face = frame[y:y + h, x:x + w]

    # Convert to Tkinter format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# Button click handler
def capture_face():
    if last_face is not None:
        save_face(last_face)

# GUI window
app = tk.Tk()
app.title("Face Detection GUI")
app.geometry("700x550")

video_label = tk.Label(app)
video_label.pack()

detector_choice = ttk.Combobox(app, values=["haar", "mediapipe"])
detector_choice.current(0)
detector_choice.bind("<<ComboboxSelected>>", change_detector)
detector_choice.pack(pady=5)

capture_button = tk.Button(app, text="Capture Face", command=capture_face)
capture_button.pack(pady=10)

# Start camera
cap = cv2.VideoCapture(0)
last_face = None

update_frame()
app.mainloop()

cap.release()
cv2.destroyAllWindows()
