import cv2
import face_recognition
import os
import streamlit as st
from PIL import Image
import numpy as np

# --- Load face detector ---
cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(cascade_path)
if face_classifier.empty():
    st.error("❌ Error: Could not load Haar Cascade.")
    st.stop()

# --- Load known faces from database directory ---
def load_known_faces(directory=os.path.join(os.path.dirname(__file__), 'database')):
    faces_dict = {}
    st.info("🔍 Loading reference faces...")

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            person_name = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            try:
                image = face_recognition.load_image_file(file_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    faces_dict[person_name] = encoding[0]
                    st.success(f"✅ Loaded face for: {person_name}")
                else:
                    st.warning(f"⚠️ No face found in: {filename}")
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {e}")
    return faces_dict

# --- Face recognition logic for images ---
def recognize_faces_in_frame(image, faces_dict):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    recognized_count = 0
    feedback_given = False

    for (x, y, w, h) in faces:
        face_rgb = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(face_rgb)
        if encodings:
            encoding = encodings[0]
            for name, ref_encoding in faces_dict.items():
                match = face_recognition.compare_faces([ref_encoding], encoding, tolerance=0.6)[0]
                if match:
                    recognized_count += 1
                    feedback_given = True
                    st.success(f"🎯 Recognized: {name}")
                    cv2.rectangle(image, (x, y), (x+w, y+h), (127, 0, 255), 2)
                    cv2.rectangle(image, (x, y-30), (x + len(name)*15, y), (0, 0, 0), -1)
                    cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    break

    if len(faces) == 0:
        st.warning("😶 No face detected.")
    elif not feedback_given:
        st.warning("❓ Face detected but not recognized.")
    return image, recognized_count

# --- Real-time recognition function ---
def run_live_face_recognition(faces_dict):
    st.info("📷 Starting real-time face recognition...")
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        st.error("❌ Error: Could not open webcam.")
        return

    stframe = st.empty()
    stop_button = st.button("Stop Recognition")
    
    while not stop_button:
        ret, frame = video_capture.read()
        if not ret:
            st.error("❌ Failed to read from camera.")
            break

        frame, _ = recognize_faces_in_frame(frame, faces_dict)
        stframe.image(frame, channels="BGR")
        
        if stop_button:
            break

    video_capture.release()
    st.success("🛑 Webcam session ended.")

# --- Capture one frame from webcam ---
def capture_image_from_webcam(faces_dict):
    st.info("📷 Opening webcam. Press the button to capture image.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Error: Cannot access webcam.")
        return

    stframe = st.empty()
    capture_button = st.button("Capture Image")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stframe.image(frame, channels="BGR")

        if capture_button:
            break

    cap.release()

    if capture_button:
        st.info("🧠 Processing captured image...")
        result_img, count = recognize_faces_in_frame(frame, faces_dict)
        st.image(result_img, channels="BGR")
        st.success(f"✅ Faces recognized: {count}")

# --- Select image from file ---
def select_image_from_pc(faces_dict):
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        st.info("🧠 Processing selected image...")
        result_img, count = recognize_faces_in_frame(image, faces_dict)
        st.image(result_img, channels="BGR")
        st.success(f"✅ Faces recognized: {count}")

# --- Main Streamlit App ---
def main():
    st.title("Face Recognition System")

    # Load faces at startup
    faces_dict = load_known_faces()
    if not faces_dict:
        st.error("❌ No valid reference faces found. Please add some faces to the database directory.")
        st.stop()

    # Sidebar menu
    st.sidebar.title("Menu")
    option = st.sidebar.selectbox(
        "Choose an option:",
        ["Real-time Recognition", "Capture Image", "Select Image", "Exit"]
    )

    if option == "Real-time Recognition":
        run_live_face_recognition(faces_dict)
    elif option == "Capture Image":
        capture_image_from_webcam(faces_dict)
    elif option == "Select Image":
        select_image_from_pc(faces_dict)
    elif option == "Exit":
        st.stop()

if __name__ == "__main__":
    main()
