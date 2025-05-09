# face-recognition

import cv2
import face_recognition
import os
from tkinter import filedialog, Tk

# --- Load face detector ---
FACE_CASCADE_PATH = 'Haarcascades/haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_classifier.empty():
    print("❌ Error: Could not load Haar Cascade.")
    exit(1)

# --- Load known faces from database directory ---
def load_known_faces(directory='database/'):
    faces_dict = {}
    print("🔍 Loading reference faces...")

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            person_name = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            try:
                image = face_recognition.load_image_file(file_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    faces_dict[person_name] = encoding[0]
                    print(f"✅ Loaded face for: {person_name}")
                else:
                    print(f"⚠️ No face found in: {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
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
                    print(f"🎯 Recognized: {name}")
                    cv2.rectangle(image, (x, y), (x+w, y+h), (127, 0, 255), 2)
                    cv2.rectangle(image, (x, y-30), (x + len(name)*15, y), (0, 0, 0), -1)
                    cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    break

    if len(faces) == 0:
        print("😶 No face detected.")
    elif not feedback_given:
        print("❓ Face detected but not recognized.")
    return image, recognized_count

# --- Main Menu ---
def main():
    faces_dict = load_known_faces()
    if not faces_dict:
        print("❌ No valid reference faces found. Exiting.")
        exit(1)

    print("\nChoose an option:")
    print("1. Real-time face recognition (live webcam)")
    print("2. Capture one image from webcam")
    print("3. Select an image from your PC")
    choice = input("Enter 1, 2 or 3: ").strip()

    if choice == '1':
        run_live_face_recognition(faces_dict)

    elif choice == '2':
        capture_image_from_webcam(faces_dict)

    elif choice == '3':
        select_image_from_pc(faces_dict)

    else:
        print("❌ Invalid choice. Exiting.")

# --- Option 1: Real-time recognition ---
def run_live_face_recognition(faces_dict):
    print("📷 Starting real-time face recognition...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🟢 Press 'q' to quit.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("❌ Failed to read from camera.")
            break

        frame, _ = recognize_faces_in_frame(frame, faces_dict)
        cv2.imshow('Real-Time Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam session ended.")

# --- Option 2: Capture one frame from webcam ---
def capture_image_from_webcam(faces_dict):
    print("📷 Opening webcam. Press SPACE to capture image.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Image - Press SPACE", frame)
        key = cv2.waitKey(1)
        if key % 256 == 32:  # SPACE key
            break

    cap.release()
    cv2.destroyAllWindows()

    print("🧠 Processing captured image...")
    result_img, count = recognize_faces_in_frame(frame, faces_dict)
    cv2.imshow("Captured Result", result_img)
    cv2.imwrite("captured_result.jpg", result_img)
    print(f"✅ Faces recognized: {count}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Option 3: Select image from file ---
def select_image_from_pc(faces_dict):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image",
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    image = cv2.imread(file_path)
    print("🧠 Processing selected image...")
    result_img, count = recognize_faces_in_frame(image, faces_dict)
    cv2.imshow("Selected Image Result", result_img)
    cv2.imwrite("file_result.jpg", result_img)
    print(f"✅ Faces recognized: {count}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Run the program ---
if __name__ == "__main__":
    main()
