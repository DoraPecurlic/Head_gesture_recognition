import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
from collections import Counter
import time
import os

import pyautogui
pyautogui.FAILSAFE = False
import tkinter as tk
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "best_val_acc_model.h5")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

model = load_model(model_path)
with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
actions = label_encoder.classes_


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


def preprocess_landmarks(landmarks):
    nose = landmarks[1]
    
    
    rel_frame = [[x- nose[0], y- nose[1]] for (x,y) in landmarks]
    
   
    chin = landmarks[152]
    mid_eyebrows = landmarks[168]
    face_height = np.linalg.norm(np.array(chin) - np.array(mid_eyebrows))
    
    if face_height == 0:
        return np.array(rel_frame).flatten()
    
    
    scaled_frame = [[x/face_height, y/face_height] for (x,y) in rel_frame]
    return np.array(scaled_frame).flatten()


def extract_face_keypoints(results):
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        keypoints =[[lm.x, lm.y] for lm in landmarks.landmark]
        return preprocess_landmarks(keypoints)
        
    else:
        print("Lice NIJE detektirano")
        return np.zeros(956)
    
window = tk.Tk()
window.title("Head Gesture Recognition")
window.geometry("640x480+0+0")
window.wm_attributes("-topmost", True)
window.resizable(False, False)
label = tk.Label(window)
label.pack()

sequence = []
predictions = []
threshold = 0.7
current_gesture = ""
last_gesture = "" 
gesture_cooldown = 2
last_gesture_time = time.time()
frame_counter = 0


confidence_map = {
    "no_gesture": 0.75,
    "turn_right": 0.78,
    "turn_left": 0.78,
    "turn_right_twice": 0.70,
    "turn_left_twice": 0.70,
    "nod": 0.80,
    "up": 0.75,
    "down": 0.75,
    "tilt_right": 0.75,
    "tilt_left": 0.75
}

cap = cv2.VideoCapture(0)

def get_gesture_override(res, actions):
    gesture_override = None
    if "nod" in actions and "no_gesture" in actions:
        nod_idx = actions.tolist().index("nod")
        nogest_idx = actions.tolist().index("no_gesture")
        nod_prob = res[nod_idx]
        nogest_prob = res[nogest_idx]
        max_other_nod = max([p for i, p in enumerate(res) if i != nod_idx])
        max_other_nogest = max([p for i, p in enumerate(res) if i != nogest_idx])

        if nod_prob > 0.80 and nod_prob - max_other_nod > 0.15:
            gesture_override = "nod"
        elif nogest_prob > 0.75 and nogest_prob - max_other_nogest > 0.25:
            gesture_override = "no_gesture"
    return gesture_override



def perform_action(gesture):
    match gesture:
        case "up":
            pyautogui.press('f')
        case "down":
            pyautogui.press('esc')
        case "nod":
            pyautogui.press('space')
        case "tilt_right":
            pyautogui.hotkey('shift', 'n')
        case "tilt_left":
            pyautogui.hotkey('shift', 'p')
        case "turn_right":
            pyautogui.press('up')
        case "turn_left":
            pyautogui.press('down')
        case "turn_right_twice":
            for _ in range(3):
                pyautogui.press('l')
        case "turn_left_twice":
            for _ in range(3):
                pyautogui.press('j')


def show():
    global frame_counter, sequence, predictions, last_gesture, current_gesture, last_gesture_time

    ret, frame = cap.read()
    if not ret:
        window.after(10, show)
        return

    frame_counter += 1
    if frame_counter % 3 != 0:
        window.after(10, show)
        return

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    keypoints = extract_face_keypoints(results)

    sequence.append(keypoints)
    sequence = sequence[-17:]

    if len(sequence) == 17:
        input_data = np.expand_dims(sequence, axis=0)
        res = model.predict(input_data, verbose=0)[0]
        for i, prob in enumerate(res):
            print(f"{actions[i]} : {prob:.2f}")

        pred_class = np.argmax(res)
        predictions.append(pred_class)

        gesture = actions[pred_class]
        gesture_override = get_gesture_override(res, actions)


        if len(predictions) >= 5:
            most_common = Counter(predictions[-5:]).most_common(1)[0][0]
            gesture = actions[most_common]
            min_conf = confidence_map.get(gesture, 0.7)

            if res[most_common] < min_conf:
                gesture = last_gesture

            if gesture_override:
                gesture = gesture_override

    
            if gesture == "turn_right_twice" and res[most_common] < 0.6:
                gesture = "turn_right"
            elif gesture == "turn_left_twice" and res[most_common] < 0.7:
                gesture = "turn_left"

            now = time.time()
            if now - last_gesture_time > gesture_cooldown:
                if gesture != last_gesture:
                    recent_gestures = [actions[pred] for pred in predictions[-20:]]
                    if recent_gestures.count(gesture) > 12:
                        current_gesture = gesture
                        last_gesture = gesture
                        last_gesture_time = now
                        perform_action(current_gesture)
                else:
                    current_gesture = gesture
            else:
                current_gesture = last_gesture

    
    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, current_gesture, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    img = Image.fromarray(image)
    imgTk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgTk
    label.configure(image=imgTk)

    window.after(10, show)


show()
window.mainloop()
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
