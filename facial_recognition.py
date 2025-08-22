import os
import sys
import time
import platform
import ctypes
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

DATA_DIR = Path("face_data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FACE_SAMPLES_PER_USER = 60
MIN_FACES_TO_TRAIN = 20
CAM_INDEX = 0

REQUIRED_CONSEC_MATCHES = 8
RECOGNITION_CONFIDENCE_MAX = 80

CENTER_TOLERANCE = 0.18
BRIGHTNESS_CLIP_LIMIT = 2.0
EYE_MIN_AREA = 80

WARNING_LIMIT = 3
WARN_COOLDOWN = 1.6
SESSION_TIMEOUT_MIN = 10.0

ALERT_FLASH_SECS = 0.8

HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
HAAR_EYE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

MODEL_PATH = MODEL_DIR / "lbph_face_model.xml"
LABELS_PATH = MODEL_DIR / "labels.npy"

def lock_computer():
    system_name = platform.system().lower()
    try:
        if "windows" in system_name:
            os.system("rundll32.exe user32.dll,LockWorkStation")
        elif "darwin" in system_name:
            os.system("/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend")
        else:
            os.system("loginctl lock-session || gnome-screensaver-command -l || xdg-screensaver lock")
    except Exception as e:
        st.error(f"Lock failed: {e}")

def turn_off_screen():
    system_name = platform.system().lower()
    try:
        if "windows" in system_name:
            HWND_BROADCAST     = 0xFFFF
            WM_SYSCOMMAND      = 0x0112
            SC_MONITORPOWER    = 0xF170
            ctypes.windll.user32.SendMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)
        elif "darwin" in system_name:
            os.system("pmset displaysleepnow")
        else:
            os.system(
                "xset dpms force off "
                "|| (command -v swaymsg >/dev/null 2>&1 && swaymsg 'output * dpms off') "
                "|| (command -v hyprctl >/dev/null 2>&1 && hyprctl dispatch dpms off) "
                "|| true"
            )
    except Exception as e:
        st.error(f"Turn-off-screen failed: {e}")

def toast_alert(msg: str):
    try:
        st.toast(msg, icon="⚠️")
    except Exception:
        st.warning(msg)

def sound_alert():
    try:
        system_name = platform.system().lower()
        if "windows" in system_name:
            import winsound
            winsound.Beep(1000, 300)
        elif "darwin" in system_name:
            os.system("afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true")
        else:
            os.system(
                "paplay /usr/share/sounds/freedesktop/stereo/dialog-warning.oga 2>/dev/null "
                "|| aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null "
                "|| true"
            )
    except Exception:
        pass

def flash_overlay(frame, alpha=0.30, color=(0, 0, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def load_or_init_model():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        st.error("LBPHFaceRecognizer missing. Install the contrib build:\n\n`pip install opencv-contrib-python`")
        return None, {}
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        recognizer.read(str(MODEL_PATH))
        labels = np.load(LABELS_PATH, allow_pickle=True).item()
    else:
        recognizer = None
        labels = {}
    return recognizer, labels

def save_model(recognizer, labels):
    recognizer.write(str(MODEL_PATH))
    np.save(LABELS_PATH, labels)

def train_model_from_data():
    faces = []
    y = []
    user_to_id = {}
    current_label = 0
    for user_dir in DATA_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        files = list(user_dir.glob("*.png"))
        if len(files) < MIN_FACES_TO_TRAIN:
            st.warning(f"Skipping '{user_dir.name}': only {len(files)} images (min {MIN_FACES_TO_TRAIN}).")
            continue
        user_to_id[user_dir.name] = current_label
        for fp in files:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            y.append(current_label)
        current_label += 1
    if not faces:
        raise RuntimeError("No sufficient training data found. Register users first.")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        raise RuntimeError("LBPHFaceRecognizer missing. Install opencv-contrib-python")
    recognizer.train(faces, np.array(y))
    save_model(recognizer, user_to_id)
    return user_to_id

def find_primary_face(gray):
    faces = HAAR_FACE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100,100))
    if len(faces) == 0:
        return None
    faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    return faces_sorted[0]

def detect_gaze_direction(face_gray):
    eyes = HAAR_EYE.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
    if len(eyes) == 0:
        return 'unknown'
    clahe = cv2.createCLAHE(clipLimit=BRIGHTNESS_CLIP_LIMIT, tileGridSize=(8,8))
    dirs = []
    for (ex, ey, ew, eh) in eyes[:2]:
        eye = face_gray[ey:ey+eh, ex:ex+ew]
        eye = cv2.equalizeHist(eye)
        eye = clahe.apply(eye)
        _, th = cv2.threshold(eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.medianBlur(th, 3)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < EYE_MIN_AREA:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx_norm = (cx / ew) - 0.5
        cy_norm = (cy / eh) - 0.5
        horiz = 'center'
        vert  = 'center'
        if cx_norm < -CENTER_TOLERANCE:
            horiz = 'left'
        elif cx_norm > CENTER_TOLERANCE:
            horiz = 'right'
        if cy_norm < -CENTER_TOLERANCE:
            vert = 'up'
        elif cy_norm > CENTER_TOLERANCE:
            vert = 'down'
        if horiz != 'center':
            dirs.append(horiz)
        elif vert != 'center':
            dirs.append(vert)
        else:
            dirs.append('center')
    if not dirs:
        return 'unknown'
    vals, counts = np.unique(np.array(dirs), return_counts=True)
    return vals[np.argmax(counts)]

def register_user_streamlit(username: str):
    user_dir = DATA_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    st.info("Capturing images. Keep your face in frame. This auto-finishes when enough samples are collected.")
    frame_placeholder = st.empty()
    progress = st.progress(0, text="Starting camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return
    count = 0
    last_save_time = 0
    save_gap = 0.07
    try:
        while count < FACE_SAMPLES_PER_USER:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = HAAR_FACE.detectMultiScale(gray, 1.2, 5, minSize=(100,100))
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                now = time.time()
                if now - last_save_time > save_gap:
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (200,200))
                    fp = user_dir / f"{username}_{int(time.time()*1000)}.png"
                    cv2.imwrite(str(fp), face_resized)
                    count += 1
                    last_save_time = now
            cv2.putText(frame, f"User: {username}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Captured: {count}/{FACE_SAMPLES_PER_USER}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            progress.progress(int(100*count/FACE_SAMPLES_PER_USER), text=f"Captured {count}/{FACE_SAMPLES_PER_USER}")
        progress.progress(100, text="Training model...")
        labels = train_model_from_data()
        st.success(f"✅ Registered '{username}'. Model trained.\nLabels: {labels}")
    except Exception as e:
        st.error(f"Registration error: {e}")
    finally:
        cap.release()

def login_and_monitor_streamlit(expected_user: str):
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("login_verified_frames", 0)
    st.session_state.setdefault("alert_log", [])
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    notice_placeholder = st.empty()
    alerts_placeholder = st.empty()
    recognizer, labels = load_or_init_model()
    if recognizer is None or not labels:
        st.error("No trained model/labels found. Please register first.")
        return
    id_to_user = {v: k for k, v in labels.items()}
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return
    warnings = 0
    last_warn_time = 0
    alert_flash_until = 0.0
    start_time = time.time()
    end_time = start_time + SESSION_TIMEOUT_MIN*60
    try:
        notice_placeholder.info("Align your face. Login requires successful face recognition.")
        while not st.session_state["logged_in"]:
            if time.time() > end_time:
                st.info("Session timed out. Returning to home.")
                cap.release()
                return
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            primary = find_primary_face(gray)
            if primary is None:
                cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                continue
            (x,y,w,h) = primary
            face_roi = gray[y:y+h, x:x+w]
            face_norm = cv2.resize(face_roi, (200,200))
            label_id, confidence = recognizer.predict(face_norm)
            recognized_user = id_to_user.get(label_id, "unknown")
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)
            cv2.putText(frame, f"{recognized_user} (conf {confidence:.1f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            if recognized_user == expected_user and confidence <= RECOGNITION_CONFIDENCE_MAX:
                st.session_state["login_verified_frames"] += 1
            else:
                st.session_state["login_verified_frames"] = 0
            cv2.putText(frame,
                        f"Login verifying: {st.session_state['login_verified_frames']}/{REQUIRED_CONSEC_MATCHES}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            if st.session_state["login_verified_frames"] >= REQUIRED_CONSEC_MATCHES:
                st.session_state["logged_in"] = True
                break
        notice_placeholder.success(
            "Hi, welcome to facial recognition system. To logout, look away from the screen."
        )
        while True:
            if time.time() > end_time:
                st.info("Session timed out. Returning to home.")
                break
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            primary = find_primary_face(gray)
            if primary is None:
                cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if time.time() < alert_flash_until:
                    frame = flash_overlay(frame)
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                continue
            (x,y,w,h) = primary
            face_roi = gray[y:y+h, x:x+w]
            face_norm = cv2.resize(face_roi, (200,200))
            label_id, confidence = recognizer.predict(face_norm)
            recognized_user = id_to_user.get(label_id, "unknown")
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)
            cv2.putText(frame, f"{recognized_user} (conf {confidence:.1f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            if recognized_user != expected_user or confidence > RECOGNITION_CONFIDENCE_MAX:
                cv2.putText(frame, "UNAUTHORIZED! Locking…", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                lock_computer()
                st.error("Unauthorized user detected. Logging out.")
                break
            face_for_gaze = cv2.equalizeHist(face_roi)
            direction = detect_gaze_direction(face_for_gaze)
            cv2.putText(frame, f"Gaze: {direction}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if direction not in ('center', 'unknown'):
                now = time.time()
                if now - last_warn_time > WARN_COOLDOWN:
                    warnings += 1
                    last_warn_time = now
                    toast_alert(f"Warning {warnings}/{WARNING_LIMIT}: Face front!")
                    sound_alert()
                    alert_flash_until = time.time() + ALERT_FLASH_SECS
                    st.session_state["alert_log"].append(
                        {"t": time.strftime('%H:%M:%S'), "direction": direction, "count": warnings}
                    )
            if time.time() < alert_flash_until:
                frame = flash_overlay(frame, alpha=0.35)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            stats_placeholder.info(
                f"Logged in as **{recognized_user}** · Confidence: {confidence:.1f} · "
                f"Gaze warnings: {warnings}/{WARNING_LIMIT}"
            )
            if st.session_state["alert_log"]:
                recent = st.session_state["alert_log"][-5:]
                alerts_placeholder.write(
                    "🔔 **Recent alerts:** " +
                    " | ".join([f"[{a['t']}] {a['direction']} → {a['count']}" for a in recent])
                )
            if warnings >= WARNING_LIMIT:
                cv2.putText(frame, "Max warnings reached. Turning screen off…",
                            (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                st.error("You looked away three times. Screen will turn off and you are logged out.")
                lock_computer()
                turn_off_screen()
                break
    except Exception as e:
        st.error(f"Monitoring error: {e}")
    finally:
        cap.release()
        st.session_state["logged_in"] = False
        st.session_state["login_verified_frames"] = 0

st.set_page_config(page_title="Facial Recognition System", layout="centered")
st.title("👤 Facial Recognition System")
st.caption("Register with your face, then log in. Gaze is monitored; look away 3× → screen turns off.")

mode = st.radio("Choose an option", ["Login", "Register"], horizontal=True)

if mode == "Register":
    st.subheader("Register")
    username = st.text_input("Enter a new username", value="", max_chars=40)
    if st.button("Start Registration", type="primary", disabled=(len(username.strip()) == 0)):
        register_user_streamlit(username.strip())
elif mode == "Login":
    st.subheader("Login")
    username = st.text_input("Enter your registered username", value="", max_chars=40)
    st.caption("Login requires a positive face match. Only the logged-in user may stay in front of the computer.")
    if st.button("Start Login", type="primary", disabled=(len(username.strip()) == 0)):
        login_and_monitor_streamlit(username.strip())
