import argparse
import time
from collections import deque
import threading

import cv2
import numpy as np

from .classifier import GestureClassifier
from .utils import extract_features, draw_hand_landmarks, put_text_box
import mediapipe as mp

try:
    import pyttsx3  # optional
except Exception:
    pyttsx3 = None


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam Sign Language Recognition with MediaPipe")
    parser.add_argument("--model", type=str, default="models/gesture_model.joblib", help="Path to trained joblib model")
    parser.add_argument("--labels", type=str, default="models/labels.txt", help="Path to labels txt (one per line)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture")
    parser.add_argument("--min_det_conf", type=float, default=0.6, help="MediaPipe min detection confidence")
    parser.add_argument("--min_trk_conf", type=float, default=0.6, help="MediaPipe min tracking confidence")
    parser.add_argument("--smoothing", type=int, default=8, help="Smoothing window (frames) for probabilities")
    parser.add_argument("--tts", action="store_true", help="Enable text-to-speech")
    parser.add_argument("--speak_each", action="store_true", help="Speak each recognized sign immediately")
    parser.add_argument("--flip", action="store_true", help="Flip the webcam feed horizontally")
    parser.add_argument("--auto_append", action="store_true", help="Automatically append stable predictions")
    parser.add_argument("--stable_frames", type=int, default=6, help="Frames required for a stable prediction")
    parser.add_argument("--min_conf", type=float, default=0.6, help="Minimum confidence to accept a prediction")
    parser.add_argument("--deaf_mode", action="store_true", help="Large on-screen captions, no speech output")
    parser.add_argument("--speak_mode", action="store_true", help="Map gestures to words and speak full sentence on commit gesture")
    parser.add_argument("--speak_continuous", action="store_true", help="Speak the sentence whenever it updates")
    return parser.parse_args()


def init_tts(enabled: bool):
    if not enabled:
        return None
    if pyttsx3 is None:
        print("pyttsx3 not available. Install it or run without --tts")
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 1.0)
        try:
            voices = engine.getProperty('voices')
            preferred = None
            for v in voices:
                name = (getattr(v, 'name', '') or '').lower()
                lang = ''.join(getattr(v, 'languages', [])).lower() if hasattr(v, 'languages') else ''
                if ('en' in lang or 'english' in name) and any(k in name for k in ['zira', 'aria', 'eva', 'female']):
                    preferred = v.id
                    break
            if preferred is None and voices:
                preferred = voices[0].id
            if preferred:
                engine.setProperty('voice', preferred)
        except Exception:
            pass
        return engine
    except Exception as e:
        print(f"Failed to init TTS: {e}")
        return None


def speak_async(engine, text: str):
    if not engine or not text:
        return
    def _run():
        try:
            print(f"[TTS] Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
        except Exception as ex:
            print(f"[TTS] Error: {ex}")
    threading.Thread(target=_run, daemon=True).start()


def main():
    args = parse_args()

    tts_engine = init_tts(args.tts)

    classifier = GestureClassifier(model_path=args.model, labels_path=args.labels, smoothing=args.smoothing)

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    sentence = ""
    last_spoken_text = ""
    last_prediction = deque(maxlen=10)
    recent_labels = deque(maxlen=max(3, args.stable_frames))
    last_appended = ""
    sentence_tokens: list[str] = []

    if args.deaf_mode:
        args.tts = False
        args.speak_each = False
        args.auto_append = True

    gesture_to_word = {
        "Open": "hello",
        "Point": "you",
        "Two": "I",
        "Three": "love",
        "Four": "thank",
        "Thumb": " ",  # space
        "Fist": "yes",
        "Flat": "no",
        "OK": "good",
        "Pinch": "please",
        "Claw": "help",
        "CShape": "drink",
        "VShape": "see",
        "LShape": "stop",
        "PalmUp": "what",
        "PalmDown": "where",
        "PalmIn": "who",
        "PalmOut": "why",
        "Hook": "hungry",
        "FlatO": "sorry",
        "BentV": "sleep",
        "FiveSpread": "friend",
        "ThreeBent": "family",
        "OneBent": "work"
    }

    commit_gestures = {"OK", "Pinch"}    # commit & speak collected tokens
    clear_gestures = {"Fist", "One"}     # now clears with Fist OR One

    if args.speak_mode:
        args.tts = True
        args.speak_each = False
        args.auto_append = True

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_trk_conf,
    ) as hands:
        fps_time = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            pred_label = "No Hand"
            pred_prob = 0.0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)
                    feats = extract_features(hand_landmarks, frame.shape)
                    label, prob = classifier.predict(feats)
                    pred_label, pred_prob = label, prob

            if args.auto_append:
                if pred_label not in ("No Hand", "Unknown"):
                    recent_labels.append(pred_label)
                else:
                    recent_labels.clear()
                if len(recent_labels) == recent_labels.maxlen and len(set(recent_labels)) == 1 and pred_prob >= args.min_conf:
                    stable = pred_label
                    if args.speak_mode:
                        if stable in clear_gestures:
                            sentence_tokens.clear()
                            sentence = ""
                            last_appended = ""
                        elif stable in commit_gestures:
                            sentence = " ".join([t for t in sentence_tokens if t.strip()])
                            if tts_engine and sentence.strip():
                                speak_async(tts_engine, sentence)
                            sentence_tokens.clear()
                            last_appended = ""
                        else:
                            token = gesture_to_word.get(stable)
                            if token is not None and token != last_appended:
                                if token == " ":
                                    sentence += " "
                                else:
                                    sentence_tokens.append(token)
                                    preview = " ".join([t for t in sentence_tokens if t.strip()])
                                    sentence = preview
                                    if args.speak_continuous and tts_engine and sentence.strip():
                                        speak_async(tts_engine, sentence)
                                last_appended = token
                    else:
                        if stable in clear_gestures:
                            sentence = ""
                            last_appended = ""
                        else:
                            to_append = stable
                            if to_append == "Thumb":
                                to_append = " "
                            if to_append != last_appended:
                                sentence += to_append
                                last_appended = to_append
                                if args.speak_each and tts_engine and isinstance(to_append, str) and to_append.strip():
                                    speak_async(tts_engine, to_append)
                                if args.speak_continuous and tts_engine and sentence.strip():
                                    speak_async(tts_engine, sentence)

            now = time.time()
            dt = now - fps_time
            fps_time = now
            fps = 1.0 / dt if dt > 0 else 0.0

            if args.deaf_mode:
                h, w = frame.shape[:2]
                bar_h = int(0.22 * h)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                display_text = sentence if sentence.strip() else f"{pred_label} ({pred_prob:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(1.2, w / 900.0)
                thickness = 2
                margin = 20
                max_width_px = w - 2 * margin
                words = display_text.split()
                lines = []
                cur = ""
                for word in words:
                    test = (cur + " " + word).strip()
                    (tw, th), _ = cv2.getTextSize(test, font, font_scale, thickness)
                    if tw <= max_width_px:
                        cur = test
                    else:
                        if cur:
                            lines.append(cur)
                            cur = word
                        else:
                            lines.append(word)
                            cur = ""
                if cur:
                    lines.append(cur)
                y = h - bar_h + margin + 30
                for i, line in enumerate(lines[:3]):
                    cv2.putText(frame, line, (margin, y + i * int(40 * font_scale)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            else:
                header_lines = [
                    f"Prediction: {pred_label} ({pred_prob:.2f})",
                    f"Sentence: {sentence}",
                    f"FPS: {fps:.1f}",
                    ("Auto-Append ON" if args.auto_append else "Auto-Append OFF"),
                    ("Speak Mode ON" if args.speak_mode else "Speak Mode OFF"),
                    "Keys: [A]ppend  [B]ackspace  [C]lear  [S]peak sentence  [T]oggle TTS  [Q]uit",
                ]
                put_text_box(frame, header_lines, (10, 10))

            if args.speak_each and tts_engine and pred_label not in ("No Hand", "Unknown"):
                if pred_label != last_spoken_text:
                    tts_engine.say(pred_label)
                    tts_engine.runAndWait()
                    last_spoken_text = pred_label

            cv2.imshow("Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                if pred_label not in ("No Hand", "Unknown"):
                    sentence += pred_label
                    last_prediction.append(pred_label)
            elif key == ord('b'):
                sentence = sentence[:-1]
            elif key == ord('c'):
                sentence = ""
            elif key == ord('s'):
                if tts_engine and sentence:
                    tts_engine.say(sentence)
                    tts_engine.runAndWait()
            elif key == ord('t'):
                if tts_engine is None:
                    tts_engine = init_tts(True)
                else:
                    try:
                        tts_engine.stop()
                    except Exception:
                        pass
                    tts_engine = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
