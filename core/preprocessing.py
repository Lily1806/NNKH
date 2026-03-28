import cv2
import numpy as np
import os
import glob
import mediapipe as mp
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config

mp_holistic = mp.solutions.holistic


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, lh, rh])


def process_video_file(video_path, max_frames=Config.MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None

    frames_data = []

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                             min_tracking_confidence=0.5) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            frames_data.append(keypoints)

    cap.release()

    if len(frames_data) == 0:
        return None

    frames_data = np.array(frames_data)

    # resize về max_frames
    if len(frames_data) > max_frames:
        idx = np.linspace(0, len(frames_data)-1, max_frames).astype(int)
        frames_data = frames_data[idx]
    else:
        pad = np.zeros((max_frames - len(frames_data), Config.KEYPOINT_DIM))
        frames_data = np.vstack([frames_data, pad])

    return frames_data


def preprocess_split(split_name, raw_dir, processed_dir):
    if not os.path.exists(raw_dir):
        print(f"⚠️ Skip {split_name} (no folder)")
        return

    print(f"\n🚀 Processing {split_name}...")

    classes = [d for d in os.listdir(raw_dir)
               if os.path.isdir(os.path.join(raw_dir, d))]

    total = 0

    for cls in classes:
        cls_raw = os.path.join(raw_dir, cls)
        cls_out = os.path.join(processed_dir, cls)

        os.makedirs(cls_out, exist_ok=True)

        videos = glob.glob(os.path.join(cls_raw, "*.mp4"))

        if len(videos) == 0:
            print(f"⚠️ No video in {cls}")
            continue

        for video in videos:
            name = os.path.basename(video).split('.')[0]
            save_path = os.path.join(cls_out, f"{name}.npy")

            if os.path.exists(save_path):
                continue

            print(f"Processing {video}")
            data = process_video_file(video)

            if data is None:
                print(f"❌ Skip {video}")
                continue

            np.save(save_path, data)
            total += 1

    print(f"✅ Done {split_name}: {total} files")


def preprocess_all_data():
    Config.setup_directories()

    if not os.path.exists(Config.LABEL_MAPPING_PATH):
        print("❌ Missing label_mapping.pkl")
        return

    splits = [
        ("train", Config.DATA_RAW_TRAIN, Config.DATA_PROCESSED_TRAIN),
        ("public_test", Config.DATA_RAW_PUBLIC_TEST, Config.DATA_PROCESSED_PUBLIC_TEST),
        ("private_test", Config.DATA_RAW_PRIVATE_TEST, Config.DATA_PROCESSED_PRIVATE_TEST)
    ]

    for name, raw, out in splits:
        preprocess_split(name, raw, out)

    print("\n🎯 ALL DONE")


if __name__ == "__main__":
    preprocess_all_data()