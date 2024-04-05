import cv2
import os


#video_path = 'C:/User/.mp4'
save_dir = 'C:/Users/user/pythonProject/modelling_research/database'
interval = 20  # fps

def extract_frames(video_path, save_dir, interval):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_save_path = os.path.join(save_dir, f"0001({count}).jpg")
            cv2.imwrite(frame_save_path, frame)  # Сохраняем кадр
        count += 1
    cap.release()

extract_frames(video_path, save_dir, interval)
