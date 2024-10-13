import cv2
import os

video_path = ('D:/Video/20240701_103031.mp4', 'D:/Video/20240701_104425.mp4', 'D:/Video/20240701_123431.mp4',
              'D:/Video/20240701_153417.mp4', 'D:/Video/20240701_183810.mp4', 'D:/Video/20240701_183810.mp4',
              'D:/Video/20240702_083325.mp4', 'D:/Video/20240702_083355.mp4', 'D:/Video/20240702_083529.mp4',
              'D:/Video/20240702_083637.mp4', 'D:/Video/20240702_083706.mp4', 'D:/Video/20240702_120944.mp4',
              'D:/Video/20240702_121319.mp4', 'D:/Video/20240702_121439.mp4', 'D:/Video/20240702_123859.mp4',
              'D:/Video/20240702_180347.mp4', 'D:/Video/20240702_165837.mp4', 'D:/Video/20240702_133815.mp4',
              'D:/Video/20240702_184317.mp4', 'D:/Video/20240703_103304.mp4', 'D:/Video/20240703_104512.mp4',
              'D:/Video/20240704_102659.mp4', 'D:/Video/20240703_144129.mp4', 'D:/Video/20240703_105100.mp4',
              'D:/Video/20240704_110650.mp4', 'D:/Video/20240704_110722.mp4', 'D:/Video/20240704_110754.mp4',
              'D:/Video/20240704_111038.mp4', 'D:/Video/20240704_111014.mp4', 'D:/Video/20240704_110845.mp4',
              'D:/Video/20240704_111104.mp4', 'D:/Video/20240704_112810.mp4', 'D:/Video/20240704_130036.mp4',
              'D:/Video/20240704_165647.mp4', 'D:/Video/20240704_153411.mp4', 'D:/Video/20240704_153156.mp4',
              'D:/Video/20240704_170430.mp4', 'D:/Video/20240704_185906.mp4', 'D:/Video/20240704_190150.mp4',
              'D:/Video/20240705_114017.mp4', 'D:/Video/20240705_124929.mp4', 'D:/Video/20240705_125110.mp4',
              'D:/Video/20240705_161212.mp4', 'D:/Video/20240705_182344.mp4', 'D:/Video/20240705_182949.mp4',
              'D:/Video/20240705_205207.mp4', 'D:/Video/20240705_191507.mp4', 'D:/Video/20240705_184340.mp4',
              'D:/Video/20240705_205547.mp4', 'D:/Video/20240706_133657.mp4', 'D:/Video/video_2024-07-22_10-50-01.mp4',
              'D:/Video/video_2024-07-22_10-52-13.mp4', 'D:/Video/video_2024-07-22_10-50-10.mp4')

save_dir = 'C:/Users/user/pythonProject/modelling_YOLOv8/3rd_database_png'
interval = 20  # fps


def extract_frames(video_path, save_dir, interval, photo_number):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    count = 0
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_save_path = os.path.join(save_dir, f"3rd({count+photo_number}).png")
            print(f"created {frame_save_path}")
            cv2.imwrite(frame_save_path, frame)  # Сохраняем кадр
        count += 1
    cap.release()


photo_number = 0
i = 0
while True:
    i += 1
    print(photo_number)
    photo_number += 100000
    extract_frames(video_path[i], save_dir, interval, photo_number)
