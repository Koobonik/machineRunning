import dlib
import cv
import cv2
import numpy as np

if __name__ == '__main__':
    video_path = 'data/startup_cut.mov'

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('output/output.mp4', fourcc, 30, (1920, 1080))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, 'text writing test', org=(600, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0), thickness=5)
        writer.write(frame)
        # cv2.imshow('img', frame)

    cap.release()
    writer.release()
