import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

kimwant_bgr = cv2.imread('img/kimsunho.png')
kimwant = cv2.cvtColor(kimwant_bgr, cv2.COLOR_BGR2RGB)

faces = detector(kimwant)
for face in faces:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    landmarks = landmark_predictor(kimwant, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(kimwant, (x, y), 4, (255, 0, 0), -1)

    face_descriptor = face_recognizer.compute_face_descriptor(kimwant,landmarks)
    print(face_descriptor)
image = cv2.cvtColor(kimwant, cv2.COLOR_RGB2BGR)
cv2.imwrite('image.jpg', image)
