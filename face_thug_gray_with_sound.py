import pygame
import cv2, time
import numpy as np


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
thug = cv2.imread("Thug.png", -1)

glasses_coefficient_h = 0.8
glasses_coefficient_w = 1.2

# testuje dodatkowe opcje
a_thug = False
pygame.mixer.init()
sound_t = pygame.mixer.music.load("thug_life.ogg")
print("Press q to quit!")


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (
            alpha * img_overlay[y1o:y2o, x1o:x2o, c] + alpha_inv * img[y1:y2, x1:x2, c]
        )


while True:
    check, frame = video.read()
    frame = cv2.flip(frame, 1)  # mirrored image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,  # 1.05 is a good value to start. It means an increase of 5% with each loop to the area of search.
        minNeighbors=5,
    )  # standard is 5

    if type(faces).__module__ == np.__name__:

        if not a_thug:
            pygame.mixer.music.play()
            a_thug = True

        for x, y, w, h in faces:
            """
            #green square around found face
            img=cv2.rectangle(frame,
             (x,y), #left upper corner
             (x+w,y+h), # right upper corner
             (0,255,0), #RGB color
             3) #line thickness
            """

            resized_thug = cv2.resize(
                thug, (int(glasses_coefficient_w * w), int(glasses_coefficient_h * h))
            )
            glasses_x = int((2 * x + w - glasses_coefficient_w * w) / 2)
            glasses_y = y
            overlay_image_alpha(
                frame,
                resized_thug[:, :, 0:3],
                (glasses_x, glasses_y),
                resized_thug[:, :, 3] / 255.0,
            )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:

        if a_thug:
            pygame.mixer.music.stop()
            a_thug = False

    time.sleep(0)
    cv2.imshow("Capturing...", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
