"""
Wykrzacza sie przy wiecej niz 1 twarzy.
"""
import pygame
import cv2, time
import numpy as np


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
thug = cv2.imread("Thug.png",-1)

wspolczynnik_okulary = 0.95

#testuje dodatkowe opcje
a_thug = False
pygame.mixer.init()
sound_t = pygame.mixer.music.load("thug_life.mp3")
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
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])




while True:
    check, frame = video.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # ponoć w czarnobialym lepiej szuka
    faces=face_cascade.detectMultiScale(gray,
    scaleFactor=1.15, #1.05 to dobry parametr. oznacza zwiekszenie obszaru wyszukiwania o 5% w kaÅ¼dej pÄ™tli
    minNeighbors=5)  # standardowo 5
    #print(faces)

    #print(type(faces))


    if(type(faces).__module__==np.__name__):

        if (not a_thug):
            pygame.mixer.music.play()
            a_thug = True

        for x,y,w,h in faces:
            """
            #zielony kwadrat wokół twarzy
            img=cv2.rectangle(frame, #obraz na ktorym pracujemy
             (x,y), #lewy gÃ³rny rÃ³g
             (x+w,y+h), # prawy dolny rÃ³g
             (0,255,0), #kolor RGB
             3) #grubosc kreski
             """

            resized_thug=cv2.resize(thug,(int(w),int(wspolczynnik_okulary* h)))
            overlay_image_alpha(frame,
                        resized_thug[:, :, 0:3],
                        (x, y),
                        resized_thug[:, :, 3] / 255.0)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #zmieniam kolor na czarnobialy







    else:

        if(a_thug):
            pygame.mixer.music.stop()
            a_thug = False

    time.sleep(0)
    cv2.imshow("Capturing...",frame)


    key = cv2.waitKey(1)
    if (key == ord('q')):
        break
video.release()
cv2.destroyAllWindows()
