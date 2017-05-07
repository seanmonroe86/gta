import numpy as np
import cv2, os, time
from grabscreen import grab_screen
from getkeys import key_check
from alexnet import alexnet
from direct import PressKey, ReleaseKey, W, A, S, D


WIDTH = 80
HEIGHT = 64
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)
window_size = (0, 40, 800, 640)
file_name = 'training_data.npy'




def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)
##    ReleaseKey(A)

def right():
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)
##    ReleaseKey(D)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)
    

def main():
    for i in range(2, 0, -1):
        print(i+1)
        time.sleep(1)
        time_start = time.time()

    paused = False
    while(True):

        if not paused:
            screen = grab_screen(region = (window_size))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80, 64))

            print("Time spent: {} seconds".format(time.time() - time_start))
            time_start = time.time()

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
            moves = list(np.around(prediction))
            print(moves, prediction)

            if moves == [1, 0, 0]:
                left()
            elif moves == [0, 1, 0]:
                straight()
            elif moves == [0, 0, 1]:
                right()
            else:
                print('Bad move')

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)



main()
