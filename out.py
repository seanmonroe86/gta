import numpy as np
import cv2, os, time

file_name = 'training_data.npy'


def get_key(keys):
    if keys[0]: return 'A'
    elif keys[1]: return 'W'
    else: return 'D'


def main():
    data = list(np.load(file_name))
    for frame in data[:-200]:
##        print('{}\n{}\n'.format(frame[0][0], frame[1]))
        img = frame[0]
        key = get_key(frame[1])
        cv2.imshow('test', img)
        cv2.waitKey(0)
        print('{}'.format(key))



main()
