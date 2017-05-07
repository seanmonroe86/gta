import numpy as np
import cv2, os, time
from grabscreen import grab_screen
from getkeys import key_check
def keys_to_output(keys):
    # [A,W,D]
    output = [0, 0, 0]
    if 'A' in keys: output[0] = 1
    elif 'D' in keys: output[2] = 1
    else: output[1] = 1
    return output
window_size = (0, 40, 800, 640)
file_name = 'training_data.npy'
if os.path.isfile(file_name):
    print('File exists, loading previous data.')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh.')
    training_data = []
def main():
    for i in [3, 2, 1]:
        print(i)
        time.sleep(1)
    try:
        time_per = time.time()
        while(True):
            screen = grab_screen(region = (window_size))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80, 64))
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])
            time_start = time.time()
            if len(training_data) % 3000 == 0:
                time_per = time.time() - time_per
                print('Saving...')
                np.save(file_name, training_data)
                print('{}: {}'.format(len(training_data), time_per))
                time_per = time.time()
    except KeyboardInterrupt:
        print('Saving...')
        np.save(file_name, training_data[:-100])
        print('{}'.format(len(training_data)))
main()
