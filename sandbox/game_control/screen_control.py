import cv2
import numpy as np
import pyautogui
import time
import logging
import threading
from pynput.keyboard import Key, Listener


screen_size = (1920, 1080)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.00, screen_size)

fps = 30
prev = 0


start = time.time()


def on_press(key):
    print("{0}:{1} pressed".format(time.time() - start, key))


def on_release(key):
    if key == Key.esc:
        return False


def record_screen(idk):
    while True:
        time_elpased = time.time() - 0
        img = pyautogui.screenshot()

        if time_elpased > 1.0 / fps:
            prev = time.time()
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)

        cv2.waitKey(27)

    cv2.destroyAllWindows()
    out.release()


recorder = threading.Thread(target=record_screen, args=(0,))
recorder.start()

with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

recorder.join()


