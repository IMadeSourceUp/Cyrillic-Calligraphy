import os
import sys
import time
import torch
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.resolve()) + '/src')
from const import DIR, WEIGHTS_PATH, PREDICT_PATH
from config import MODEL, ALPHABET, N_HEADS, ENC_LAYERS, DEC_LAYERS, DEVICE, HIDDEN
from utils import prediction
from models import model2

char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
idx2char = {idx: char for idx, char in enumerate(ALPHABET)}
model = model2.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,
                                nhead=N_HEADS, dropout=0.0).to(DEVICE)
if WEIGHTS_PATH != None:
    print(f'loading weights from {WEIGHTS_PATH}')
    model.load_state_dict(torch.load(WEIGHTS_PATH))
import numpy as np
import cv2 as cv


while True:
    # Capture frame-by-frame
    ret, frame = cv.imread('home.png')
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    ret, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 40))

    erosion = cv.erode(thresh1, rect_kernel, iterations=2)

    # Finding contours
    contours, hierarchy = cv.findContours(erosion, cv.RETR_LIST,
                                          cv.CHAIN_APPROX_SIMPLE)

    im2 = gray.copy()
    time.sleep(0.66)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cropped = im2[y:y + h, x:x + w]
        rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imshow('frame', im2)
        if contours:
            cropped = cv.imwrite('./predict/temp.png', cropped)
            preds = prediction(model, PREDICT_PATH, char2idx, idx2char)
            for item in preds.items():
                print(f'Предполагаемый текст: {item[1]}')
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
