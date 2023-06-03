import pytesseract
import cv2
import time
from imutils.object_detection import non_max_suppression
import numpy as np


videoFilePath = '/Users/andersrye/Documents/03349SekEvtQ.mp4'
modelFilePath = '/Users/andersrye/Downloads/frozen_east_text_detection.pb'

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)



net = cv2.dnn.readNet(modelFilePath)

stream = cv2.VideoCapture(videoFilePath, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
(W, H) = (None, None)
(newW, newH) = (160, 160)
(rW, rH) = (None, None)

prev_frame_time = 0
new_frame_time = 0
prev_frame = None
frame_count = 0
while True:
    frame_count += 1
    frame = stream.read()[1]
    if frame is None:
        break
    if frame_count % 10 != 0:
        continue

    orig = frame.copy()

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    (origH, origW) = frame.shape[:2]
    (newW, newH) = (160, 160)
    rW = origW / float(newW)
    rH = origH / float(newH)
    frame = cv2.resize(frame, (newW, newH))
    (H, W) = frame.shape[:2]
    #frame = cv2.UMat(frame)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    #roi = None

    #for (startX, startY, endX, endY) in boxes:
    #    # scale the bounding box coordinates based on the respective
    #    # ratios
    #    startX = int(startX * rW)
    #    startY = int(startY * rH)
    #    endX = int(endX * rW)
    #    endY = int(endY * rH)
    #    # in order to obtain a better OCR of the text we can potentially
    #    # apply a bit of padding surrounding the bounding box -- here we
    #    # are computing the deltas in both the x and y directions
    #    dX = int((endX - startX) * 0.15)
    #    dY = int((endY - startY) * 0.15)
    #    # apply padding to each side of the bounding box, respectively
    #    startX = max(0, startX - dX)
    #    startY = max(0, startY - dY)
    #    endX = min(origW, endX + (dX * 2))
    #    endY = min(origH, endY + (dY * 2))
    #    # extract the actual padded ROI
    #    roi = orig[startY:endY, startX:endX]

    # frame = cv2.UMat(frame)

    if len(boxes) != 0:
        text = pytesseract.image_to_string(orig, config=("-l nor --oem 1 --psm 3"))
        print(text)

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)

    cv2.putText(orig, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
    #print(fps)
    cv2.imshow("Gray", orig)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # trykk q for å avbryte

stream.release()
cv2.destroyAllWindows()
