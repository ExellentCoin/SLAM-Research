#! /usr/bin/env python3

import numpy as np
import cv2 as cv

vid = cv.VideoCapture('./videos/table_resized_muted.mp4')
screen_res = 640, 480

frames = []

orb = cv.ORB_create()
bf = cv.BFMatcher()

while(vid.isOpened()):
    ret, frame = vid.read()

    

    # Stop displaying if there is no frame (when ret is false)
    if not ret:
        break

    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate the scale and resize the window using it
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    dim = (window_width, window_height)

    frame = cv.resize(frame,dim , dim, interpolation = cv.INTER_CUBIC);

    frames.append(frame)

    if len(frames) == 1:
        kp1, des1 = orb.detectAndCompute(frame, None)
        frame = cv.drawKeypoints(frame, kp1, None, color=(0, 255, 0))
    else:
        kp1, des1 = orb.detectAndCompute(frame, None)
        kp2, des2 = orb.detectAndCompute(frames[-2], None)

        # frame = cv.drawKeypoints(frame, kp2, None, color=(0, 0, 255))

        matches = bf.knnMatch(des1, des2, k=2)
        # matches = sorted(matches, key=lambda x: x.distance)

        good = []

        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])


        #Use KnnMatcher
        for i in range(len(good)):
            print(good)
            trainIdx = good[i].trainIdx
            queryIdx = good[i].queryIdx
            print(len(kp1), trainIdx, queryIdx)
            pt1 = int(kp1[queryIdx].pt[0]), int(kp1[queryIdx].pt[1])
            pt2 = int(kp2[trainIdx].pt[0]), int(kp2[trainIdx].pt[1])
            cv.circle(frame, pt1, 3, (0, 255, 0))
            cv.circle(frame, pt2, 3, (0, 0, 255))
            cv.line(frame, pt1, pt2, (255, 0, 0), thickness=1)

    cv.imshow('video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
