import numpy as np
import cv2
import time
from tracking import Tracking
from tracking import Blob
# import SimpleCV

from utilities import bigblobKmeans
from utilities import readBuffer, getFrame
from utilities import checkBlobSize, getBlobRatio

import matplotlib.pyplot as plt
import pdb


if __name__ == '__main__':
    """input data"""
    # 191334-vv-1, 190645-vv-1
    # cap = cv2.VideoCapture('191334-vv-1.avi')
    # cap = cv2.VideoCapture('192.168.31.138_01_20160706191100879.avi')
    # cap = cv2.VideoCapture('../../data/192.168.31.138_01_20160706191100879.mp4')

    cap = cv2.VideoCapture('../../data/2016-07-20/4mm-2.65/192.168.0.100_01_20160720171945536.mp4')
    # cap = cv2.VideoCapture('../../data/2016-07-21/3-4mm/192.168.1.145_01_20160721164209992.mp4')
    # cap = cv2.VideoCapture('../../data/2016-07-21/3-4mm/192.168.1.145_01_20160721164044307.mp4')
    # cap = cv2.VideoCapture('../../data/2016-07-21/3-2.8mm/192.168.1.147_01_20160721171223357.mp4')

    startOffset = 400
    cap = readBuffer(startOffset, cap)
    ret, frame = cap.read()

    """parameters"""
    # fgbg = cv2.BackgroundSubtractorMOG2()
    fgbg = cv2.BackgroundSubtractorMOG2(history=20, varThreshold=1000)
    # fgbg = cv2.BackgroundSubtractorMOG2(history=10, varThreshold=200)

    kernelSize = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    # detector = cv2.SimpleBlobDetector()

    scale = 0.5
    output_width  = int(frame.shape[1] * scale)
    output_height = int(frame.shape[0] * scale)
    CODE_TYPE = cv2.cv.CV_FOURCC('m','p','4','v')
    video = cv2.VideoWriter('output_detection.avi',CODE_TYPE,30,(output_width,output_height*2),1)

    areaThreshold = 25 * 25 * 3.14
    countingHalfMargin = 20
    trackingHalfMargin = 40 #20
    countUpperBound = output_height / 2 - countingHalfMargin
    countLowerBound = output_height / 2 + countingHalfMargin
    validTrackUpperBound = output_height / 2 - trackingHalfMargin
    validTrackLowerBound = output_height / 2 + trackingHalfMargin
    distThreshold = 100#70
    distThreshold_fit = 100 ## bigger threshold for fitted centers
    inactiveThreshold = 10
    peopleBlobSize = 100
    trackingObj = Tracking(countUpperBound, countLowerBound, validTrackUpperBound, validTrackLowerBound, peopleBlobSize)
    tracks = []
    totalUp = 0
    totalDown = 0

    ratioUp = 0
    ratioDown = 0

    frameInd = startOffset

    singlePersonBlobSize = 10000
    Visualize = True
    useKmeans = False
    # higherFgPixelList = []
    # midHighFgPixelList = []
    # midLowFgPixelList = []
    # lowerFgPixelList = []

    while(cap.isOpened()):
        start = time.clock()
        # ret, frame = cap.read()
        frame = getFrame(cap,frameInd)
        if ret == False:
            break

        print 'Frame # %s' % frameInd
        frameInd += 1

        # resize image, background subtraction and post-processing of blob
        frame = cv2.resize(frame, (output_width, output_height), interpolation = cv2.INTER_CUBIC)
        fgmask = fgbg.apply(frame,0.001)
        ret, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) # THRESH_BINARY, THRESH_TOZERO
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        maskedFrame = cv2.bitwise_and(frame, frame, mask = fgmask)


        """check blob size in differnt regions"""
        # (higherFgPixel, midHighFgPixel, midLowFgPixel, lowerFgPixel) = checkBlobSize(fgmask)
        # higherFgPixelList.append(higherFgPixel)
        # midHighFgPixelList.append(midHighFgPixel)
        # midLowFgPixelList.append(midLowFgPixel)
        # lowerFgPixelList.append(lowerFgPixel)


        # find blobs
        contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center = None
        blobs = []
        

        
        if useKmeans:
            maxArea = 0
            for cnt in contours:
                if cv2.contourArea(cnt) < areaThreshold:
                    continue
                maxArea = max(cv2.contourArea(cnt), maxArea)

            if maxArea >= singlePersonBlobSize*2: # big blob exists
                n_clusters = np.int(np.sum(fgmask!=0)/singlePersonBlobSize)  ##fit the whole frame
                centroidList = bigblobKmeans(frame, fgmask, n_clusters)
                for nn in range(n_clusters):
                    center = (int(centroidList[nn][0]), int(centroidList[nn][1]))
                    blobs.append(Blob((int(center[0]), int(center[1])), l, l + w, u, u + h))
                    if Visualize:
                        cv2.putText(frame, str(center), center, font, 0.5, (255,255,255), 1)
                        cv2.putText(frame, 'kmeans %s clusters' % (n_clusters), (5, output_height - 50), font, 0.5, (255,255,255), 1) #show number of fitted clusters

                # tracking
                tracks, nUp, nDown = trackingObj.updateAllTrack(blobs, tracks, distThreshold_fit, inactiveThreshold)

            else: # no big blob in this frame
                for cnt in contours:
                    if cv2.contourArea(cnt) < 5000:
                        continue
                    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                    l,u,w,h = cv2.boundingRect(cnt)
                    # blobs.append(Blob((int(x), int(y)), l, l + w, u, u + h))
                    temp = np.zeros_like(fgmask)
                    temp[u:u+h,l:l+w] =1
                    blobmask = fgmask*temp
                    (peak, peakLoc) = getBlobRatio(blobmask, validTrackUpperBound, validTrackLowerBound)
                    blobs.append(Blob((int(x), int(y)), l, l + w, u, u + h, peak, peakLoc, frameInd))

                    if Visualize:
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        # cv2.ellipse(frame,ellipse,(0,255,255),2)
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, str(center), center, font, 1, (0,255,255), 1)
                
                # tracking
                tracks, nUp, nDown = trackingObj.updateAllTrack(blobs, tracks, distThreshold, inactiveThreshold)


        else: ##dont' distinguish big or small blobs, no kmeans
            for cnt in contours:
                if cv2.contourArea(cnt) < 5000:
                    continue
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                l,u,w,h = cv2.boundingRect(cnt)
                # blobs.append(Blob((int(x), int(y)), l, l + w, u, u + h))
                temp = np.zeros_like(fgmask)
                temp[u:u+h,l:l+w] =1
                blobmask = fgmask*temp
                (peak, peakLoc) = getBlobRatio(blobmask, validTrackUpperBound, validTrackLowerBound)
                blobs.append(Blob((int(x), int(y)), l, l + w, u, u + h, peak, peakLoc, frameInd))

                if Visualize:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    # cv2.ellipse(frame,ellipse,(0,255,255),2)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(center), center, font, 1, (0,255,255), 1)

            # tracking
            # tracks, nUp, nDown = trackingObj.updateAllTrack(blobs, tracks, distThreshold, inactiveThreshold)
            (tracks, nUp, nDown, nUp_ratio, nDown_ratio) = trackingObj.updateAllTrack(blobs, tracks, distThreshold, inactiveThreshold)
            totalUp += nUp
            totalDown += nDown
            ratioUp += nUp_ratio
            ratioDown += nDown_ratio
            # print '# UP %s' % totalUp
            # print '# DOWN %s' % totalDown


        if Visualize:
            cv2.line(frame, (0, validTrackUpperBound), (output_width - 1, validTrackUpperBound), (255, 0, 0), 2)
            cv2.line(frame, (0, validTrackLowerBound), (output_width - 1, validTrackLowerBound), (255, 0, 0), 2)
            cv2.line(frame, (0, countUpperBound), (output_width - 1, countUpperBound), (0, 0, 255), 2)
            cv2.line(frame, (0, countLowerBound), (output_width - 1, countLowerBound), (0, 0, 255), 2)
            cv2.putText(frame, '# UP %s' % totalUp, (5, output_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(frame, '# DOWN %s' % totalDown, (5, output_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            cv2.putText(frame, '# ratio UP %s' % (ratioUp), (output_width-400, output_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(frame, '# ratio DOWN %s' % (ratioDown), (output_width-400, output_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            for idxTrack, track in enumerate(tracks):
                track.plot(frame)
                # track.printTrack()

        end = time.clock()
        # print('fps: {}'.format(1 / (end - start)))

        if Visualize:
            maskedFrame = np.vstack((frame, maskedFrame))
            cv2.imshow('frame', maskedFrame)
            video.write(maskedFrame)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    cap.release()
    video.release()
    cv2.destroyAllWindows()
