import numpy as np
import cv2
import time
from tracking import Tracking
from tracking import Blob

from utilities import bigblobKmeans
from utilities import readBuffer, getFrame
from utilities import checkBlobSize, bigblobHorizontal

import matplotlib.pyplot as plt
import pdb


if __name__ == '__main__':
    """input data"""
    # cap = cv2.VideoCapture('191334-vv-1.avi')
    # cap = cv2.VideoCapture('192.168.31.138_01_20160706191100879.avi')
    # cap = cv2.VideoCapture('../../data/192.168.31.138_01_20160706191100879.mp4')

    cap = cv2.VideoCapture('../../data/2016-07-20/4mm-2.65/192.168.0.100_01_20160720171945536.mp4')
    # cap = cv2.VideoCapture('../../data/2016-07-21/3-4mm/192.168.1.145_01_20160721164209992.mp4')
    # cap = cv2.VideoCapture('../../data/2016-07-21/3-4mm/192.168.1.145_01_20160721164044307.mp4')
    # cap = cv2.VideoCapture('../../data/2016-07-21/3-2.8mm/192.168.1.147_01_20160721171223357.mp4')

    startOffset = 377
    cap = readBuffer(startOffset, cap)
    ret, frame = cap.read()

    """parameters"""
    # fgbg = cv2.BackgroundSubtractorMOG2()
    # fgbg = cv2.BackgroundSubtractorMOG2(history=10, varThreshold=500)
    fgbg = cv2.BackgroundSubtractorMOG2(history=20, varThreshold=1000)
    # fgbg = cv2.BackgroundSubtractorMOG2(history=10, varThreshold=200)

    kernelSize = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    # detector = cv2.SimpleBlobDetector()

    scale = 0.5
    output_width  = int(frame.shape[1] * scale)
    output_height = int(frame.shape[0] * scale)
    CODE_TYPE = cv2.cv.CV_FOURCC('m','p','4','v')
    # video = cv2.VideoWriter('output_detection.avi',CODE_TYPE,6,(output_width,output_height),1)
    video = cv2.VideoWriter('output_detection.avi',CODE_TYPE,30,(output_width,output_height),1)

    # areaThreshold = 25 * 25 * 3.14
    countingHalfMargin = 20
    trackingHalfMargin = 50
    countUpperBound = output_height / 2 - countingHalfMargin
    countLowerBound = output_height / 2 + countingHalfMargin
    validTrackUpperBound = output_height / 2 - trackingHalfMargin
    validTrackLowerBound = output_height / 2 + trackingHalfMargin
    distThreshold = 70
    distThreshold_fit = 100 ## bigger threshold for fitted centers
    inactiveThreshold = 10
    trackingObj = Tracking(countUpperBound, countLowerBound, validTrackUpperBound, validTrackLowerBound)
    tracks = []
    totalUp = 0
    totalDown = 0

    ratioUp = 0
    ratioDown = 0

    frameInd = startOffset

    singlePersonBlobSize = 21000
    Visualize = True

    # higherFgPixelList = []
    # midHighFgPixelList = []
    # midLowFgPixelList = []
    # lowerFgPixelList = []
    # horizRatioList = []

    # horizRatioThresh = 0.4  #hallway
    horizRatioThresh = 0.22 #july 20


    while(cap.isOpened()):
        start = time.clock()
        # ret, frame = cap.read()
        frame = getFrame(cap,frameInd)
        if ret == False:
            break

        print 'Frame # %s' % frameInd
        frameInd += 1
        # if frameInd in [400,500,600]:
            # pdb.set_trace()

        # resize image, background subtraction and post-processing of blob
        frame = cv2.resize(frame, (output_width, output_height), interpolation = cv2.INTER_CUBIC)
        fgmask = fgbg.apply(frame)
        ret, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) # THRESH_BINARY, THRESH_TOZERO

        """closing and opening"""
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) 
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)


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
        maxArea = 0

        for cnt in contours:
            maxArea = max(cv2.contourArea(cnt), maxArea)



        if len(horizRatioList)>0:
            if frameInd-np.array(horizRatioList)[-1,2]>inactiveThreshold:
                horizRatioList = []

        if maxArea >= singlePersonBlobSize*2: # big blob exists
            n_clusters = np.int(np.sum(fgmask!=0)/singlePersonBlobSize)  ##fit the whole frame
            centroidList = bigblobKmeans(frame, fgmask, n_clusters)
            for nn in range(n_clusters):
                center = (int(centroidList[nn][0]), int(centroidList[nn][1]))
                blobs.append(Blob((int(center[0]), int(center[1])), l, l + w, u, u + h))
                if Visualize:
                    cv2.putText(frame, str(center), center, font, 0.5, (255,255,255), 1)
                    cv2.putText(frame, str(n_clusters), (5, output_height - 50), font, 0.5, (255,255,255), 1) #show number of fitted clusters

            # tracking
            tracks, nUp, nDown = trackingObj.updateTrack(blobs, tracks, distThreshold_fit, inactiveThreshold)

        else: # no big blob in this frame
            for cnt in contours:
                if cv2.contourArea(cnt) < 5000:
                    continue
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                l,u,w,h = cv2.boundingRect(cnt)
                blobs.append(Blob((int(x), int(y)), l, l + w, u, u + h))

                if Visualize:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    # cv2.ellipse(frame,ellipse,(0,255,255),2)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(center), center, font, 1, (0,255,255), 1)

            # tracking
            tracks, nUp, nDown = trackingObj.updateTrack(blobs, tracks, distThreshold, inactiveThreshold)
            
            # ratioUp += nUp
            # ratioDown += nDown

        totalUp += nUp
        totalDown += nDown
        print '# UP %s' % totalUp
        print '# DOWN %s' % totalDown


        """horizontal ratio"""
        horizRatio = bigblobHorizontal(fgmask, validTrackUpperBound, validTrackLowerBound)
        peak = np.max(horizRatio)
        if peak>0.1:
            peakLoc = np.where(horizRatio==peak)[0]
            peakLoc = np.sum(peakLoc)/len(peakLoc)
            horizRatioList.append((peak, peakLoc, frameInd))

        if len(horizRatioList)>2:

            # horizRatioList = np.vstack( (np.array(horizRatioList)[(np.array(horizRatioList)[:,2][1:]-np.array(horizRatioList)[:,2][:-1])==1,:] /
            # ,np.array(horizRatioList)[-1,:]))

            if (validTrackUpperBound+np.array(horizRatioList)[0,1]> countLowerBound) and (validTrackUpperBound+peakLoc < countUpperBound): 
                # ratioUp += 1*round(np.max(np.array(horizRatioList)[:,0])/0.4)
                slope, intercep = np.polyfit(np.array(horizRatioList)[:,2], np.array(horizRatioList)[:,1], deg = 1)
                # print slope, intercep
                # pdb.set_trace()
                if slope<-2 and intercep>1000:
                    ratioUp += 1*round(np.mean(np.array(horizRatioList)[:,0])/horizRatioThresh)
                    horizRatioList =[] ## treated as counted 
                
            elif (validTrackUpperBound+np.array(horizRatioList)[0,1]< countUpperBound) and (validTrackUpperBound+peakLoc>countLowerBound):
                # ratioDown += 1*round(np.max(np.array(horizRatioList)[:,0])/0.4)
                # pdb.set_trace()
                slope, intercep = np.polyfit(np.array(horizRatioList)[:,2], np.array(horizRatioList)[:,1], deg = 1)
                # print slope, intercep
                if slope>2 and intercep<-10:
                    ratioDown += 1*round(np.mean(np.array(horizRatioList)[:,0])/horizRatioThresh)
                    horizRatioList =[] ## treated as counted 


        # Visualize tracking region, counting region and tracks
        if Visualize:
            cv2.line(frame, (0, validTrackUpperBound), (output_width - 1, validTrackUpperBound), (255, 0, 0), 2)
            cv2.line(frame, (0, validTrackLowerBound), (output_width - 1, validTrackLowerBound), (255, 0, 0), 2)
            cv2.line(frame, (0, countUpperBound), (output_width - 1, countUpperBound), (0, 0, 255), 2)
            cv2.line(frame, (0, countLowerBound), (output_width - 1, countLowerBound), (0, 0, 255), 2)
            cv2.putText(frame, '# UP %s' % totalUp, (5, output_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(frame, '# DOWN %s' % totalDown, (5, output_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            cv2.putText(frame, '# UP %s' % (ratioUp), (output_width-300, output_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(frame, '# DOWN %s' % (ratioDown), (output_width-300, output_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            for idxTrack, track in enumerate(tracks):
                track.plot(frame)
                # track.printTrack()

        end = time.clock()
        print('fps: {}'.format(1 / (end - start)))

        if Visualize:
            cv2.imshow('frame',frame)
            video.write(frame)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    cap.release()
    video.release()
    cv2.destroyAllWindows()
