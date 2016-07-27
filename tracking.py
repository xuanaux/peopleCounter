import cv2
import numpy as np
import pdb



colors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0), (0,128,128), (128,0,128), (128,128,0)]

class Blob(object):
    def __init__(self, center, minx, maxx, miny, maxy, peak, peakLoc, frameInd):
        self.center = center
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

        ## horizontal foreground pixel ratio and ratio peak
        self.peak = peak
        self.peakLoc = peakLoc
        self.frameInd = frameInd


class Track(object):
    def __init__(self, direction, color):
        self.centerList = []
        self.lifetime = 0
        self.inactiveCount = 0  #frame number staying inactive
        self.activeCount = 0  
        self.direction = direction  ##the blob's location
        self.generalDirection = None
        self.counted = False
        self.color = color
        # self.minx = np.nan
        # self.maxx = np.nan
        # self.miny = np.nan
        # self.maxy = np.nan

        self.lifeStart = np.nan
        self.peak = []
        self.peakLoc = []
    def updateTrack(self, blob):
        self.centerList.append(blob.center)
        self.minx = blob.minx
        self.maxx = blob.maxx
        self.miny = blob.miny
        self.maxy = blob.maxy
        # self.minx = np.nanmin((self.minx, blob.minx))
        # self.maxx = np.nanmax((self.maxx, blob.maxx))
        # self.miny = np.nanmin((self.miny, blob.miny))
        # self.maxy = np.nanmax((self.maxy, blob.maxy))

        if blob.peak>0.1:
            self.peak.append(blob.peak)
            self.peakLoc.append(blob.peakLoc)

        self.lifeStart = int(np.nanmin((self.lifeStart, blob.frameInd)))
        self.lifeEnd = int(blob.frameInd)
        
        self.inactiveCount = 0
        self.activeCount += 1

    def predictCenter(self):
        # return self.centerList[-1]
        if len(self.centerList)==1:
            return self.centerList[-1]
        elif len(self.centerList)==2:
            return np.dot([0.4,0.6],(self.centerList[-2:]))
        else:
            return np.dot([0.1,0.4,0.5],(self.centerList[-3:]))

    def plot(self, img):
        for i in reversed(xrange(len(self.centerList) - 1)):
            cv2.line(img, self.centerList[i + 1], self.centerList[i], self.color)
            cv2.circle(img, self.centerList[i + 1], 2, self.color, -1)

    def printTrack(self):
        for i in reversed(xrange(len(self.centerList) - 1)):
            print self.centerList[i],
        print ' '


    def fitTracklet(self,upperH,lowerH):
        """estimate direction"""
        if ((np.array(self.centerList)[:,1][1:]-np.array(self.centerList)[:,1][:-1])>=0).sum()/float(len(self.centerList))>=0.70:
            peopleDirection = 1 ## downward
        elif ((np.array(self.centerList)[:,1][1:]-np.array(self.centerList)[:,1][:-1])<=0).sum()/float(len(self.centerList))>=0.70:
            peopleDirection = 2 # upward
        elif np.mean(np.array(self.centerList)[:,1][-3:])< lowerH and np.mean(np.array(self.centerList)[:,1][:3])>upperH:
            peopleDirection = 2 # upward
        elif np.mean(np.array(self.centerList)[:,1][:3])< lowerH and np.mean(np.array(self.centerList)[:,1][-3:])>upperH:
            peopleDirection = 1 # downward
        else:
            peopleDirection = 3 # odd cases

        self.generalDirection = peopleDirection
        self.counted = True


    def fitHorizontalRatio(self, validTrackUpperBound,countUpperBound, countLowerBound):
        ratioUp,ratioDown = 0,0
        # horizRatioThresh = 0.4  #hallway
        horizRatioThresh = 0.2 #july 20

        if (validTrackUpperBound+self.peakLoc[0]> countLowerBound) and (validTrackUpperBound+self.peakLoc[-1] < countUpperBound): 
            # ratioUp += 1*round(np.max(np.array(horizRatioList)[:,0])/0.4)
            slope, intercep = np.polyfit(range(len(self.peakLoc)), self.peakLoc, deg = 1)
            print slope, intercep
            # pdb.set_trace()
            if slope<-2:
                ratioUp += 1*round(np.max(self.peak)/horizRatioThresh)
                self.counted = True

        elif (validTrackUpperBound+self.peakLoc[0]< countUpperBound) and (validTrackUpperBound+self.peakLoc[-1]>countLowerBound):
            # ratioDown += 1*round(np.max(np.array(horizRatioList)[:,0])/0.4)
            slope, intercep = np.polyfit(range(len(self.peakLoc)),self.peakLoc, deg = 1)
            print slope, intercep
            # pdb.set_trace()
            if slope>2:
                ratioDown += 1*round(np.max(self.peak)/horizRatioThresh)
                self.counted = True

        return (ratioUp,ratioDown)


class Tracking(object):
    def __init__(self, countUpperBound, countLowerBound, validTrackUpperBound, validTrackLowerBound, peopleBlobSize):
        self.countUpperBound = countUpperBound
        self.countLowerBound = countLowerBound
        self.validTrackUpperBound = validTrackUpperBound
        self.validTrackLowerBound = validTrackLowerBound
        self.peopleBlobSize = peopleBlobSize
        self.counter = 0


    """updateAllTracks"""
    def updateAllTrack(self, blobs, tracks, distThreshold, inactiveThreshold):
        # print tracks
        nBlob = len(blobs)
        nTrack = len(tracks)
        distMatrix = np.zeros((nTrack, nBlob))
        blobMark = np.zeros(nBlob) #whether blob is assigned 
        trackMark = np.zeros(nTrack)  #whether tracklet is assigned with a blob in current frm
        for idxBlob, blob in enumerate(blobs):
            for idxTrack, track in enumerate(tracks):
                distMatrix[idxTrack, idxBlob] = self.distBlobTrack(blob, track)

        nAssignedBlob = 0
        for idxBlob, blob in enumerate(blobs):
            minDist = 10000
            minIdxTrack = 0
            closestTrack = None
            # for each blob, find the closest track < distThreshold. 
            # if the track is not picked yet, assign blob to the track
            for idxTrack, track in enumerate(tracks):
                if distMatrix[idxTrack, idxBlob] < minDist:
                    minDist = distMatrix[idxTrack, idxBlob]
                    minIdxTrack = idxTrack
                    closestTrack = track

            if minDist < distThreshold and trackMark[minIdxTrack] == 0:
                # print minDist
                closestTrack.updateTrack(blob)
                trackMark[minIdxTrack] = 1
                blobMark[idxBlob] = 1
                nAssignedBlob += 1

        # print 'Assigned blob: %s' % nAssignedBlob

        # for not assigned blob, determine if it is a valid new track
        newTracks = []
        for idxBlob, blob in enumerate(blobs):
            if blobMark[idxBlob] == 1:
                continue
            direction = self.appearRegion(blob)
            if direction != 0:
                newTrack = Track(direction, colors[self.counter % len(colors)])
                newTrack.updateTrack(blob)
                newTracks.append(newTrack)
                self.counter += 1

        nUp = 0
        nDown = 0
        nDown_ratio = 0
        nUp_ratio = 0
        for idxTrack, track in enumerate(tracks):
            # for not assigned track, increment inactive states, delete if more than inactiveThreshold
            if trackMark[idxTrack] == 0:
                track.activeCount = 0
                track.inactiveCount += 1
                if track.inactiveCount >= inactiveThreshold:
                    trackMark[idxTrack] = 2
            # if track has passed detection region, increment up/down counter, then inactivate track
            else:
                if track.direction == -1 and track.centerList[-1][1] > self.countLowerBound:
                    nDown += max(1, int((track.maxx - track.minx) / self.peopleBlobSize + 0.5))
                    track.direction = 1
                    # track.counted = True
                elif track.direction == 1 and track.centerList[-1][1] < self.countUpperBound:
                    nUp += max(1, int((track.maxx - track.minx) / self.peopleBlobSize + 0.5))
                    track.direction = -1
                    # track.counted = True
           

            # elif not track.counted:
            #     if (np.min(np.array(track.centerList)[:,1])<= self.validTrackUpperBound) and (np.max(np.array(track.centerList)[:,1])>=self.validTrackLowerBound):
            #         track.fitTracklet(self.validTrackUpperBound,self.validTrackLowerBound)
            #         if track.generalDirection==1:
            #             nDown += 1
            #         elif track.generalDirection==2:
            #             nUp += 1

            if not track.counted:
                if (np.min(np.array(track.centerList)[:,1])<= self.countUpperBound) and (np.max(np.array(track.centerList)[:,1])>=self.countLowerBound):
                # if (track.direction == -1 and track.centerList[-1][1] > self.countLowerBound) or (track.direction == 1 and track.centerList[-1][1] < self.countUpperBound):
                    (ratioUp,ratioDown) = track.fitHorizontalRatio(self.validTrackUpperBound, self.countUpperBound, self.countLowerBound)
                    nDown_ratio += ratioDown
                    nUp_ratio += ratioUp


        tracks = [tracks[i] for i in range(len(tracks)) if trackMark[i] <= 1]
        # append new tracks to track list
        tracks.extend(newTracks)
        return (tracks, nUp, nDown, nUp_ratio, nDown_ratio)

    def distBlobTrack(self, blob, track):
        predictCenter = track.predictCenter()
        dist = np.linalg.norm(np.array(blob.center) - np.array(predictCenter))
        # print blob.center, predictCenter, dist
        return dist


    def appearRegion(self, blob):
        """return non-zero values if within the detection region"""
        if blob.center[1] < self.validTrackUpperBound:
            return -1
        elif blob.center[1] > self.validTrackLowerBound:
            return 1
        else:
            return 0
