import cv2
import numpy as np


class Matchers(object):
    def __init__(self, sift=True):
        # feature detector, sift or surf
        if sift:
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.surf = cv2.xfeatures2d.SURF_create()

        # matcher
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, img1, img2, direction=None):
        if self.sift is None:
            imageSet1 = self.getSURFFeatures(img1)
            imageSet2 = self.getSURFFeatures(img2)
        else:
            imageSet1 = self.getSIFTFeatures(img1)
            imageSet2 = self.getSIFTFeatures(img2)

        print("Direction : ", direction)
        matches = self.flann.knnMatch(imageSet2['des'], imageSet1['des'], k=2)
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']

            matchedPointsCurrent = np.float32(
                [pointsCurrent[i].pt for (__, i) in good]
            )
            matchedPointsPrev = np.float32(
                [pointsPrevious[i].pt for (i, __) in good]
            )

            #cv2.drawKeypoints(img2, pointsCurrent, img2)
            #cv2.imshow("xx", img2)
            #cv2.waitKey()


            #cv2.drawMatchesKnn(img1, pointsPrevious, img2, pointsCurrent, good, None, flags=2)
            #cv2.drawMatches(img1, pointsPrevious, img2, pointsCurrent, good)


            H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
            return H
        return None

    def getSURFFeatures(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

    def getSIFTFeatures(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}
