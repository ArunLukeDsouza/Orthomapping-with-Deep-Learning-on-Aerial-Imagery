# Deep Combiner


import cv2
import numpy as np
import utilities as util
import geometry as gm
import copy
import os


class Combiner:
    def __init__(self,imageList_,dataMatrix_):
        '''
        :param imageList_: List of all images in dataset.
        :param dataMatrix_: Matrix with all pose data in dataset.
        :return:
        '''
        self.imageList = []
        self.dataMatrix = dataMatrix_




        for i in range(0,len(imageList_)):
            image = imageList_[i][::2,::2,:] #downsample the image to speed things up. 4000x3000 is huge!
            M = gm.computeUnRotMatrix(self.dataMatrix[i,:])
            #Perform a perspective transformation based on pose information.
            #Ideally, this will mnake each image look as if it's viewed from the top.
            #We assume the ground plane is perfectly flat.
            correctedImage = gm.warpPerspectiveWithPadding(image,M)
            self.imageList.append(correctedImage) #store only corrected images to use in combination
        self.resultImage = self.imageList[0]
    def createMosaic(self):
        for i in range(1,len(self.imageList)):
            self.combine(i)
        return self.resultImage

    def combine(self, index2):
        '''
        :param index2: index of self.imageList and self.kpList to combine with self.referenceImage and self.referenceKeypoints
        :return: combination of reference image and image at index 2
        '''

        #Attempt to combine one pair of images at each step. Assume the order in which the images are given is the best order.
        #This intorduces drift!
        image1 = copy.copy(self.imageList[index2 - 1])
        image2 = copy.copy(self.imageList[index2])

        '''
        Descriptor computation and matching.
        Idea: Align the images by aligning features.
        '''
        # Keypoint extraction from .npz
        path = os.getcwd()
        print(path)
        image_path = os.path.join(path, "d2_net/datasets/resized_images")

        # pair_path = "/Users/arun/Downloads/AERO2ASTRO/Project GIS/ortho_d2net/ortho/d2_net/qualitative/images/pair_4"

        feat1 = np.load(os.path.join(image_path, '1.jpg.d2-net'))
        feat2 = np.load(os.path.join(image_path, '2.jpg.d2-net'))

        gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        kp1_npz = feat1['keypoints']
        kp1_del = np.delete(kp1_npz, 2, 1)
        kp1 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1_del]

        gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        kp2_npz = feat2['keypoints']
        kp2_del = np.delete(kp2_npz, 2, 1)
        kp2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2_del]

        descriptors1 = feat1['descriptors']
        descriptors2 = feat2['descriptors']

        print("descriptors1", descriptors1.shape)
        print("descriptors2", descriptors2.shape)


        #Visualize matching procedure.
        keypoints1Im = cv2.drawKeypoints(image1, kp1, outImage = None, color=(0,0,255))
        util.display("KEYPOINTS",keypoints1Im)
        keypoints2Im = cv2.drawKeypoints(image2 ,kp2, outImage = None, color=(0,0,255))
        util.display("KEYPOINTS",keypoints2Im)

        matcher = cv2.BFMatcher() #use brute force matching
        matches = matcher.knnMatch(descriptors2,descriptors1, k=2) #find pairs of nearest matches
        #prune bad matches
        # https://stackoverflow.com/questions/50945385/python-opencv-findhomography-inputs

       #define constants
        MIN_MATCH_COUNT = 20
        MIN_DIST_THRESHOLD = 0.7
        RANSAC_REPROJ_THRESHOLD = 4.0

        good = []
        for m,n in matches:
            if m.distance < MIN_DIST_THRESHOLD *  n.distance:
                good.append(m)
        matches = copy.copy(good)
        print("Number of Good Matches: ", len(good))

        #Visualize matches
        matchDrawing = util.drawMatches(gray2,kp2,gray1,kp1,matches)
        util.display("matches",matchDrawing)


        if len(good) > MIN_MATCH_COUNT:

            #NumPy syntax for extracting location data from match data structure in matrix form
            src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

            A = cv2.estimateAffinePartial2D(src_pts,dst_pts)
            if A == None:
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

        else:
            raise Exception("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))



        '''
        Compute Affine Transform
        Idea: Because we corrected for camera orientation, an affine transformation *should* be enough to align the images
        '''



        '''
        Compute 4 Image Corners Locations
        Idea: Same process as warpPerspectiveWithPadding() excewpt we have to consider the sizes of two images. Might be cleaner as a function.
        '''
        height1,width1 = image1.shape[:2]
        height2,width2 = image2.shape[:2]
        corners1 = np.float32(([0,0],[0,height1],[width1,height1],[width1,0]))
        corners2 = np.float32(([0,0],[0,height2],[width2,height2],[width2,0]))
        warpedCorners2 = np.zeros((4,2))
        for i in range(0,4):
            cornerX = corners2[i,0]
            cornerY = corners2[i,1]
            if A != None: #check if we're working with affine transform or perspective transform
                warpedCorners2[i,0] = A[0][0,0]*cornerX + A[0][0,1]*cornerY + A[0][0,2]
                warpedCorners2[i,1] = A[0][1,0]*cornerX + A[0][1,1]*cornerY + A[0][1,2]
            else:
                warpedCorners2[i,0] = (H[0,0]*cornerX + H[0,1]*cornerY + H[0,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
                warpedCorners2[i,1] = (H[1,0]*cornerX + H[1,1]*cornerY + H[1,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
        allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
        [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
        [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)

        '''Compute Image Alignment and Keypoint Alignment'''
        translation = np.float32(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1]))
        warpedResImg = cv2.warpPerspective(self.resultImage, translation, (xMax-xMin, yMax-yMin))
        if A == None:
            fullTransformation = np.dot(translation,H) #again, images must be translated to be 100% visible in new canvas
            warpedImage2 = cv2.warpPerspective(image2, fullTransformation, (xMax-xMin, yMax-yMin))
        else:
            warpedImageTemp = cv2.warpPerspective(image2, translation, (xMax-xMin, yMax-yMin))
            warpedImage2 = cv2.warpAffine(warpedImageTemp, np.float32(A[0]), (xMax-xMin, yMax-yMin))
        self.imageList[index2] = copy.copy(warpedImage2) #crucial: update old images for future feature extractions

        resGray = cv2.cvtColor(self.resultImage,cv2.COLOR_BGR2GRAY)
        warpedResGray = cv2.warpPerspective(resGray, translation, (xMax-xMin, yMax-yMin))

        '''Compute Mask for Image Combination'''
        ret, mask1 = cv2.threshold(warpedResGray,1,255,cv2.THRESH_BINARY_INV)
        mask3 = np.float32(mask1)/255

        #apply mask
        warpedImage2[:,:,0] = warpedImage2[:,:,0]*mask3
        warpedImage2[:,:,1] = warpedImage2[:,:,1]*mask3
        warpedImage2[:,:,2] = warpedImage2[:,:,2]*mask3

        result = warpedResImg + warpedImage2
        #visualize and save result
        self.resultImage = result
        util.display("result",result)
        cv2.imwrite("results/intermediateResult"+str(index2)+".png",result)
        return result
