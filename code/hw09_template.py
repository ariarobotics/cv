import numpy as np
import cv2 as cv

def get_keypoint(left_img, right_img):
    l_img = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    r_img = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)
    
    sift = cv.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
    key_points2, descriptor2 = sift.detectAndCompute(r_img, None)
    
    return key_points1, descriptor1, key_points2, descriptor2


def match_keypoints(descriptor1, descriptor2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor1, descriptor2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)    
    
    return good_matches


img1 = cv.imread('data/left.png')
img2 = cv.imread('data/right.png')    

f, cx, cy = 48.5, 48.5, 48.5
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(img1, img2)

good_matches = match_keypoints(descriptor1, descriptor2)

pts1 = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([key_points2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

F, inlier_mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 2, 0.99, maxIters=1000)

E = K.T @ F @ K 
positive_num, R, t, positive_mask = cv.recoverPose(E, pts1, pts2, K, mask=inlier_mask)

P0 = K @ np.eye(3, 4, dtype=np.float32)
Rt = np.hstack((R, t))
P1 = K @ Rt
pts1_inlier = pts1[inlier_mask.ravel() == 1]
pts2_inlier = pts2[inlier_mask.ravel() == 1]
X = cv.triangulatePoints(P0, P1, pts1_inlier, pts2_inlier)
X /= X[3]
X = X.T