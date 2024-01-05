import cv2
import os
#reading the test sample 
sample= cv2.imread("archive/SOCOFing/Altered/Altered-Easy/1__M_Left_index_finger_Obl.BMP")

#creating a space for all the local variables of key points the image the filename and what score it has to be printed
best_score=0
filename=None
image=None
kp1,kp2,mp=None,None,None

#that runs through each and every train sample
for file in[ file for file in os.listdir("archive/SOCOFing/Real")]:
    fingerprint_image=cv2.imread("archive/SOCOFing/Real/"+file)
    sift=cv2.SIFT_create()

#sift algorithm
    keypoints_1, descriptors_1=sift.detectAndCompute(sample,None)
    keypoints_2, descriptors_2=sift.detectAndCompute(fingerprint_image,None)

# matches based on the feature vectors which is input the sift algorithm
    matches=cv2.FlannBasedMatcher(dict(algorithm=1,trees=10), dict()).knnMatch(descriptors_1,descriptors_2,k=2)
    match_points=[]


#threshold point 
    for p, q in matches:
        if p.distance< 0.1*q.distance:
            match_points.append(p)
    keypoint=0
    if len(keypoints_1)< len(keypoints_2):
        keypoint=len(keypoints_1)
    else:
        keypoint=len(keypoints_2)
    if len(match_points)/keypoint*100>best_score:
        best_score=len(match_points)/keypoint*100
        filename=file
        image=fingerprint_image
        kp1,kp2,mp=keypoints_1,keypoints_2,match_points
        result=cv2.drawMatches(sample,kp1,fingerprint_image,kp2,mp,None)


print("Best Match"+filename)
print("Score"+str(best_score))

cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()