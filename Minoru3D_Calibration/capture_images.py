import numpy as np
import cv2
import time

CamL_id, CamR_id = 2, 3
CamL, CamR = cv2.VideoCapture(CamL_id), cv2.VideoCapture(CamR_id)
output_path = "./data/"

start = time.time()
T = 5
count = 0

while True:
    timer = T - int(time.time() - start)
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()
    img1_temp = frameL.copy()
    cv2.putText(img1_temp, f"{timer}", (50, 50), 1, 5, (55, 0, 0), 5)
    cv2.imshow("imgR", frameR)
    cv2.imshow("imgL", img1_temp)

    grayR, grayL = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    retR, cornersR = cv2.findChessboardCorners(grayR, (9, 6), None)
    retL, cornersL = cv2.findChessboardCorners(grayL, (9, 6), None)
    if (retR == True) and (retL == True) and timer <= 0:
        count += 1
        cv2.imwrite(f"{output_path}stereoR/img{count}.png", frameR)
        cv2.imwrite(f"{output_path}stereoL/img{count}.png", frameL)
    if timer <= 0:
        start = time.time()
    if cv2.waitKey(1) & 0xFF == 27:
        print("Closing the cameras!")
        break

CamR.release()
CamL.release()
cv2.destroyAllWindows()
