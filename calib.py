import numpy as np
import cv2
import glob
from scipy.linalg import svd
import pry


#Calibration
#estimating parameters for calibrating
#Task to estimate fx, fy, cx, cy, k1, k2

#camera relies on target to estimate camera intrinsic parameters

#CALIBRATION TARGET: Checkerboard.pdf


'''
Checkerboard.pdf
each square = 21.5 mm

x is image points
X is world points
k = [k(1), k(2)] is radial distortion parameters
K is camera calibration matrix
R is rotation matrix
t is translation of camera in the world frame


'''

def correspondance():
	size = 21.5 #mm size of each square
	grida, gridb = (6, 9)
	start = 0
	endx = size * 5
	endy = size * 8
	a = np.linspace(start, endx, grida)  
	b = np.linspace(start, endy, gridb) 
	worldX, worldY = np.meshgrid(a, b)  #my x and y world coordinates
	Z = np.zeros((6, 9))


	imgPts = []



	allimgs = glob.glob('./Calibration_Imgs/*.jpg')#testing


	for filename in allimgs:
		img = cv2.imread(filename) 
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray,(grida, gridb), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

		#ref: https://learnopencv.com/camera-calibration-using-opencv/
		if ret == True:
	
			# refining pixel coordinates for given 2d points.
			fincorners = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
			imgPts.append(fincorners.reshape(54,-1))
			# display corners
			img = cv2.drawChessboardCorners(img, (grida, gridb), fincorners, ret)

			#cv2.imshow('img',img)
			#cv2.waitKey(1000)
			cv2.imwrite("corners.png", img)
		


	
		print(imgPts)
		#u = imgPts[:,0]
		#v = imgPts[:,1]
		X = worldX
		Y = worldY
		matrix = []
	

		print("all pts")
		print(imgPts)
		#imgPts = imgPts.reshape(54,-1)

		curImg = imgPts[0]
		

		print("curimg whole ", len(curImg))
		print("curimg whole ", curImg)
		#choose4pts 
		chosenWorld = [[worldX[0], worldY[0]], [worldX[1], worldY[1]], [worldX[6], worldY[6]], [worldX[7], worldY[7]]]
		chosenImg = [[curImg[0][0], curImg[0][1]], [curImg[1][0], curImg[1][1]], [curImg[6][0], curImg[6][1]], [curImg[7][0], curImg[7][1]]]
		for coor in chosenImg:
			x = int(coor[0])
			y = int(coor[1])
			cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
		cv2.imshow('img',img)
		cv2.imwrite("4corners.png", img)
		cv2.waitKey(1000)

		
		for i in range(len(chosenImg)):
	
			r1 = [[X[i], Y[i], Z[i], 1], [0, 0, 0, 0], [-(u[i])*X[i], -(u[i])*Y[i], -(u[i])*Z[i], -(u[i])] ]
			r2 = [[0, 0, 0, 0], [X[i], Y[i], Z[i], 1], [0, 0, 0, 0], [-(v[i])*X[i], -(v[i])*Y[i], -(v[i])*Z[i], -(v[i])]]
			matrix = [matrix, r1, r2]
		U, s, VT = svd(matrix)  #ref https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
		Sigma = diag(s)
		B = U.dot(Sigma.dot(VT))

		print("MATRIX R and T")
		print(B)



correspondance()







