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

	count = 0
	V = []
	all_h = []
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

			cv2.imwrite("corners.png", img)
		
			X = worldX

			Y = worldY

			curImg = imgPts[count] #every image
		
		
			#choose4pts 
			chosenWorld = [[worldX[0][0], worldY[0][0]], [worldX[1][0], worldY[1][0]], [worldX[0][1], worldY[0][1]], [worldX[1][1], worldY[1][1]]]
			chosenImg = [[curImg[0][0], curImg[0][1]], [curImg[1][0], curImg[1][1]], [curImg[6][0], curImg[6][1]], [curImg[7][0], curImg[7][1]]]
			cv2.circle(img, (int(curImg[0][0]),int(curImg[0][1])), 20, (255, 0, 0), 2) #red
			cv2.circle(img, (int(curImg[1][0]), int(curImg[1][1])), 20, (0, 0, 255), 2) #blue
			cv2.circle(img, (int(curImg[6][0]), int(curImg[6][1])), 20, (0, 255, 0), 2) #green
			cv2.circle(img, (int(curImg[7][0]), int(curImg[7][1])), 20, (0, 0, 0), 2) #black

			cv2.imwrite("img" + str(count) +".png", img)

			chosenWorld =np.float32(np.array(chosenWorld))
			chosenImg = np.float32(np.array(chosenImg))

			h = cv2.getPerspectiveTransform(chosenWorld, chosenImg)
			all_h.append(h)


			v12 = getvij(h, 0,1)
			v11 = getvij(h, 0,0)
			v22 = getvij(h, 1,1)

			V.append([v12, np.subtract(v11, v22)])


		
			count += 1



	V = np.array(V)
	V = V.reshape((-1,V.shape[2]))


	u, s, v = np.linalg.svd(V, full_matrices = True) 
	b = v[-1]


	B = getBigB(b)




	
	A, lam = intparameters(b)

	all_R = []
	for i in range(len(all_h)):

		R = getRT(A, all_h[i], lam)
		all_R.append(R)

	print("All R ")
	print(all_R)


	alpha  = A[0][0]
	beta = A[1][1]
	u0 = A[0][2]
	v0 = A[1][2]
	#centerDx, centerDy = minimize(A[0][0], A[1][1], A[0][2], A[1][2]) #alpha, beta, u0, v0

	
	k = [0,0]
	for i in range(len(100)):
		k1 = k[0]
		k2 = k[1]

		x = u0 + alpha * changedX
		y = v0 + Beta * changedY 
		changedX = x + x*(k1*(x**2  + y**2 ) + k2*(x**2 + y ** 2) ** 2)
		changedY = y + y*((k1 * (x**2 + y ** 2) + k2*(x**2 + y**2)) ** 2) 

		


		topl = (x - u0) * (x ** 2 + y**2 )
		topr = (x - u0) * ((x ** 2 + y** 2)**2)
		botl = (y - v0) * (x**2 + y ** 2)
		botr = (y - v0) * ((x**2  + y **2 ) ** 2)

		D = np.array([topl, topr], [botl, botr])

		d = np.array([(changedX  - u0), (changedY - Y)])

		k = np.linalg.inv(np.float32(np.dot(transpose(D), D) )) * transpose(D) * d 

		print("K values ")
		print(k)

		





	#b = littleb(V)
	'''
	U, s, VT = svd(matrix)  #ref https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
	Sigma = diag(s)
	B = U.dot(Sigma.dot(VT))
	'''

	

def getvij(h, i, j):
	vij = [h[0][i] * h[0][j], h[0][i]*h[1][j]  + h[1][i] * h[0][j], h[1][i]*h[1][j], h[2][i]*h[0][j] + h[0][i]*h[2][j], h[2][i]*h[1][j] + h[1][i]*h[2][j], h[2][i]*h[2][j]]

	return vij

def transpose(value):
	ret = cv2.transpose(np.float32(value))
	return ret


#def littleb(V):
	#h = cv2.getPerspectiveTransform(chosenWorld, chosenImg)

def getBigB(b):

	#b = b.reshape(6,-1)

	B = [[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]]
	return B



def intparameters(B):
	print("B[0][0] ", B[0])


	v0 = (B[1] * B[3] - B[0] * B[4]) / (B[0] * B[2] - B[1]**2)
	lam = B[5] - (B[3]**2 + v0 * (B[1] * B[3] - B[0] * B[4])) / B[0]
	print("lam ", lam)
	alpha = np.sqrt(lam / B[0])
	beta = np.sqrt((lam * B[0]) / (B[0] * B[2] - B[1]**2))
	gamma = -B[1] * (alpha ** 2) * beta / lam
	print("gamma ", lam)
	u0 = gamma*v0 / beta - B[3] * alpha ** 2/ lam

	#A = [[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]

	A =[
		[alpha, gamma, u0],
		[0, beta, v0],
		[0, 0, 1]
		]

	return A, lam



def getRT(A, h, lam):

	
	invA = np.linalg.inv(np.float32(np.array(A)))
	h1 = np.array([h[0][0], h[0][1], h[0][2]])
	h2 = np.array([h[1][0], h[1][1], h[1][2]])
	h3 = np.array([h[2][0], h[2][1], h[2][2]])

	#lam = 1 / np.absolute(np.dot(invA ,h1))
	#lam2 = 1 / np.absolute(np.dot(invA ,h2))  


	r1 = np.absolute(np.dot(np.dot(lam, invA), h1))
	r2 = np.absolute(np.dot(np.dot(lam, invA), h2))

	r3 = np.cross(r1, r2)
	t = np.dot(np.dot(lam, invA), h3)


	R = [r1, r2, r3, t] #Make 3 by 4 matrix 
	print("len of R ", R)
	R = np.array([r1, r2, r3, t])
	R.reshape(4,-1)
	


	return R





correspondance()
















