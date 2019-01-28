import cv2
import numpy as np
import pickle

def build_squares(img):
	x, y, w, h = 420, 140, 10, 10
	d = 10
	imgCrop = None
	crop = None
	for i in range(10):
		for j in range(5):
			if np.any(imgCrop == None):
				imgCrop = img[y:y+h, x:x+w]
			else:
				imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
			#print(imgCrop.shape)
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
			x+=w+d
		if np.any(crop == None):
			crop = imgCrop
		else:
			crop = np.vstack((crop, imgCrop)) 
		imgCrop = None
		x = 420
		y+=h+d
	return crop

def get_hand_hist():
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	x, y, w, h = 300, 100, 300, 300
	flagPressedC, flagPressedS, flagPressedQ = False, False ,False

	# img切割的範圍
	imgCrop = None
	while True:

		# 搜尋目標圖片
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		cv2.imshow("hsv", hsv)
		
		keypress = cv2.waitKey(1)
		if keypress == ord('c'):	

			# 擷取imgCrop圖片的HSV
			hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
			# print(hsvCrop.shape)
			flagPressedC = True

			# region cv2.calcHist用法
			# cv2.calcHist(images, channels, mask, histSize, ranges)
			# imaages：要分析的圖片檔
			# channels：產生的直方圖類型。例：[0]→灰階，[0, 1, 2]→RGB三色。
			# mask：optional，若有提供則僅計算mask的部份。
			# histSize：要切分的像素強度值範圍，預設為256。每個channel皆可指定一個範圍。例如，[32,32,32] 表示RGB三個channels皆切分為32區段。
			# ranges：X軸(像素強度)的範圍，預設為[0,256]（非筆誤，calcHist函式的要求，最後那個值是表示<256）。
			# endregion

			# 用hsvCrop抓取所需的直方圖範圍，藉此追蹤手部
			hist = cv2.calcHist([hsvCrop], [0, 1], None, [60,140], [0, 180, 0, 256])
			# hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256]) origin

			# 正規化 結果在0~1之間 
			cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
		elif keypress == ord('s'):
			flagPressedS = True	
			break
		elif keypress == ord('q'):
			flagPressedQ = True	
			break
		if flagPressedC:
			# 直方圖反向投影  利用直方圖顯示感興趣的區塊
			dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
			cv2.imshow("dst", dst)

			# 複製
			dst1 = dst.copy()

			# 定義結構元素
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(14,14))
			# disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)) origin

			# region cv2.filter2D用法
			# cv2.filter2D(src, ddepth, kernel, dst, anchor)
			# src：輸入圖。
			# dst：輸出圖，和輸入圖的尺寸、通道數相同。
			# ddepth：輸出圖深度。
			# kernel：使用的核心。
			# anchor：錨點，預設為核心中央
			# endregion

			# 濾波(filtering) 提取感興趣的視覺特徵
			cv2.filter2D(dst,-1,disc,dst)

			# 高斯模糊
			blur = cv2.GaussianBlur(dst, (15,15), 0)
			cv2.imshow("blur", blur)
			
			# 中位數模糊
			blur = cv2.medianBlur(blur, 15)

			# 二值化
			ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			
			# 將 "多個單通道" 圖像合併成 "一個多通道圖像"
			thresh = cv2.merge((thresh,thresh,thresh))
			cv2.imshow("Thresh", thresh)
			
		if not flagPressedS:
			imgCrop = build_squares(img)
		cv2.imshow("Set hand histogram", img)
	cam.release()
	cv2.destroyAllWindows()

	# 開啟一個檔名"hist"的檔案 
	if not flagPressedQ:
		with open("hist", "wb") as f:
			# 將"hist"寫入
			pickle.dump(hist, f)

get_hand_hist()
