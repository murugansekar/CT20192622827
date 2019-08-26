# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 04:29:30 2019

@author: TamV3E11576
"""

import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
#from Knn_predict import Knn_predict 
import imutils
import pytesseract as tess
import pytesseract
import time
from sklearn.svm import SVC
#from ml_config import MachineLearningConfig
#from ml_validation import AccuracyValidation
import cv2
import re
import numpy as np

def preprocess(img):
	cv2.imshow("Input",img)
	imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
	cv2.imshow("blurred",imgBlurred)
	gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray",gray)
	cv2.waitKey(1)
	sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
	cv2.imshow("Sobel",sobelx)
	cv2.waitKey(0)
	ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imshow("Threshold",threshold_img)
	cv2.waitKey(0)
#    cv2.destroyAllWindows()
	return threshold_img

def cleanPlate(plate):
	print ("CLEANING PLATE. . .")
	#0gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	thresh= cv2.dilate(gray, kernel, iterations=1)

	_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	im1,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if contours:
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)

		max_cnt = contours[max_index]
		max_cntArea = areas[max_index]
		x,y,w,h = cv2.boundingRect(max_cnt)

		if not ratioCheck(max_cntArea,w,h):
			return plate,None

		cleaned_final = thresh[y:y+h, x:x+w]
		#cv2.imshow("Function Test",cleaned_final)
		return cleaned_final,[x,y,w,h]

	else:
		return plate,None


def extract_contours(threshold_img):
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    cv2.imshow("Morphed",morph_img_threshold)
    cv2.waitKey(0)
#    cv2.destroyAllwindows()
    cv2.destroyAllWindows()
    contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    return contours


def ratioCheck(area, width, height):
	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	aspect = 4.7272
	#aspect = 2
	min = 15*aspect*15  # minimum area
	max = 125*aspect*125  # maximum area
	#min = 15*aspect*15
	#max = 80*aspect*80
	rmin = 3
	rmax = 6

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True

def isMaxWhite(plate):
	avg = np.mean(plate)
	#if((avg>=150)and(avg<=500)):
	if(avg>=115):
		return True
	else:
 		return False

def validateRotationAndRatio(rect):
	(x, y), (width, height), rect_angle = rect

	if(width>height):
		angle = -rect_angle
	else:
		angle = 90 + rect_angle

	if angle>15:
	 	return False

	if height == 0 or width == 0:
		return False

	area = height*width
	if not ratioCheck(area,width,height):
		return False
	else:
		return True

def cleanAndRead(img,contours):
	for i,cnt in enumerate(contours):
		min_rect = cv2.minAreaRect(cnt)

		if validateRotationAndRatio(min_rect):

			x,y,w,h = cv2.boundingRect(cnt)
			plate_img = img[y:y+h,x:x+w]


			if(isMaxWhite(plate_img)):
				clean_plate, rect = cleanPlate(plate_img)

				if rect:
					x1,y1,w1,h1 = rect
					x,y,w,h = x+x1,y+y1,w1,h1
					cv2.imshow("Cleaned Plate",clean_plate)
					#cv2.waitKey(0)
					plate_im = Image.fromarray(clean_plate)
					img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
					cv2.imshow("Detected Plate",img)
					cv2.destroyAllWindows()

def crop_img(img,countours):
    for i,cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)
        if validateRotationAndRatio(min_rect):
            x,y,w,h = cv2.boundingRect(cnt)
            plate_img = img[y:y+h,x:x+w]
            grayplate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("croppped Plate",grayplate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('cropped.png',grayplate)
            cv2.imshow("croppped Plate",grayplate)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            print ("Detected Text : ")
            ret,thresh2 = cv2.threshold(grayplate,50,255,cv2.THRESH_BINARY_INV)
            kernel = np.ones((3,3),np.uint8)
            thresh2 = cv2.dilate(thresh2,kernel,iterations = 1)
            kernel = np.ones((1,1),np.uint8)
            thresh2 = cv2.erode(thresh2,kernel,iterations = 1)
           
#            text,text1 = Knn_predict('cropped.png')
#            print(text1)
            return grayplate,thresh2
			


if __name__ == '__main__':
    print("DETECTING PLATE . . .")
#    config = MachineLearningConfig()
#
#    image_data, target_data = config.read_training_data(config.training_data[0])
#
## kernel can be linear, rbf e.t.c
#    svc_model = SVC(kernel='linear', probability=True)
#
#    svc_model.fit(image_data, target_data)
#
##config.save_model(svc_model, 'SVC_model')
#
################################################
## for validation and testing purposes
################################################
#
#    validate = AccuracyValidation()
#
#    validate.split_validation(svc_model, image_data, target_data, True)
#
#    validate.cross_validation(svc_model, 3, image_data,
#    target_data)
    img = cv2.imread("Dataset1/9.jpg")
#    cv2.imwrite('image/car.jpg',img)
    threshold_img = preprocess(img)
    contours= extract_contours(threshold_img)
    grayplate,thresh2= crop_img(img,contours)
    cv2.imwrite('inputimage.jpg',grayplate)
    contours, hierarchy= cv2.findContours(thresh2,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    ### Step #2 - Reshape to 2D matrices
    contours1 = contours[0].reshape(-1,2)
    ### Step #3 - Draw the points as individual circles in the image
    img1 = thresh2.copy()
    (h, w) = img1.shape[:2]
    image_size = h*w
    mser = cv2.MSER_create()
    mser.setMaxArea(int(image_size/2))
    mser.setMinArea(10)
    regions, rects = mser.detectRegions(thresh2)
   
# With the rects you can e.g. crop the letters
    ii = 1
    for (x, y, w, h) in rects:
#        cv2.rectangle(img1, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=1)
        crop_img = img1[y:y+h, x:x+w]
       
        I1 = cv2.resize(crop_img,(20,20))
        testImgDet = I1.reshape(1, -1)
#        result = svc_model.predict(testImgDet)
#        print(result)
        pytesseract.pytesseract.tesseract_cmd = (r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe')
#        text = pytesseract.image_to_string(crop_img)
    text = pytesseract.image_to_string(Image.open('inputimage.jpg'))
    finaldata = re.sub('[":.&%$#@|/\!@#$§]', '', text)
    print(finaldata)
    cv2.imshow(" binary croppped Plate",img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#    file = open('sample.txt','w') 
#    file.write(text1) 
#    file.close() 