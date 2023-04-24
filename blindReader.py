from picamera2 import Picamera2 as PiCamera, Preview
import time
import numpy as np
from PIL import Image as im
#from scipy.ndimage import interpolation as inter
from PIL import Image,ImageEnhance,ImageFilter
import pygame
import pytesseract
import cv2


## loading alphabets
alphabets_file = open(r"soundData/Amharic_alphabets.txt", encoding="utf-8")
alphabets = alphabets_file.read()
alphabets = alphabets.replace("\n","")

#cv2.imshow("pic",output)
def capture_image():
	camera = PiCamera()
	camera_config = camera.create_still_configuration(lores={"size":(1200,1200)}, display='lores')
	camera.configure(camera_config)
	#camera.start_preview(Preview.QTGL)
	camera.start()
	#time.sleep(5)
	#camera.capture_file("test.png")
	output = camera.capture_array('lores')
	output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
	output = output[20:1180, 20:1180]
	cv2.imwrite("test2.png",output)
	output = cv2.imread("test2.png")
	return output
def ocr(image):
	custom_config =r'--oem 3 --psm 6'
	string = pytesseract.image_to_string(image, lang = 'amh')
	return string

def alignImage(img):
	ref_image = cv2.imread("ref.png")
	img1Gray = cv2.cvtColor(ref_image,cv2.COLOR_BGR2GRAY)
	img2Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	orb = cv2.ORB_create(4) #max_features
	keypoints1, descriptors1 = orb.detectAndCompute(img1Gray,None)
	keypoints2, descriptors2 = orb.detectAndCompute(img2Gray,None)
	
	
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1,descriptors2,None)
	
	imMatches = cv2.drawMatches(ref_image,keypoints1,img,keypoints2,matches,None)
	cv2.imwrite("matches.jpg",imMatches)
	
	points1 = np.zeros((len(matches),2), dtype=np.float32)
	points2 = np.zeros((len(matches),2), dtype=np.float32)
	
	for i, match in enumerate(matches):
		points1[i,:] = keypoints1[match.queryIdx].pt
		points2[i,:] = keypoints2[match.queryIdx].pt
	
	h,mask = cv2.findHomography(points1,points2,cv2.RANSAC)
	
	height,width, channel = img.shape
	img1Reg = cv2.warpPerspective(img,h,(width,height))
	
	return img1Reg,h
def correctSkew_scipy(img):
	# convert to binary
	wd, ht = img.size
	pix = np.array(img.convert('1').getdata(), np.uint8)
	bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
	plt.imshow(bin_img, cmap='gray')
	plt.savefig('binary.png')
	def find_score(arr, angle):
		data = inter.rotate(arr, angle, reshape=False, order=0)
		hist = np.sum(data, axis=1)
		score = np.sum((hist[1:] - hist[:-1]) ** 2)
		return hist, score
	delta = 1
	limit = 5
	angles = np.arange(-limit, limit+delta, delta)
	scores = []
	for angle in angles:
		hist, score = find_score(bin_img, angle)
		scores.append(score)
	best_score = max(scores)
	best_angle = angles[scores.index(best_score)]
	print('Best angle: {}'.formate(best_angle))
	# correct skew
	data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
	img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
	return img
def enhanceSharpness(img):
	
	contrast = ImageEnhance.Contrast(img)
	new_img = contrast.enhance(3)
	bright = ImageEnhance.Brightness(new_img)
	new_img = bright.enhance(1.1)
	sharp = ImageEnhance.Sharpness(new_img)
	new_img = sharp.enhance(3)
	
	
	return new_img
def deskew(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	# threshold the image, setting all foreground pixels to
	# 255 and all background pixels to 0
	thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# grab the (x, y) coordinates of all pixel values that
	# are greater than zero, then use these coordinates to
	# compute a rotated bounding box that contains all
	# coordinates
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	# the `cv2.minAreaRect` function returns values in the
	# range [-90, 0); as the rectangle rotates clockwise the
	# returned angle trends to 0 -- in this special case we
	# need to add 90 degrees to the angle
	if angle < -45:
		angle = -(90 + angle)
	# otherwise, just take the inverse of the angle to make
	# it positive
	else:
		angle = -angle
	# rotate the image to deskew it
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	return rotated
def removeObjects(image):
	mask = np.ones(image.shape, dtype=np.uint8) * 255
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
	dilate = cv2.dilate(thresh, kernel, iterations=3)

	cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
		area = cv2.contourArea(c)
		if area < 10000:
			x,y,w,h = cv2.boundingRect(c)
			mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
	
	cv2.imwrite("thres.png", thresh)
	cv2.imwrite("dilate.png", dilate)
	cv2.imwrite("mask.png", mask)
def readLoud(string):
	pygame.mixer.init()
	for s in string:
		print(s)
		if not(s in alphabets) or s==" " or "":
			s="sp"
		sound = pygame.mixer.Sound("soundData/splitted_sound_2/"+s+".wav")
		
		playing = sound.play()
		while playing.get_busy():
			pass
output = capture_image()
image = enhanceSharpness(Image.fromarray(output))
image = np.array(image)

image = deskew(image)
removeObjects(image)
string  = ocr(image)
readLoud(string[:10])
#cv2.imwrite("corrected.png",image)
