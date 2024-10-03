import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
Sources:
https://stackoverflow.com/questions/31460267/python-opencv-color-tracking
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
'''

'''
HSV Version:
	colorLower = np.array([100, 50, 10])
	colorUpper = np.array([120, 255, 240])
	hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsvImage, colorLower, colorUpper) 
BGR Version:
	colorLower = np.array([100, 0, 0])
	colorUpper = np.array([255, 125, 125])
	mask = cv2.inRange(frame, colorLower, colorUpper) 

HSV seems to be more reliable in a wider range of lighting conditions

A bright flashlight leaves a dark spot in the observed threshold output,
reducing tracking ability.
'''
def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Display the resulting frame
	colorLower = np.array([100, 50, 10])
	colorUpper = np.array([120, 255, 240])
	hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsvImage, colorLower, colorUpper) 
	maskImage = cv2.bitwise_and(frame, frame, mask=mask)

	maskImageGray = cv2.cvtColor( maskImage, cv2.COLOR_BGR2GRAY )
	ret, thresh = cv2.threshold(mask, 50, 255, 0)
	contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
	for item in contours:
		x,y,w,h = cv2.boundingRect( item )
		if w > 20 and h > 20:
			cv2.rectangle( frame, (x,y), (x+w,y+h), (0,255,0), 2)
	
	cv2.imshow('frame', frame)

	box_y = 100
	box_x = 100
	start_y = len(frame)//2 - box_y//2
	start_x = len(frame[0])//2 - box_x//2
	crop_img = frame[ start_y : start_y + box_y, start_x : start_x + box_x]
	crop_img = crop_img.reshape(crop_img.shape[0] * crop_img.shape[1],3)
	clt = KMeans(n_clusters=3) #cluster number
	clt.fit(crop_img)

	hist = find_histogram(clt)
	bar = plot_colors2(hist, clt.cluster_centers_)

	plt.axis("off")
	plt.imshow(bar)
	plt.show()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


