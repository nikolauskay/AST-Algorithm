import cv2
import numpy as np

#----------------Thresholding Algorithms-------------------------------------
# This function takes a grayscale input and performs the ast algorithm
def asthresh(gray, interpolNum, debug):
    # Adding a blur the image with a 3x3 Gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Create the histogram and flatten the data
    hdata = cv2.calcHist([blur],[0],None,[256],[0,256])
    hdata = hdata.flatten()
    
    # Normalize histogram
    hdata = (1/np.sum(hdata)) * hdata
    
    # Find argMax of histogram
    highPass = np.argmax(hdata)
    
    # Clip the search area down to values larger than highPass
    searchHist = hdata[highPass:]
    
    # Define fall back value for lowPass and a counter variable
    lowPass = 255
    count = 0
    lastValue = 0
    
    # Continue to iterate through remaining histogram
    for val in searchHist:
        #print(val)
        # Check for diminishing contributions
        if 0 < val < 0.001:
            lowPass = highPass + count
            break
        count += 1
        
    # Interpolate 90% between the two filters as an int
    tVal = highPass + int((lowPass-highPass)*interpolNum)
    
    # Display the detected values if debug is on
    if debug == "True":
        print('highPass: ', highPass)
        print('lowPass: ', lowPass)
        print('ast-tVal:', tVal)

    # Apply binary threshold with computed tVal and return
    mask = cv2.threshold(gray,tVal,255,cv2.THRESH_BINARY)[1]
    return mask
    
def canny(gray):
    immax = np.max(gray)
    # Adding a blur the image with a 3x3 Gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(blur, immax * 0.1, immax * 0.5)
    return edge
