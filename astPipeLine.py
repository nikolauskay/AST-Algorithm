from skimage import measure, morphology, filters
import numpy as np
import cv2

#----------------Pipeline Functions------------------------------------------
# This function takes a binary input image and performs a size thresh based on the average detected object size
def sizeThresh(thresh, sizeT, debug):
    # Initialize some variables for the function
    count = 0
    total = 0
    perimeters = []
    areas = []
    extents = []
    eccentricities = []
    circularities = []
    
    # Connected component anaylsis
    labels = measure.label(thresh, connectivity=2, background=0)
    props = measure.regionprops(labels)

    # Add detected areas to an areas array
    for prop in props:
        perimeters.append(prop.perimeter)
        areas.append(prop.area)
        extents.append(prop.extent)
        eccentricities.append(prop.eccentricity)
        #print("eccentricity: ",prop.eccentricity)

    # Size filter
    if sizeT != 0 and len(areas) != 0:
        threshold_area = np.percentile(areas, sizeT)
        print("")
        print("size Thresh")
        print(threshold_area)
        print("area array size")
        print(len(areas))
        print("-----------------")
    else:
        print("")
        print("shape Thresh")
        threshold_area = 1
    selected_labels = [prop.label for prop in props if prop.area >= threshold_area]
    print("selected areas length")
    print(len(selected_labels))
    
    
    #if sizeT != 0:
    #    print("areas")
    #    for lab in areas:
    #        print(lab)
    #print("")
    
    # Make a histogram for the 1D array areas where the index relates to the area and the
    # value is the number of objects of size index.
    # areaHist = sizeHist(areas)

    # a blank mask to build with
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Print out stats if debug is on
    if debug == "True":
        print('number of objects detected:', len(perimeters))
        # show histogram of areas array
        plt.plot(areaHist)
        plt.show()
    # reset and reuse the count variable
    count = 0
    
    
    # cycle through all connected areas...
    for label in np.unique(selected_labels):
    # if this is the background label, ignore it
        if label == 0:
            continue
        if sizeT == 0:
            # filter for line segments
            if eccentricities[count] >= .98:
                # filter
                if extents[count] <= .68:
                    labelMask = np.zeros(thresh.shape, dtype="uint8")
                    labelMask[labels == label] = 255
                    # add object to output
                    mask = cv2.add(mask, labelMask)
        else:
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            # add object to output
            mask = cv2.add(mask, labelMask)
        count += 1
    if sizeT != 0:
        kernel = np.ones((3,3),np.uint8)  # Example kernel of size 3x3
        mask = cv2.dilate(mask, kernel, iterations = 1)

    return mask
    
# Perform line detection using Probabilistic Hough Lines
def detectLine(inMask, resizer):
    # use a smaller line length for smaller images
    if resizer == 0:
        minLineLength = 50
        maxLineGap = 3
    else:
        minLineLength = 100
        maxLineGap = 10
        
    detections = 0
    mask = np.zeros(inMask.shape, np.uint8)
    # labels = measure.label(maskBack, connectivity=2, background=0)
    # The rho and angle accuracy is 1 pixel and pi/180 degrees respectively.
    lines = cv2.HoughLinesP(inMask,1,np.pi/180,10,minLineLength,maxLineGap)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                #midpoint_x = (x1 + x2) / 2
                #midpoint_y = (y1 + y2) / 2
                #labelValue = labels[int(midpoint_y), int(midpoint_x)]
                #mask[labels == labelValue] = 255
                cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),2)
                detections += 1
    return mask, detections
