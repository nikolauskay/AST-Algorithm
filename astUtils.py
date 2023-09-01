import cv2

#----------------Utilities------------------------------------------------
# Calculates and returns the PSF of a line segment given 2 endpoints and a
# grayscale image.
def calculate_psf(gray, x1, y1, x2, y2):
    # get orientation of line
    dir, delta = get_orientation(x1, y1, x2, y2)
    avgWidth = 0
    # check line padding
    if checkLinePadding(gray, x1, y1, x2, y2, dir, psfBias) and delta > minimumStreakLength:
        mainLine = get_line_coords(img, x1, y1, x2, y2)
        hists = generate_histograms(gray, mainLine, psfBias, dir)
        if len(hists) > minimumStreakLength:
            for hist in hists:
                width  = calculate_mound_width(hist)
                avgWidth += width
            avgWidth = avgWidth/len(hists)
        else:
            avgWidth = 0
    return int(avgWidth)

# returns 0, as there is no well-defined FWHM. Otherwise, the function returns
# the distance between the first and last indices where the histogram crosses
# the half maximum value.
def calculate_mound_width(histogram):
    # Finds the maximum value in the histogram and calculates half that value
    max_val = np.max(histogram)
    half_max = max_val / 2
    
    # It then finds the indices of the histogram where the value is greater than or equal to the half maximum value.
    half_max_indices = np.where(histogram >= half_max)[0]
    if len(half_max_indices) < 2:
        return 0
    else:
        return half_max_indices[-1] - half_max_indices[0]
        
# Returns a list of histograms for each row or column in image based on the
# provided coordinates, width, and axis. If axis is 0, histograms are generated
# for each row in the image. If axis is 1, histograms are generated for each
# column in the image.
def generate_histograms(image, coords, width, axis):
    histograms = []
    for coord in coords:
        x, y = coord
        if axis == 0:
            row = image[y, x-width//2:x+width//2+1]
            histograms.append(np.histogram(row, bins=width, range=(0, 256))[0])
        elif axis == 1:
            col = image[y-width//2:y+width//2+1, x]
            histograms.append(np.histogram(col, bins=width, range=(0, 256))[0])
    return histograms

# Returns a list of x,y coordinates in a line segment
def get_line_coords(img, x1, y1, x2, y2):
    # Get the indices of the line segment
    indices = np.linspace(0, 1, max(img.shape), endpoint=True)
    x_indices = (1 - indices) * x1 + indices * x2
    y_indices = (1 - indices) * y1 + indices * y2

    # Round the indices to the nearest integer
    x_indices = np.round(x_indices).astype(np.int32)
    y_indices = np.round(y_indices).astype(np.int32)

    # Clip the indices to the image dimensions
    x_indices = np.clip(x_indices, 0, img.shape[1] - 1)
    y_indices = np.clip(y_indices, 0, img.shape[0] - 1)

    # Create a list of coordinates along the line segment
    coords = list(zip(x_indices, y_indices))

    return coords

# Returns 0 if points fall out of range of the bias
# Returns 1 if points are within the image
def checkLinePadding(gray, x1, y1, x2, y2, dir, bias):
    if dir == 0:
        if checkBias(gray, y1, bias, 0):
            if checkBias(gray, y2, bias, 0):
                return 1
        return 0
    else:
        if checkBias(gray, x1, bias, 1):
            if checkBias(gray, x2, bias, 1):
                return 1
        return 0

# Verifies that the neighbouring pixels are part of the image
def checkBias(image, coord, bias, dir):
    maxHeight = image.shape[dir]
    if coord + bias > maxHeight:
        return 0
    elif coord - bias < 0:
        return 0
    else:
        return 1
        
# Returns 0 if streak is mostly horizontal
# Returns 1 if streak is mostly vertical
def get_orientation(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    if dx > dy:
        return 0, dx # wide
    elif dx < dy:
        return 1, dy # tall
    else:
        return 0, dx # wide (assuming equal dimensions means wide)

# Print runtime
def printRuntime(time):
    print("Processing time in seconds =", time)
    print('')
    print('')

# Build size histogram
def sizeHist(areas):
    # Initialize a list with 256 locations with zeros values
    histogram = [0] * 256
    
    # Count areas by size
    for area in areas:
        # Small objects get recorded
        if area < 255:
            histogram[area] += 1
        # Large objects are grouped together
        else:
            histogram[255] += 1
    return histogram

# Show image
def peep(mask, name):
    cv2.imshow(name, mask)
    k = cv2.waitKey(0)
    
# Save the provided image to the output path
def saveImage(mask, prefix, writeDir, path, fileName):

    if writeDir == "True":
        cv2.imwrite(path +'/' + prefix + fileName, mask)
    else:
        if not os.path.isdir(path + '/' + fileName):
            os.mkdir(path + '/' + fileName)
        cv2.imwrite(path +'/' + fileName + '/' + prefix + fileName, mask)
