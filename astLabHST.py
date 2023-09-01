# Usage python astLabHST.py --input input --output output --threshType ast --sizeThresh True --lineDet True --eval groundTruth --writeDir True


# Filename: astLabHST.py
# Version: 1.0
# Author: Nikolaus Kollo (A00411854)
# Date: 31 July 2023

# Description : Specify an input dir containing RGB images
# and the ast algorithm will be performed and saved in the
# output dir.  Stats can be calculated and displayed


# BEGIN PREAMBLE ------------------------------------------------------

# import the necessary packages
from imutils import contours

from matplotlib import pyplot as plt
import xlsxwriter
import numpy as np
import array as arr
import re
import argparse
import os
import time
import pathlib
import cv2

import astUtils
import astEvaluations
import astThreshTypes
import astPipeLine

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input folder")
ap.add_argument("-o", "--output", required=True,
    help="path to output folder")
ap.add_argument("-d", "--debug", required=False,
    help="shows debug imagery, enable with True or False")
ap.add_argument("-e", "--threshType", required=False,
    help="specify thresh type")
ap.add_argument("-x", "--sizeThresh", required=False,
    help="remove objects that are smaller than 65 pixels")
ap.add_argument("-w", "--writeDir", required=False,
    help="when True, all files will be written to a single directory, sub-dirs otherwise")
ap.add_argument("-y", "--lineDet", required=False,
    help="when True, line detection will happen on output mask")
ap.add_argument("-v", "--eval", required=False,
    help="Specifies the location of the groundTruth masks corresponding to the input directory")
args = vars(ap.parse_args())


    
#----------------Main Program------------------------------------------------
interpolNum = 0.9
sizeFilter = 20
max_size = 2800

# Initialize some variables for IO
fileHolder = "blank"
path = "blank"

# Initialize summing variables to calculate averages
iouCan = 0
hausdorffCan = 0
precisionCan = 0
recallCan = 0
accuracyCan = 0
f1Can = 0
mseCan = 0
rt1Can = 0
rt2Can = 0
rt3Can = 0
detCan = 0
rocFpCan = 0
rocTpCan = 0
rocAucCan = 0
    
# Establishing file structure
root = str(pathlib.Path(__file__).parent.resolve()) + '/'
path = root + args["output"]

if not os.path.isdir(path):
    os.mkdir(path)
    
if args["eval"] is not None:
    # and some for the stats
    statsCounter = 2
    workbook = xlsxwriter.Workbook(path + '/maskStats.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'FileName')
    worksheet.write('B1', 'IOU')
    worksheet.write('C1', 'Hausdorff')
    worksheet.write('D1', 'Percision')
    worksheet.write('E1', 'Recall')
    worksheet.write('F1', 'Accuracy')
    worksheet.write('G1', 'F1')
    worksheet.write('H1', 'MSE')
    worksheet.write('I1', 'Thresh Time')
    worksheet.write('J1', 'Morph Filter Time')
    worksheet.write('K1', 'Line Detect Time')
    worksheet.write('L1', 'Detected Lines')

# Checks for .DS_Store file that causes errors with IO
if os.path.exists(root + args["input"] + '.DS_Store'):
    os.remove(root + args["input"] + '.DS_Store')
    
# Load an array of files from the in directory
inFolder = os.listdir(args["input"])
fileCount = 0
# cycle through each file in the input folder and run algorithms
for file in inFolder:
    # Variable declarations for file
    img = cv2.imread(args["input"] + '/' + file)
    if args["eval"] is not None:
        print("Evaluation #", fileCount)
        fileCount += 1
        tFile = file.split(".")
        truth = tFile[0].replace("ast-detect-", "")
        truth2 = truth.replace("flc", "flc.fits_all_objects")
        truth = cv2.imread(args["eval"] + '/' + truth2 + '.png')
        print("Evaluating with mask...")
        print(args["eval"] + '/' + truth2 + '.png')
        print("------------------------")
        filename = args["eval"] + '/' + truth2 + '.png'
        truth = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
        if os.path.isfile(filename):
            print(f"The file {filename} exists.")
        else:
            print(f"The file {filename} does not exist.")
    fileName = file

    # Make the output directory if it doesn't exist
    if not os.path.isdir(path):
        os.mkdir(path)
        
    # Print filename and send to pipeline function
    print(fileName)
    
    # Convert input to grayscale image
    print(img.shape)
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Calculate the aspect ratio
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_w = w
    new_h = h
    # Set the desired size of the smallest dimension
    min_dim_size = 512

    # Resize the image while preserving aspect ratio
    resizer = 0
    if (w > max_size or h > max_size):
        resizer = sizeFilter
        if w < h:
            new_w = min_dim_size
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = min_dim_size
            new_w = int(new_h * aspect_ratio)
        
    
    
    # Runtime assignment
    check1 = time.time()
    
    if args["threshType"] == "canny":
        astMask = astThreshTypes.canny(gray)
    elif args["threshType"] == "ast":
        # Obtain AST mask
        astMask = astThreshTypes.asthresh(gray, interpolNum, args["debug"])
    else:
        astMask = gray

    # Runtime calculation
    check2 = time.time()
    runTime = check2 - check1
    runTime2 = 0
    runTime3 = 0
    astUtils.printRuntime(runTime)
    if args["threshType"] == "canny":
        astUtils.saveImage(astMask, "canny-", args["writeDir"], path, fileName)
    elif args["threshType"] == "ast":
        astUtils.saveImage(astMask, "ast-", args["writeDir"], path, fileName)
    
    # Remove small objects
    if args["sizeThresh"] == "True":
        print('Removing small objects...')
        resizedMask = cv2.resize(astMask, (new_w, new_h))
        check1 = time.time()
        astMask = astPipeLine.sizeThresh(resizedMask, 0, args["debug"])
        check2 = time.time()
        astMask = cv2.resize(astMask, (w, h))
        runTime2 = check2 - check1
        astUtils.printRuntime(runTime2)
        if args["threshType"] == "canny":
            astUtils.saveImage(astMask, "canny-size-", args["writeDir"], path, fileName)
        else:
            astUtils.saveImage(astMask, "ast-size-", args["writeDir"], path, fileName)
    
    # Detect Objects
    if args["lineDet"] == "True":
        print('Detecting Objects')
        check1 = time.time()
        resized_img = cv2.resize(astMask, (new_w, new_h))
        resized_img, detections = astPipeLine.detectLine(resized_img, resizer)
        if resizer > 0:
            print("------RUNNING SIZE THRESH!!!!!--------")
            #resized_img = sizeThresh(resized_img, resizer)
        astMask = cv2.resize(resized_img, (w, h))
        print(detections)
        check2 = time.time()
        runTime3 = check2 - check1
        astUtils.printRuntime(runTime3)
        if args["threshType"] == "canny":
            astUtils.saveImage(astMask, "canny-detect-", args["writeDir"], path, fileName)
        else:
            astUtils.saveImage(astMask, "ast-detect-", args["writeDir"], path, fileName)
    else:
        detections = 1
            
    # Evaluate Detections
    if args["eval"] is not None:
        print('Calculating Stats')
        testMask = astMask
        if detections > 0:
            iou = astEvaluations.calculate_iou(testMask, truth)
            if iou == 0:
                hausdorff = 2000
                percision = 0
                recall = 0
                accuracy = 0
                f1 = 0
                mse = 2000
                rocAuc = 0.5
            else:
                print("Tests==================")
                print("Hausdorff")
                hausdorff = astEvaluations.hausdorff_distance(testMask, truth)
                print("Precision")
                precision = astEvaluations.calculate_precision2(truth, testMask)
                print("recall")
                recall = astEvaluations.calculate_recall(testMask, truth)
                print("accuracy")
                accuracy = astEvaluations.calculate_accuracy2(testMask, truth)
                print("f1")
                f1 = astEvaluations.calculate_f1(precision, recall)
                print("mse")
                mse = astEvaluations.calculate_mse(testMask, truth)
                print("roc-auc")
                binary_truth = np.where(truth == 255, 1, 0)
                binary_mask = np.where(astMask == 255, 1, 0)
                rocFp, rocTp, rocAuc = astEvaluations.calculate_roc_curve_auc(binary_mask.flatten(), binary_truth.flatten())
                print("Scoring================")
                print("iou: ", iou)
                print("hausdorff: ", hausdorff)
                print("precision: ", precision)
                print("recall", recall)
                print("accuracy: ", accuracy)
                print("f1: ", f1)
                print("mse: ", mse)
                print("roc-auc: ", rocAuc)
            

        else:
            iou = 0
            hausdorff = 2000
            percision = 0
            recall = 0
            accuracy = 0
            f1 = 0
            mse = 2000
            rocAuc = 0.5
            
            
        # add values to summing variables
        iouCan += iou
        hausdorffCan += hausdorff
        precisionCan += precision
        recallCan += recall
        accuracyCan += accuracy
        f1Can += f1
        mseCan += mse
        if args["threshType"] == "none":
            runTime = 0
        rt1Can += runTime
        rt2Can += runTime2
        rt3Can += runTime3
        if ((statsCounter - 2) == 0):
            statsCounter += 1
        detCan /= (statsCounter - 2)
        rocFpCan /= (statsCounter - 2)
        rocTpCan /= (statsCounter - 2)
        rocAucCan /= (statsCounter - 2)

        detCan += detections
        rocAucCan += rocAuc
        # add values to worksheet
        worksheet.write('A' + str(statsCounter), file)
        worksheet.write('B' + str(statsCounter), iou)
        worksheet.write('C' + str(statsCounter), hausdorff)
        worksheet.write('D' + str(statsCounter), precision)
        worksheet.write('E' + str(statsCounter), recall)
        worksheet.write('F' + str(statsCounter), accuracy)
        worksheet.write('G' + str(statsCounter), f1)
        worksheet.write('H' + str(statsCounter), mse)
        worksheet.write('I' + str(statsCounter), runTime)
        worksheet.write('J' + str(statsCounter), runTime2)
        worksheet.write('K' + str(statsCounter), runTime3)
        worksheet.write('L' + str(statsCounter), detections)
        worksheet.write('M' + str(statsCounter), rocAuc)

        statsCounter += 1
        print("Evaluation Complete")
        print("")
if args["eval"] is not None and (statsCounter - 2) != 0:
    # Average and add to final row of worksheet
    # -2 accounts for label row and last statsCounter++
    file = "averages"
    iouCan /= (statsCounter - 2)
    hausdorffCan /= (statsCounter - 2)
    precisionCan /= (statsCounter - 2)
    recallCan /= (statsCounter - 2)
    accuracyCan /= (statsCounter - 2)
    f1Can /= (statsCounter - 2)
    mseCan /= (statsCounter - 2)
    rt1Can /= (statsCounter - 2)
    rt2Can /= (statsCounter - 2)
    rt3Can /= (statsCounter - 2)
    detCan /= (statsCounter - 2)
    rocFpCan /= (statsCounter - 2)
    rocTpCan /= (statsCounter - 2)
    rocAucCan /= (statsCounter - 2)
    
    # add values to worksheet
    worksheet.write('A' + str(statsCounter), file)
    worksheet.write('B' + str(statsCounter), iouCan)
    worksheet.write('C' + str(statsCounter), hausdorffCan)
    worksheet.write('D' + str(statsCounter), precisionCan)
    worksheet.write('E' + str(statsCounter), recallCan)
    worksheet.write('F' + str(statsCounter), accuracyCan)
    worksheet.write('G' + str(statsCounter), f1Can)
    worksheet.write('H' + str(statsCounter), mseCan)
    worksheet.write('I' + str(statsCounter), rt1Can)
    worksheet.write('J' + str(statsCounter), rt2Can)
    worksheet.write('K' + str(statsCounter), rt3Can)
    worksheet.write('L' + str(statsCounter), detCan)
    worksheet.write('M' + str(statsCounter), rocAucCan)
    workbook.close()
# END MAIN PROGRAM ----------------------------------------------------
