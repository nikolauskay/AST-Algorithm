# AST-Algorithm

The astLab and astLabHST scripts can be run on night sky and Hubble Space Telescope images respectively. Sample input images are provided in the input2 and input directories respectively. Sample ground truth segmentation masks are included in the two ground truth folders where the groundTruth folder corresponds to the HST image in the input folder.

Here is a usage example for each of the two scripts.

python astLabHST.py --input input --output output --threshType ast --sizeThresh True --lineDet True --eval groundTruth --writeDir True

python astLab.py --input input2 --output output2 --threshType ast --sizeThresh True --lineDet True --eval groundTruth2 --writeDir True

After running these scripts, the threshold masks, morphology masks and detection masks will be found in the output/output2 folders.

Stats will be printed in the terminal window and saved to an Excel file in the output folder after the script terminates.
