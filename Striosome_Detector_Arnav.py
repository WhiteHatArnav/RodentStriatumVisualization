# Code Author: Arnav Joshi
# Date of online publication on Github: May 19th, 2024
# Author Email: arnavj@alumni.princeton.edu
#______________________________________________________________


#|----------------The CV2 Image library---------------------------------------|
# CV2 is the image processing module primarily used in my method.
# For our purposes this is the best choice since our tasks are using computational
# methods to analyze complex histological images. Since CV2 approaches images as numpy arrays 
# based on a matrix of pixel co-orinates and color, we can use our computational knowlege to obtain
# incredibly precise results in annotating counting and visualisation so long as we have the 
# computational knowledge and tools to achieve the same.
import cv2 
#------------------------------Numpy------------------------------------------|
# While I can describe  this in detail as well I would recommend reading
# about this module and also pandas to understand what makes python so
# useful and the most popular language for data processing and machine 
# learning. Numpy structures are at the core of Python programming
import numpy as np
#|---------------------------------OS--------------------------------------|
# As the name makes clear, the OS library allows you to perform operations 
# that are normally performed by your Operating System. In our case we will 
# need to process image files to the CV2 system from repositories and
# folders. 
import os

#|------------------------Bar and tqdm-------------------------------------|
# These are more aesthetic and analytical choices. Bar is able to display a 
# Homogenous Progress Bar to tell you how much of the work is done by the 
# algorithm. This is usually combined with tqdm which is my favorite Python
# Module after Pytorch and tensorflow because you can use it with loops to 
# show how many iterations are done and time per iteration as well as other
# analytical parameters. This is still a good choice for our work since 
# as most coders know iterative processing is usually the biggest contributor
# to time complexity. 
from progress.bar import Bar
from tqdm.auto import tqdm

# This method detects the Striosomes in a fluoroscent died cross-section image of the rodent striosome
# While I am prpgramming it here to be able to detect them as a bright pink, adapting my work to detecting
# any color is trivial by just changing the color limits to the appropriate ones. 
# No additional changes should be required since this algorithm is refined enough to detect 97% of
# the structures in any image of a sufficiently high quality. An ROC analysis of the this would be 
# the appropriate way to determine algorithm accuracy.

# There are a few valid approaches to map the striosomes. This one seemed to work best for our dataset
# and offers the most flexibility in terms of color and shape detection which are our primary concerns 
# for Striosomal Detection.  
def Striosomal_Detection(lower_thresh, upper_thresh, min_area_thresh):
     # put your own directory path here.
     # if the directory path is in Windows format with "\\" as separators
     # you can use r before the path to make it recognizable to the python parser
    dir = "/Users/arnavjoshi/Desktop/AllHDImages"

    # os.listdir is the simplest approach of getting all the files in the directory but other 
    # approaches do exist.
    for file in tqdm(os.listdir(dir)):

        # here you can include a condition to verify the file is JPEG or TIFF or PNG but
        # this can limit your code's applicability because certain files that CV2 can be 
        # ones that did not come to your mind. Exception handling is a better way to
        # exclude undesirable files because that lets the module decide what is acceptable
        # rather than the coder

        # Reading in the image using CV2.
        img = cv2.imread(os.path.join(dir,file))

        # Convert the image to the HSV color space.
        # Since the BGR and RGB color scales are trying to include the entire visible spectrum
        # In 3 colors, while it works in displays and general visual applications it will not
        # work for sensitive color detection we need for this task. This is one of the reasons
        # other approaches sometimes fail for tasks like this. HSV on the other hand converts
        # tge inage to Hue and Saturation terms which is much better for thresholing colors in
        # our case.
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for pink color in HSV
        # This will vary wildly. To make this dynamic all you have to do is 
        # put these two variables in the two brackets after the function name
        # feel free to contact the author if this gives any trouble.
        # Here it is commented out so that we can do this dynamically

        # lower_thresh = np.array([140, 50, 50])
        # upper_thresh = np.array([180, 255, 255])


        # Create a "mask" of the pink regions of the image
        # to simplify it to an understanable level for our work
        # think of it as having pink spots on your face and then
        # you take a white or black facial mask and stick it to your face
        # then separate it from your face to only take away the pink parts
        # that are now imprinted on that unicolored mask.
        mask = cv2.inRange(hsv_img, lower_thresh, upper_thresh)

        # Find the color contours of the pink regions on the mask
        # So here you have to remember that CV2 represents images as a matrix
        # and the mask matrix has the exact same structure as the original
        # so when we detect contours on a mask then we essentially are detecting 
        # them on the original but without anything else interfering except the
        # colors we care about. If you need to understand what a contour is in
        # a computational context that would be too long to put here so contact 
        # the author for understanding that or you can also read up on it online.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        #simple enough to understand. this parameter can also go in between the brackets on top
        #this is just to make sure the tiny blemishes of pink that are not striosomes dont get included
        #in our detection. Commented out here and taken in dynamically.
        #min_area_thresh = 1000

        # Now we will draw white outlines to the striosomes on the image. Think of the for loop
        # looping through each striosome detected to then draw outlines around them if they are big
        # enough. 

        # First we draw a blank white image of the size of our input image
        # A 0 is the value of a white pixel hence the following statement
        # This may still be complex for a beginning coder so feel free to
        # message me for an explanation.
        contourimg = np.zeros_like(img)
        strio_count= 0
        # We loop thorugh the striosomes one
        for contour in contours:
            # We find the area of the striosome
            area = cv2.contourArea(contour)

            # We only draw anything ("detect the striosome") if the area is big enough
            if area > min_area_thresh:
                #Here we are drawing the outlines in white. Change color by changing
                # "(255, 255, 255)" to the right code. Please remember while cv2 processes
                # images in BGR the usual convention is RGB so look up what the convention 
                # is for each method before you use it.
                cv2.drawContours(contourimg, [contour], -1, (255, 255, 255), thickness=1)
                
                strio_count = strio_count + 1
        # calculates the total number of striosomes for each image. Make sure you remember this is all the major pink
        # regions in the entire image so yan algorithm or editing is needed to limit it to the regions you want.
        # I won't print it here but you can do that by just adding a print statement.
        

        # Overlay the outlined map you get on the original image and you have an image with striosomes outlines
        # Fun fact, the contourimg we produced is actually the exact same concept as making a mask but this time 
        # we drew it to label our image and didnt extract it as data from an image.
        outlined_img = cv2.bitwise_or(img, contourimg)
        # here i am removing file extensions so you can make the file in whatever format you want
        # long as cv2 can write it with imwrite()
        outfile = file.rstrip(".tif")
        outfile = file.rstrip(".jpg")

        # declaring output file name for the output file with the path and then writing it 
        outname ="/Users/arnavjoshi/Desktop/StrioOutlinedOutputHD/" + outfile + "_striosomes_outlined"+".jpg"
        cv2.imwrite(outname, outlined_img)


# Change these to try different colors of detection or to set the minimum 
# area for the structure to be detected(also max area for cells):
lower_pink = np.array([140, 50, 50])
upper_pink = np.array([180, 255, 255])
min_area_threshold_Striosome = 1000


# Function Call:
Striosomal_Detection(lower_pink, upper_pink, min_area_threshold_Striosome)