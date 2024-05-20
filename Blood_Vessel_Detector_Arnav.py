# Code Author: Arnav Joshi
# Date of online publication on Github: May 17th, 2024
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

# This follows a similar logic as the striosome detection with some minor changes. 
# The approach is described stepwise in the function.
# After testing 3 detection methods this still was the best approach for black color
# as well. I have taken a slightly different approach to the
# logical and iterative steps to emphasize that we  can do this many ways long 
# as we use the right tools and take the right basic approach.

def Blood_Vessel_Detection(lower_thresh, upper_thresh, min_area_thresh):
    # Input and Output Directory Paths to make it more organize and simpler
    # to customize code source and outflow directory
    input_dir = '/Users/arnavjoshi/Desktop/AllHDImages'
    output_dir = '/Users/arnavjoshi/Desktop/BloodVesselsOutlinedHD'

    # Minimum area threshold to make sure our algorithm is 
    # not circling random black dots as blood vessels
    min_area_threshold = 200

    # Get list of image files in the input directory
    image_files = [file for file in os.listdir(input_dir) ]

    # Loop to read and mark blood vessels in each image in our input dir
    for file_name in tqdm(image_files, colour='green'):
    
        # Reading in the image using CV2
        orig_img = cv2.imread(os.path.join(input_dir, file_name))

        # Convert the image to the HSV color space.
        # Since the BGR and RGB color scales are trying to include the entire visible spectrum
        # In 3 colors, while it works in displays and general visual applications it will not
        # work for sensitive color detection we need for this task. This is one of the reasons
        # other approaches sometimes fail for tasks like this. HSV on the other hand converts
        # tge inage to Hue and Saturation terms which is much better for thresholing colors in
        # our case.
        hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for black color in HSV
        # This will vary wildly. To make this dynamic all you have to do is 
        # put these two variables in the two brackets after the function name
        # feel free to contact the author if this gives any trouble
       # Adjust upper bound as needed for black color

        # Create a black "mask" of the black regions of the image
        # to simplify it to an understanable level for our work
        # think of it as having Black spots on your face and then
        # you take a white facial mask and stick it to your face
        # then separate it from your face to only take away the black parts
        # that are now imprinted on that unicolored mask.
        mask = cv2.inRange(hsv_img, lower_thresh, upper_thresh)

        # Find the color contours of the black regions on the mask
        # So here you have to remember that CV2 represents images as a matrix
        # and the mask matrix has the exact same structure as the original
        # so when we detect contours on a mask then we essentially are detecting 
        # them on the original but without anything else interfering except the
        # colors we care about. If you need to understand what a contour is in
        # a computational context that would be too long to put here so contact 
        # the author for understanding that or you can also read up on it online.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Just a different way to loop thorugh the contours and remove black regions too 
        # small to be blood vessels
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
    
        # Draw green outlines on the original image just because we dont want it to be the same color
        # as the white outlines for the pink regions in case we put it on the same image
        outlined_img = orig_img.copy()
        cv2.drawContours(outlined_img, filtered_contours, -1, (0, 255, 0), thickness=1)

        #In case you want a blood vessel count uncomment the following:
        #print("Blood Vessel Count:", len(filtered_contours))

        # Save the resulting image with outlines but again I have done this slightly differently
        # there is always an opportunity to learn more
        
        output_file_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '_outlined.jpg')
        cv2.imwrite(output_file_path, outlined_img)


# Change these to try different colors of detection or to set the minimum 
# area for the structure to be detected(also max area for cells).
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30]) 
min_area_threshold_Vein = 200

# Function Call:
Blood_Vessel_Detection(lower_black, upper_black, min_area_threshold_Vein)

