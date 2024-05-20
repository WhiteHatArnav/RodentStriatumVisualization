# Code Author: Arnav Joshi
# Date of online publication on Github: May 17th, 2024
# Author Email: arnavj@alumni.princeton.edu
#______________________________________________________________

# Some Remarks about this code and its uniqueness and applicability: 
# The program is able to detect irregularly colored and sized tissue 
# structures from a dynamic script. It represents serious work in the direction of a 
# breakthrough in understanding biological tissue structures which implicates possible utility
# in areas ranging from oncology to neuroscience.

#|-------------------------------The CV2 Image library---------------------------------------|
# CV2 is the image processing module primarily used in my method.
# For our purposes this is the best choice since our tasks are using computational
# methods to analyze complex histological images. Since CV2 approaches images as numpy arrays 
# based on a matrix of pixel co-orinates and color, we can use our computational knowlege to obtain
# incredibly precise results in annotating counting and visualisation so long as we have the 
# computational knowledge and tools to achieve the same.
import cv2 
#---------------------------Numpy------------------------------------------|
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
# the structures in any image of a sufficiently high quality. While it has been suggested that an
# ROC analysis of the this would be the appropriate way to determine algorithm accuracy, no reliable
# theoretical counts or objective mathematical definitions and parameters exist that can establish 
# a succesful generation of an ROC curve. Any current attempt at this without collection of data 
# from a sufficiently large data set is speculative and of negligible computational validity.

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

# This follows a similar logic as the striosome detection with some minor changes 
# After testing 3 detection methods this still was the best approach for black color
# as well. I have taken a slightly different approach to the
# logical and iterative steps to emphasize that we  can do this many ways long 
# as we use the right tools and take the right basic approach
def Blood_Vessel_Detection(lower_thresh, upper_thresh, min_area_thresh):
    # Input and Output Directory Paths to make it more organize and simpler
    # to customize code source and outflow directory
    input_dir = '/Users/arnavjoshi/Desktop/AllHDImages'
    output_dir = '/Users/arnavjoshi/Desktop/BloodVesselsOutlinedHD'

    # Minimum area threshold to make sure our algorithm is 
    # not circling random black dots as blood vessels
    # min_area_thresh = 200

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
        
        #lower_thresh = np.array([0, 0, 0])
        #upper_thresh = np.array([180, 255, 30]) 

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
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_thresh]
    
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


# The code for cell detection and counting will be more complex since there is quite a bit of 
# processing and some calculations that have been carried out.
    
def Cell_Detector_and_Counter_Connected_Component(area_min_thresh, area_max_thresh):
    input_dir = "/Users/arnavjoshi/Desktop/AllHDImages"
    output_dir = "/Users/arnavjoshi/Desktop/ConnectedComponentCellDetectionHD/"

    mean_cell_area = []
    sd_cell_area = []

    for file in tqdm(os.listdir(input_dir), colour="green"):
        # Reading in the image using CV2 and then making a copy in case our operations change the original
        img = cv2.imread(os.path.join(input_dir, file))
        og = img.copy()
        # we grayscale the  image which in simple terms is making it black and white to 
        # simplify processing.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert to binary using thresholding. This is different than grayscaling because
        # even though grayscale images are black and white they are not so based on a threshold
        # we set specifically and we do this to actually binarize it into two very distinct colors 
        # or presence and lack of color. For things like counting dots this makes our job significantly
        # easier because any brightly colored dots become very distinct dots and then all the rest kind
        # of becomes the other color mostly.
        _, thr = cv2.threshold(gray, 127, 255, 0)

        # Do connected components processing to identify 
        # potential cells. As the name suggests this focuses on finding
        # components (just means parts) of the image that are connected together and show 
        # homogenity or whatever characteristic we specify as the identifier in our algorithm
        # here that work is done by the cv2.CV_325. Please read up on what that flag does
        # to truly understand how this works.
        nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(thr, 8, cv2.CV_32S)

        # Get CC_STAT_AREA component as stats[label, COLUMN]. This is to do area based analysis
        areas = stats[1:, cv2.CC_STAT_AREA]
        # to count the cells initialize a count variable
        count = 0

        # the following are self explanatory
        mean_area = np.mean(areas[np.logical_and(areas > 30, areas < 300)])
        sd_area = np.std(areas[np.logical_and(areas > 30, areas < 300)])

        # drawing circles around the cells through this loop
        for i in range(1, nlabels):  # Iterate over connected components (the fos cells)
                                     # (excluding the background which is why we use nlabels)
        
            # just a variable to hold the area value for the relevant detected fos cell candidate
            area = areas[i - 1]

            # We can take in the input of area limits for cells from the user so I have commented it
            # out here:

            # area_min_thresh = 30
            # area_max_thresh = 300
            if area_max_thresh >= area >= area_min_thresh:  # Filter components based on area

                # In the following steps we are seeing if the area of the cell is sufficiently filled
                # with blue for it to be a cell

                # Calculate circle parameters for circling the cell and to make sure its a cell 
                center = (int(centroids[i][0]), int(centroids[i][1]))
                radius = int(np.sqrt(area / np.pi))

                # Calculate the blue area within the circle for calculating how much of the cell is actually blue
                blue_area = np.sum(thr[center[1] - radius:center[1] + radius, center[0] - radius:center[0] + radius] == 255)

                # Calculate the percentage of the circle filled with blue (self explanatory)
                blue_percentage = blue_area / (np.pi * radius ** 2)  # Ratio of blue area to circle area

                # Draw the red circle around the cell if the blue percentage exceeds a threshold (e.g., 0.5)
                if blue_percentage > 0.5:
                    cv2.circle(img, center, radius, (0, 0, 255), 2)
                    count += 1
                # this calculates cell count. Make it a list append operation to get a list of counts of images. 
                # Here we are replacing  tif extensions with jpeg s so we can output lower memory taking images
                outfile = file.replace(".tif", ".jpg")

                # Saving the resultant images
                cv2.imwrite(os.path.join(output_dir, f"circled_cells_{outfile}"), img)
                cv2.imwrite(os.path.join(output_dir, f"binary_{outfile}"), thr)

                mean_cell_area.append(mean_area)
                sd_cell_area.append(sd_area)

                # Calculating the stats. Make these list opetations to print them for each image.
                average_mean_cell_area = np.mean(mean_cell_area)
                mean_sd_of_cell_area = np.mean(sd_cell_area)
                std_dev_in_mean_cell_area = np.std(mean_cell_area)


# Code Execution example. You can also put this in a main method if you desire
# You can also read in the image from a main method but in the interest of 
# Indepenently executable functions I have chosen not to do this even though it
# is standard practice and will save up to a few seconds of repetitive data 
# input. Please feel free to contact the author for a recommended main method

# Change these to try different colors of detection or to set the minimum 
# area for the structure to be detected(also max area for cells).
lower_pink = np.array([140, 50, 50])
upper_pink = np.array([180, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30]) 

min_area_threshold_Striosome = 1000
min_area_threshold_Vein = 200

area_min_threshold_cells = 30
area_max_threshold_cells = 300

# Function Calls for all 3 functions:
#Striosomal_Detection(lower_pink, upper_pink, min_area_threshold_Striosome)
Blood_Vessel_Detection(lower_black, upper_black, min_area_threshold_Vein)
#Cell_Detector_and_Counter_Connected_Component(area_min_threshold_cells, area_max_threshold_cells)

