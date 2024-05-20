# Code Author: Arnav Joshi
# Date of online update on Github: May 19th, 2024
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
# The code for cell detection and counting will be more complex since there is quite a bit of 
# processing and some calculations that have been carried out
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
        # Iterate over connected components (the fos cells)
        # (excluding the background which is why we use nlabels)
        
        for i in tqdm(range(1, nlabels), leave = False):  
     
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



# Set min and max area for cells:
area_min_threshold_cells = 30
area_max_threshold_cells = 300

# Function Call:
Cell_Detector_and_Counter_Connected_Component(area_min_threshold_cells, area_max_threshold_cells)

