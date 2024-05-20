# RodentStriatumVisualization

# Details about the code:
Description: This is a Python Programmes to identify and outline structures in Rodent Striatum and Visualie it in 3-D.
Code Author: Arnav Joshi
Date of online upload on Github: May 19th, 2024
Author Email: arnavj@alumni.princeton.edu

# Aim of this Readme
This Readme will provide general information about the code in this repository. Specific details of the scripts are present in each script as comments.


# Some Remarks about this code and its uniqueness and applicability: 
The program is able to detect irregularly colored and sized tissue structures from a dynamic script. It represents serious work in the direction of a breakthrough in understanding biological tissue structures which implicates possible utility in areas ranging from oncology to neuroscience.

# Here is a general summary of each of the Python Scripts in this Repository as of May 19th, 2024: 

# 2-D Detection and Analysis

Striosome_Detector_Arnav: This  script contains a dynamic method that detects the Striosomes in a fluoroscent died cross-section image of the rodent striosome and outlines them clearly using thresholding and contour detecion from the CV2 module and an example of its execution. The Script also is setup to take as many images as needed from an input directory and paste the annotated image into an output directory. The area thresholds and color thresholds can be user defined so it can be used in any form of tissue structure detection from a 2-D image with distinctly colored features.

Blood_Vessel_Detector_Arnav: This  script also contains a dynamic method but this time the metho detects the Blood Vessels in a fluoroscent died cross-section image of the rodent striosome using thresholding and contour detecion from the CV2 module and an example of its execution. The difference here is only in color limits and some minor syntax differences to allow anyone using the code to be familiar with other ways to approach non analytical parts of the code.

Cell_Counter_Connected_Component_Arnav: This script uses the CV2 connectedComponentsWithStats() method for cell detection. While this requires some pre-processing its a good approach to cell detecion along with just simple contour detection. This script has considerable image pre processing like grayscaling then binarization and so is very sensitive to image quality and the thresholds for cell color as well but it is definitely a viable approach to fos cell identification. The unique feature of this script is that it can identify cells based on a threshold of how much of the cell area is actually filled with the desied color that means fos positive. Right now I have made the condition for a cell being full enough in its circular zone to be at a 50% threshold of the area being blue. This will usually underdetect so please feel free to move it up and down based on what you feel is the fullness coefficient that is appropriate.

CoreScript2DAnnotationandAnalysis_Arnav: This Script defines and executes all 3 of the methods mentioned above. It is what I would recommend be the format of a final program that is made because making the scripts dynamic then integrating them into one logical structure along with parallelization(not done here) and machine learning training algorithms to learn from the outcomes(does work but we will need larger data set so its not in this code), and a user friendly input GUI could give us a software to detect histological features. Software like this already exists but there is always room for improvement and adding features like even more flexible thresholding and the ability to use more abstract concepts like fullness of round areas to detect cells. This code is very heavy computationally unless you trim down to relevant sections of the image and its not because of any computation but the sheer number of cells.

# 3-D Detection and Analysis

3D_Striosome_Visualization_Pending_Alignment_Arnav: This code is set up to be able to take a sequence of any number of cross-sections and them map the extracted striosomes from them in 3-D space in a stacked structure using the same logic as in out 2-D Striosome detector. The more fine descriptions are in the code's comments but essentially it uses a combination of scatterploting from plotly and defining a layout as well along with a fill and extend approach on the x-y cartesian plane for each cross section to fill gaps from distortions and areas of striosomes that are not as distinct due to lack of vertical depth. This same fill and extend logic can be applied to the z direction using alignment based on blood vessel locations in each cross section. The only part that needs to be added is the alignment and shifting algorithm to stack the cross sections in a vertically aligned manner.



