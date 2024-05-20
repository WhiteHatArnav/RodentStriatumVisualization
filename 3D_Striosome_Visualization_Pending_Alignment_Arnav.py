# See Readme or the 2D Core script for an explanation of these libraries
import cv2
import numpy as np
from tqdm.auto import tqdm
import os

# Plotly has an interactive output interface in 3-D unlike matplotlibs and has hence been chosen here
# The way we are using it uses a form of OOP that makes the plotting task have the advantages that come
# with using objects instea of just a simple script with no OOP
# The working and functionality of plotly is well known but still complex enough to not fit in the
# reasonable comment space we want here but there is extensive documentation available online for plotly
import plotly.io as pio
import plotly.graph_objs as go
 
# I have not made the input parameters dynamic here since we will need to determine 
# the ideal color thresholds and areas. 

# This function is just the code we have on the 2-D script for strio detection just slightly edited
def Striosome_Detector(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 100, 100])  
    upper_pink = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_img, lower_pink, upper_pink)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # I have made minimum area threshold higher based on striosomes in the cross section images
    # for this use case
    min_area_threshold = 5000  
    filtered_contours = [cnt for cnt in tqdm(contours, desc="Striosome Detection") if cv2.contourArea(cnt) > min_area_threshold]
    return filtered_contours


# Alignment function for blood vessels
#def Align_Blood_Vessels():
    # You can either make another function for detecting the large blood vessels of striosomes or do that in here
   


# This is the dir containing cross-section images
cross_secs_dir = '/Users/arnavjoshi/Desktop/EnhancedCrossSecs'

# This time I will read the images into a list because for operations on 
# serial cross sections to visualize the striosome in 3D the order of the 
# images is important and from this list we can tell what order the images 
# were read in.
                
input_imgs = []
for filename in os.listdir(cross_secs_dir):
    img = cv2.imread(os.path.join(dir, filename))
    input_imgs.append(img)

# Extract the shape and fill (contour) of striosomes detected in the cross section images
# Keep in mind that this puts all 8 cross section contours into a single variable
# For a simpler iterative structure later
Cross_Sec_contours = [Striosome_Detector(img) for img in input_imgs]

# We will use this list to store the trace of each cross section image
# which will become a 3-D scatterplot of points. This will provide the most 
#
scatter_points = []

# Plot each striosome and build a surface by filling gaps caused by plotting 2-D data in 3-D
# (This is a slightly hard concept to simplify but feel free to contact the author if you
# have questions)
for i, cnt_set in enumerate(Cross_Sec_contours):
    for cnt in tqdm(cnt_set, desc = f"Plotting Striosomes from (Cross-section Number: {i+1})"):
        
        # In the following set of code lines I have done a fill and extend operation horizontally
        # on the image so that irregular border areas and distorted striosome cross sections by big
        # blood vessels become a cohesive solid object on the X-Y plane while maintaining shape accuracy

        # While I can easily extend this to the Z axis but we need proper alignment to guide that Fill and 
        # Extend Operation. I would recommend using this same approach but just let the alignment data guide
        # the filling of the Z gaps an also shift the plot points according to that.

        # The logic of shifting is relatively simple and that of filling is given here. All that is important
        # to do now is to perfect the alignment logic. call the Align_Blood_Vessels method here 
        # when it is complete and then add the logic described above

        # We need a list for the location of the co-ordinates to fill with pink and  extend
        # to generate the 3-D plot
        plot_cds = []

        # We can use x or y  co-ords to cycle through the pixels, here I use y co-ords
        # (these images are usually wider than taller so this is slightly better)
        # but depending on how the striatum is centered that may change.
        for y_cd in range(input_imgs[i].shape[0]):

            #extracting the applicable y_cds
            row = cnt[np.where(cnt[:, :, 1] == y_cd)[0]]

            # we want to avoid empty rows because that will significantly add computation time
            # and not add anything to the plot unless our alignment algorithm is able to specifically
            # give us x and y co-ordinates
            if len(row) != 0:

                #finding minimym and maximum x points for left and right boundary
                x_cd_left_bound = min(row[:, 0, 0])
                x_cd_right_bound = max(row[:, 0, 0])

                # now we use the extend method to get the filled and extended point data set for 
                # the co-ordinates we need
                plot_cds.extend([(x, y_cd, i) for x in range(x_cd_left_bound, x_cd_right_bound + 1)])

        # This step is optional based on what device and computing power and display system you are using
        # It is a process called downsampling in which you reduce the precision of the point definitions
        # by a multiplicative factor
                
        # Reduce sample gradient to every dwn_ratio(th) sample point
        dwn_ratio = 10
        dwnsmp_plot_cds = [plot_cds[j] for j in range(0, len(plot_cds), dwn_ratio)]  

        # Append the downsampled filled points to data with adjusted opacity
        scatter_points.append(go.Scatter3d(x=[p[0] for p in  dwnsmp_plot_cds], y=[p[1] for p in  dwnsmp_plot_cds], 
                                 z=[p[2] for p in  dwnsmp_plot_cds], mode='markers',marker=dict(color='purple'), 
                                  opacity=0.5))  




# Define the layout of the plot. It is easier to read documentation on this method than to comment
# each detail here but please feel free to ask the author for any clarifications
spread = go.Layout(scene = dict(aspectmode = 'manual', aspectratio = dict(x = 1, y = 1, z = 0.1)),
                   scene_camera = dict(eye = dict(x = 1.5, y = -1.5, z = 0.5)))
# Here we create the Figure object from our scatter points and the layount we defined
fig = go.Figure(data = scatter_points, layout = spread)

# Show the plot the figure object created
fig.show()

# Save the resultant plot
pio.write_html(fig, file='//Users/arnavjoshi/Desktop/cross_section_shaped_partially_aligned_plot_Arnav_Joshi.html', auto_open = True)


