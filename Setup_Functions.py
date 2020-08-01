from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import pandas as pd
from xml.etree import ElementTree
from PIL import Image
from mrcnn.utils import Dataset
from matplotlib.patches import Rectangle
import random
import time
import math
import glob
import cv2
import numpy as np
from numpy import random
from numpy import zeros
from numpy import asarray
import sys
import statistics
import PIL
import torchvision
import torch
torch.set_grad_enabled(False)
import matplotlib
import matplotlib.pylab as plt
from scipy import signal
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model

from numpy import expand_dims
from numpy import mean
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import matplotlib.pyplot as pyplot

#For Human Pose Analysis
plt.rcParams["axes.grid"] = False
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model = model.eval().cuda()

def vis_keypoints(img, kps, draw, kp_thresh=2, alpha=0.7):
    #Returns Fencer_Data from Human Pose
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    #The keypoints of interest are [left wrist, right wrist, left knee, right knee, left shoulder, right shoulder] with [x,y,confidence] for each point
    fencer_data = []
    fencer_kp = [9,10,13,14,5,6]

    for keypoint in fencer_kp:    
      fencer_data.append([int(kps[0][keypoint]),int(kps[1][keypoint]),int(kps[2][keypoint])])

    if draw == True:
      # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
      cmap = plt.get_cmap('rainbow')
      colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
      colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

      # Perform the drawing on a copy of the image, to allow for blending.
      kp_mask = np.copy(img)

      # Draw mid shoulder / mid hip first for better visualization.
      mid_shoulder = (
          kps[:2, dataset_keypoints.index('right_shoulder')] +
          kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
      sc_mid_shoulder = np.minimum(
          kps[2, dataset_keypoints.index('right_shoulder')],
          kps[2, dataset_keypoints.index('left_shoulder')])
      mid_hip = (
          kps[:2, dataset_keypoints.index('right_hip')] +
          kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
      sc_mid_hip = np.minimum(
          kps[2, dataset_keypoints.index('right_hip')],
          kps[2, dataset_keypoints.index('left_hip')])
      nose_idx = dataset_keypoints.index('nose')
      if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
          cv2.line(
              kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
              color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
      if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
          cv2.line(
              kp_mask, tuple(mid_shoulder), tuple(mid_hip),
              color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

      # Draw the keypoints.
      for l in range(len(kp_lines)):
          i1 = kp_lines[l][0]
          i2 = kp_lines[l][1]
          p1 = kps[0, i1], kps[1, i1]
          p2 = kps[0, i2], kps[1, i2]
          if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
              cv2.line(
                  kp_mask, p1, p2,
                  color=colors[l], thickness=2, lineType=cv2.LINE_AA)
          if kps[2, i1] > kp_thresh:
              cv2.circle(
                  kp_mask, p1,
                  radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
          if kps[2, i2] > kp_thresh:
              cv2.circle(
                  kp_mask, p2,
                  radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return [cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0), fencer_data]

def overlay_keypoints(image, kps, scores, draw):
  kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).cpu().numpy()
  fencer_data = []
  for region in kps:
    [image, fencer_data_temp] = vis_keypoints(image, region.transpose((1, 0)), draw)
    fencer_data.append(fencer_data_temp)
  return (image, fencer_data)

def fencer_data_compact(fencer_data):
  # Abbreviates Fencer Data to Relevant Data
  # Condenses fencer_data to (Left, Right) (Weapon Hand, Front Knee, Shoulder Center)

  fencer_data_compact = []
  fencer_wrist = []
  fencer_knee = []
  shoulder_center = []
  keypoints = [fencer_wrist,fencer_knee,shoulder_center]

  #Sorts the 4 datapoints with respect to x and returns the center two points
  fencer_wrist.append(fencer_data[0][0])
  fencer_wrist.append(fencer_data[0][1])
  fencer_wrist.append(fencer_data[1][0])
  fencer_wrist.append(fencer_data[1][1])
  fencer_wrist = sorted(fencer_wrist, key = lambda x: x[0])
  fencer_data_compact.append(fencer_wrist[1:3])

  fencer_knee.append(fencer_data[0][2])
  fencer_knee.append(fencer_data[0][3])
  fencer_knee.append(fencer_data[1][2])
  fencer_knee.append(fencer_data[1][3])
  fencer_knee = sorted(fencer_knee, key = lambda x: x[0])
  fencer_data_compact.append(fencer_knee[1:3])

  for i in range(2):
    shoulder_temp = []
    for j in range(len(fencer_data[i][4])):
      shoulder_temp.append(int((fencer_data[i][4][j] + fencer_data[i][5][j])/2))    
    shoulder_center.append(shoulder_temp)
  shoulder_center[0], shoulder_center[1] = shoulder_center[1], shoulder_center[0]
  fencer_data_compact.append(shoulder_center)

  return (fencer_data_compact)

def human_pose_analysis(frame):
  # image = PIL.Image.open(file_name)
  image = frame
  image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
  output = model([image_tensor])[0]

  result_image = np.array(image.copy())

  #Uses only six keypoints
  [result_image, fencer_data] = overlay_keypoints(result_image, output['keypoints'][:6], output['keypoints_scores'][:6], True)

  keypoints = [output['keypoints'][:2], output['keypoints_scores'][:2]]

  return (fencer_data, keypoints)

def fencer_data_verification(Left_Torso_Position, left_torso_size_average, Right_Torso_Position, right_torso_size_average, fencer_data, frame):
  #Tests that the fencer_pose_data is near the torso of the fencer
  #Left and Right Torso positions are single x,y values in this function
  #Format fencer_data to (Left, Right) (Weapon Hand, Front Knee, Shoulder Center)
    #[[wristLx, wristLy, wristLconf][wristRx, wristRy, wristRconf]],[[kneeLx, kneeLy, kneeLconf][kneeRx, kneeRy, kneeRconf]],[[shldrLx, shldrLy, shldrLconf][shldrRx, shldrRy, shldrRconf]]
  #Format torso_size_average [width, height]

  fencer_data_pose_left = []
  fencer_data_pose_right = []

  for i in range(len(fencer_data)):
    #Creates the Left Fencer_Data
    lx_min = min(fencer_data[i][4][0],fencer_data[i][5][0]) - left_torso_size_average[1]/4
    lx_max = max(fencer_data[i][4][0],fencer_data[i][5][0]) + left_torso_size_average[1]/4
    ly_min = min(fencer_data[i][4][1],fencer_data[i][5][1])
    ly_max = max(fencer_data[i][4][1],fencer_data[i][5][1])
    torso_l_x = Left_Torso_Position[0]
    torso_l_y_bottom = Left_Torso_Position[1] + left_torso_size_average[1]/2
    torso_l_y_top = Left_Torso_Position[1] - left_torso_size_average[1]/2
    #Checks if the torso is between the shoulders, that the shoulders and within the torso_box and that the left pose is empty

    display(f'The left fencer data verification bounding box is:')
    display(f'{lx_min} to {lx_max} in the x direction and {torso_l_y_bottom} to {torso_l_y_top}')
    display(f'The center points are: {torso_l_x} for x and {ly_min},{ly_max} for y.')

    if torso_l_x > lx_min and torso_l_x < lx_max and torso_l_y_top < ly_min and torso_l_y_bottom > ly_max and fencer_data_pose_left == []:
      #Checks for which wrist and knee is forward
      if fencer_data[i][0][0] > fencer_data[i][1][0]:
        fencer_data_pose_left.append(fencer_data[i][0])
      else:
        fencer_data_pose_left.append(fencer_data[i][1])
      if fencer_data[i][2][0] > fencer_data[i][3][0]:
        fencer_data_pose_left.append(fencer_data[i][2])
      else:
        fencer_data_pose_left.append(fencer_data[i][3])
      fencer_data_shldr_temp = []
      fencer_data_shldr_temp.append(int((fencer_data[i][4][0] + fencer_data[i][5][0])/2))
      fencer_data_shldr_temp.append(int((fencer_data[i][4][1] + fencer_data[i][5][1])/2))
      fencer_data_shldr_temp.append(int((fencer_data[i][4][2] + fencer_data[i][5][2])/2))
      fencer_data_pose_left.append(fencer_data_shldr_temp)
      
  for i in range(len(fencer_data)):
    #Creates the Right Fencer_Data
    rx_min = min(fencer_data[i][4][0],fencer_data[i][5][0]) - left_torso_size_average[1]/4
    rx_max = max(fencer_data[i][4][0],fencer_data[i][5][0]) + left_torso_size_average[1]/4
    ry_min = min(fencer_data[i][4][1],fencer_data[i][5][1])
    ry_max = max(fencer_data[i][4][1],fencer_data[i][5][1])
    torso_r_x = Right_Torso_Position[0]
    torso_r_y_bottom = Right_Torso_Position[1] + right_torso_size_average[1]/2
    torso_r_y_top = Right_Torso_Position[1] - right_torso_size_average[1]/2
    if torso_r_x > rx_min and torso_r_x < rx_max and torso_r_y_top < ry_min and torso_r_y_bottom > ry_max and fencer_data_pose_right == []:
      #Checks for which wrist is forward
      if fencer_data[i][0][0] > fencer_data[i][1][0]:
        fencer_data_pose_right.append(fencer_data[i][1])
      else:
        fencer_data_pose_right.append(fencer_data[i][0])
      #Checks for which knee is forward
      if fencer_data[i][2][0] > fencer_data[i][3][0]:
        fencer_data_pose_right.append(fencer_data[i][3])
      else:
        fencer_data_pose_right.append(fencer_data[i][2])
      fencer_data_shldr_temp = []
      # Averages the Shoulder data
      fencer_data_shldr_temp = []
      fencer_data_shldr_temp.append(int((fencer_data[i][4][0] + fencer_data[i][5][0])/2))
      fencer_data_shldr_temp.append(int((fencer_data[i][4][1] + fencer_data[i][5][1])/2))
      fencer_data_shldr_temp.append(int((fencer_data[i][4][2] + fencer_data[i][5][2])/2))
      fencer_data_pose_right.append(fencer_data_shldr_temp)

  # Condenses the fencer data to only relevant data
  fencer_data = [fencer_data_pose_left, fencer_data_pose_right]

  # If no pose is found, then it is set to zeros
  if fencer_data[0] == []:
    fencer_data[0] = [[0,0,0],[0,0,0],[0,0,0]]
  if fencer_data[1] == []:
    fencer_data[1] = [[0,0,0],[0,0,0],[0,0,0]]

  if verbose == True:
    display(f'The compact fencer data from verification frame {frame - 1} is:')
    display(fencer_data)

  return (fencer_data)

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

def position_linear_approximation(position, previous_certainty):
  # Certainty is the number of times previous to current position that a point was not certain.
  last_known_position = ((previous_certainty+2)*(-1))

  # Finds the positional distance between two known boxes
  x_delta = int((position[-1][0] - position[last_known_position][0])/(last_known_position+1))
  y_delta = int((position[-1][1] - position[last_known_position][1])/(last_known_position+1))
  delta = [x_delta, y_delta]

  # Adjusts the previous positions, up to the previous certainty, based on a linear approximation
  for j in range(2):
    for i in range(previous_certainty+1):
      position[i - (previous_certainty+1)][j] = position[i - (previous_certainty+2)][j] - delta[j]

  return (position)

def scoring_box_lights(img, Scoring_Box_Position, scoring_box_size_average, default_color, frame, score_box_empty):

  # A high max distance is less sensitive and a lower max distance is more sensitive
  max_distance_total = 200
  max_distance_specific_color = 100

  # Defines the region of the top_left position of a 5x3 grid of the score_box, [xmin,ymin,xmax,ymax]
  # Extends the Light Search Position outside of the detected box
  xmin = Scoring_Box_Position[0] - int(scoring_box_size_average[0]/2) - int(scoring_box_size_average[0]/8)
  xmax = Scoring_Box_Position[0] - int(scoring_box_size_average[0]/2) + int(scoring_box_size_average[0]/4)
  ymin = Scoring_Box_Position[1] - int(scoring_box_size_average[1]/2)
  ymax = Scoring_Box_Position[1] - int(scoring_box_size_average[1]/2) + int(scoring_box_size_average[1]/3)
  left_light_position = [xmin, xmax, ymin, ymax]

  # Defines the region of the top_right position of a 5x3 grid of the score_box, [xmin,ymin,xmax,ymax]
  xmin = Scoring_Box_Position[0] + int(scoring_box_size_average[0]/2) - int(scoring_box_size_average[0]/4)
  xmax = Scoring_Box_Position[0] + int(scoring_box_size_average[0]/2) + int(scoring_box_size_average[0]/8)
  ymin = Scoring_Box_Position[1] - int(scoring_box_size_average[1]/2)
  ymax = Scoring_Box_Position[1] - int(scoring_box_size_average[1]/2) + int(scoring_box_size_average[1]/3)
  right_light_position = [xmin, xmax, ymin, ymax]

  if default_color != []:
    distance_temp, distance_specific_color_temp = [], []

    width = left_light_position[1]-left_light_position[0]
    height = left_light_position[3]-left_light_position[2]

    #i is the x value of the image for the Left Side/Red
    for i in range(width):
      #j is y value of the image
      for j in range(height):
        #color channel of the image [B,G,R]
        #image, img, is of format [y,x]
        pixel_position_y = left_light_position[2] + j
        pixel_position_x = left_light_position[0] + i
        b = (img[pixel_position_y, pixel_position_x, 0] - default_color[0])
        g = (img[pixel_position_y, pixel_position_x, 1] - default_color[1])
        r = (img[pixel_position_y, pixel_position_x, 2] - default_color[2])
        distance_temp.append(int((b**2 + g**2 + r**2)**(0.5)))
        distance_specific_color_temp.append(abs(r))

    #Sorts the distances and keeps the top quarter then finds the average
    distance_temp.sort()
    distance_temp = distance_temp[(int(len(distance_temp)/4)*-1):]
    distance = int(sum(distance_temp)/len(distance_temp))
    distance_specific_color_temp.sort()
    distance_specific_color_temp = distance_specific_color_temp[(int(len(distance_specific_color_temp)/4)*-1):]
    distance_specific_color = int(sum(distance_specific_color_temp)/len(distance_specific_color_temp))

    #0 is no color change from the default color)
    if distance > max_distance_total and distance_specific_color > max_distance_specific_color and score_box_empty == False:
      left_light_comparison = 1
    #1 is a color change from the default color
    else:
      left_light_comparison = 0

    #Resets b,g,r for the Right Side
    distance_temp, distance_specific_color_temp= [], []
    width = right_light_position[1]-right_light_position[0]
    height = right_light_position[3]-right_light_position[2]

    #i is the x value of the image
    for i in range(width):
      #j is y value of the image
      for j in range(height):
        #kcolor channel of the image [B,G,R]

        # pixel_position = right_light_position[2] + j,right_light_position[0] + i
        pixel_position_y = right_light_position[2] + j
        pixel_position_x = right_light_position[0] + i
        b = (img[pixel_position_y, pixel_position_x, 0] - default_color[0])
        g = (img[pixel_position_y, pixel_position_x, 1] - default_color[1])
        r = (img[pixel_position_y, pixel_position_x, 2] - default_color[2])
        distance_temp.append(int((b**2 + g**2 + r**2)**(0.5)))
        distance_specific_color_temp.append(abs(g))

    #Sorts the distances and keeps the top sixth then finds the average
    distance_temp.sort()
    distance_temp = distance_temp[(int(len(distance_temp)/6)*-1):]
    distance = int(sum(distance_temp)/len(distance_temp))
    distance_specific_color_temp.sort()
    distance_specific_color_temp = distance_specific_color_temp[(int(len(distance_specific_color_temp)/4)*-1):]
    distance_specific_color = int(sum(distance_specific_color_temp)/len(distance_specific_color_temp))

    #0 is no color change from the default color)
    if (distance > max_distance_total and distance_specific_color > max_distance_specific_color):
      right_light_comparison = 1
    #1 is a color change from the default color
    else:
      right_light_comparison = 0

  #Finds the Defualt Color
  else:
    b, g, r = 0, 0, 0
    # Cycles through the Left and Right Light Positions to determine a default color for the frame
    width = left_light_position[1]-left_light_position[0]
    height = left_light_position[3]-left_light_position[2]
    for i in range(width):
      for j in range(height):
        pixel_position_y = left_light_position[2] + j
        pixel_position_x = left_light_position[0] + i
        b = b + img[pixel_position_y, pixel_position_x, 0]
        g = g + img[pixel_position_y, pixel_position_x, 1]
        r = r + img[pixel_position_y, pixel_position_x, 2]
        default_color_left_temp = [int(b/(width*height)),int(g/(width*height)),int(r/(width*height))]
    width = right_light_position[1]-right_light_position[0]
    height = right_light_position[3]-right_light_position[2]
    for i in range(width):
      for j in range(height):
        # pixel_position = right_light_position[2] + j,right_light_position[0] + i
        pixel_position_y = left_light_position[2] + j
        pixel_position_x = left_light_position[0] + i
        b = b + img[pixel_position_y, pixel_position_x, 0]
        g = g + img[pixel_position_y, pixel_position_x, 1]
        r = r + img[pixel_position_y, pixel_position_x, 2]
        default_color_right_temp = [int(b/(width*height)),int(g/(width*height)),int(r/(width*height))]
    #Combines the Left and Right Default Colors for B,G,R
    for i in range(3):
      default_color.append((default_color_left_temp[i] + default_color_right_temp[i])/2)

    # Assumes that the lights are off during the engarde phase.
    left_light_comparison = 0
    right_light_comparison = 0

  return (left_light_comparison, right_light_comparison, default_color)

def motion_difference_tracking(frame, side, Bounding_Box, width, height, kernel_scaling, erosion_iterations, dilation_iterations):

  # Ensures Bounding_Box is not negative
  for i in range(len(Bounding_Box)):
    if Bounding_Box[i] < 0:
      Bounding_Box[i] = 0

  display(f'The original difference tracking bounding box at frame {frame - 1} is:')
  display(Bounding_Box)

  Position_y_Orig = int((Bounding_Box[3]+Bounding_Box[2])/2)

  # Uses the original frames to avoid Region of Interest Boxes
  save_path = r'/content/Mask_RCNN/videos/original/'
  image_num = frame
  image_name2 = str(image_num-1) + '.jpg'
  image_name1 = str(image_num-2) + '.jpg'
  file_name1 = os.path.join(save_path, image_name1)
  file_name2 = os.path.join(save_path, image_name2)

  # Reads the images
  image1 = cv2.imread(file_name1)
  image2 = cv2.imread(file_name2)

  # Convert to Grayscale
  image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  image_diff = cv2.absdiff(image1_gray,image2_gray)

  # Creates a Cropped Image
  crop_img = image_diff[Bounding_Box[2]:Bounding_Box[3], Bounding_Box[0]:Bounding_Box[1]]

  # Kernel is affected by Kernel Scaling which gets finer if it initially fails
  kernel_number = int(width/(100*kernel_scaling))
  
  # Ensures that the kernel is odd
  if kernel_number%2 == 0:
    kernel_number = kernel_number + 1
  kernel = np.ones((kernel_number,kernel_number),np.uint8)
  
  try:
    # Errodes
    erosion = cv2.erode(crop_img,kernel,iterations = erosion_iterations)

    # Dilates
    dilation = cv2.dilate(erosion,kernel,iterations = dilation_iterations)

    # Blurs Image
    blur = cv2.GaussianBlur(dilation,kernel.shape,0)

    # Threshold
    ret,thresh = cv2.threshold(blur,0,90,cv2.THRESH_BINARY)

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    c = max(cnts, key=cv2.contourArea)

    if side == 'Left':
      # Obtain outer left coordinate of the contour
      left = tuple(c[c[:, :, 0].argmin()][0])
      position = [right[0] + Bounding_Box[0], Position_y_Orig]
    elif side == 'Right':
      right = tuple(c[c[:, :, 0].argmax()][0])
      position = [left[0] + Bounding_Box[0], Position_y_Orig]
    else:
      display(f'Side is not given')
  # Error occurs if the entire image is erroded
  except:
    display(f'There is no data from difference imaging on the {side} side.')
    position = 'None'

  return(position)

def saturation_test(box, frame):
  # Test is a True/False return
  # Takes an image and tests it for the expected saturation

  path = r'/content/Mask_RCNN/videos/save/'
  file_name = str(frame) + '.jpg'
  name = os.path.join(path, file_name)
  img = cv2.imread(name)
  # Converts from BGR to HSV
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  
  # Tests Bellguard
  if box[2] == 1:
    blue_range = [50, 150]
    green_range = [50, 150]
    red_range = [50, 160]
    max_delta = 25
    # saturation_range = [0, 20]
    saturation_range = [0, 70]
    object_tested = 'Bellguard'
  # Tests Torso
  elif box[2] == 3:
    blue_range = [60, 150]
    green_range = [60, 150]
    red_range = [60, 160]
    max_delta = 30
    saturation_range = [0, 20]
    object_tested = 'Torso'
  else:
    display(f'The object to test does not have a color/saturation profile.')

  width = (box[0][3]-box[0][1])
  height = (box[0][2]-box[0][0])

  s_temp = []

  #i is the x value of the image
  for i in range(width):
    #j is y value of the image
    for j in range(height):
      s = img[box[0][0] + j, box[0][1] + i, 1]
      s_temp.append(s)

    #Sorts the distances and keeps the top quarter then finds the average
    s_temp.sort()
    #Truncates to the least saturated/most gray values
    s_temp = s_temp[:(int(len(s_temp)/2)*-1)]
    s_temp = s_temp[:(int(len(s_temp)*3/4)*-1)]
    #Averages the saturation values
    s_average = int(sum(s_temp)/len(s_temp))

  if s_average < saturation_range[1]:
    test_result = True
  else:
    test_result = False

  display(f'The test result for the {object_tested} saturation is {test_result} with a saturation of {s_average}.')

  return (test_result)

def box_size_finder(bbox, capture_width, capture_height, object_to_size):

  Box_Size = [[],[]]
  sum_of_boxes = [[],[]]
  frame_multiplier = 1

  if object_to_size == 'score_box':
    x_min = int(capture_width/4)
    x_max = int(capture_width*3/4)
    bbox_category = 2
  elif object_to_size == 'left':
    x_min = 0
    x_max = int(capture_width/2)
    bbox_category = 3
  elif object_to_size == 'right':
    x_min = int(capture_width/2)
    x_max = int(capture_width)
    bbox_category = 3

  # i represents the frame, minimum of 50 frames or len(bbox)
  for i in range(min(50*frame_multiplier, len(bbox))):
    # j represents the rois(specific bounding box) within the frame sorted by confidence score
    for j in range(len(bbox[i])):
      if (bbox[i][j][1] > 0.90 and bbox[i][j][0][1] > x_min and bbox[i][j][0][1] < x_max and bbox[i][j][2] == bbox_category):
        #Appends x value:
        sum_of_boxes[0].append(bbox[i][j][0][1])
        #Appends y value:
        sum_of_boxes[1].append(bbox[i][j][0][0])  
        #Appends x width value:
        Box_Size[0].append(bbox[i][j][0][3] - bbox[i][j][0][1])
        #Appends y width value:
        Box_Size[1].append(bbox[i][j][0][2] - bbox[i][j][0][0])

  x_average = average_list(sum_of_boxes[0])
  y_average = average_list(sum_of_boxes[1])

  # scoring_box_size_average [Width, Height]
  box_size_average = []
  # Appends the average scoring box width
  box_size_average.append(int(average_list(Box_Size[0])))
  # Appends the average scoring box height
  box_size_average.append(int(average_list(Box_Size[1])))

  display(f'The Average Box Size for {object_to_size} is {box_size_average}')

  return (box_size_average)

def tracking_box_default(Left, Right, Score_Box, x_padding, y_padding, engarde_length):
  # Creates a default tracking box

  Tracking_Bounding_Boxes_Temp = [[],[],[]]
  Tracking_Bounding_Boxes = []

  for i in range(engarde_length):
    Tracking_Bounding_Boxes_Temp[0].append(Left[0] - x_padding)
    Tracking_Bounding_Boxes_Temp[0].append(Left[0] + x_padding)
    Tracking_Bounding_Boxes_Temp[0].append(Left[1] - y_padding)
    Tracking_Bounding_Boxes_Temp[0].append(Left[1] + y_padding)

    Tracking_Bounding_Boxes_Temp[1].append(Right[0] - x_padding)
    Tracking_Bounding_Boxes_Temp[1].append(Right[0] + x_padding)
    Tracking_Bounding_Boxes_Temp[1].append(Right[1] - y_padding)
    Tracking_Bounding_Boxes_Temp[1].append(Right[1] + y_padding)

    Tracking_Bounding_Boxes_Temp[2].append(Score_Box[0] - x_padding)
    Tracking_Bounding_Boxes_Temp[2].append(Score_Box[0] + x_padding)
    Tracking_Bounding_Boxes_Temp[2].append(Score_Box[1] - y_padding)
    Tracking_Bounding_Boxes_Temp[2].append(Score_Box[1]+ y_padding)

    Tracking_Bounding_Boxes.append(Tracking_Bounding_Boxes_Temp)

  return (Tracking_Bounding_Boxes)

def Bell_Guard_Position_Finding(bbox, capture_width, capture_height, fencer_data, positions, frame_count, left_torso_size_average, right_torso_size_average, engarde_length, previous_certainty, camera_steady, camera_motion_threshold):
  #Format positions = [Left_Position, Right_Position, Score_Box_Position, Left_Torso_Position, Right_Torso_Position]

  x_min = []
  x_max = []
  y_min = []
  y_max = []

  Left_Position = positions[0]
  Right_Position = positions[1]
  Scoring_Box_Position = positions[2]
  Left_Torso_Position = positions[3]
  Right_Torso_Position = positions[4]

  # Any of the First engarde_length position can be used since the engarde position is an averaged constant
  # Certainty is used here as a counter for how many times a bounding box does not fall in the tracking box
  # And increases the size of the bounding box based on each miss

  certainty = [0,0,0,0,0]
  display(f'Previous Certainty at frame {frame_count - 1} is {previous_certainty}.')

  #Establishes Previous Positions to determine speed and expected positions
  previous_position_Left = Left_Position[-1]
  twice_previous_position_Left = Left_Position[-2]
  previous_position_Right = Right_Position[-1]
  twice_previous_position_Right = Right_Position[-2]
  previous_position_Scoring_Box = Scoring_Box_Position[-1]
  twice_previous_position_Scoring_Box = Scoring_Box_Position[-2]
  previous_position_Left_Torso = Left_Torso_Position[-1]
  twice_previous_position_Left_Torso = Left_Torso_Position[-2]
  previous_position_Right_Torso = Right_Torso_Position[-1]
  twice_previous_position_Right_Torso = Right_Torso_Position[-2]

  #Boxes are the bounding boxes for the current frame, passes less data to tracking function
  boxes = bbox

  Tracking_Bounding_Boxes_Temp = [[],[],[],[],[]]

  # Torso Positions are calculated prior to the BellGuard because they are an input to the bellguard position

  # Bellguard Position Tracking focuses on Tracking as opposed to detection
  # Left_Torso Position
  [current_position, certainty[3], Tracking_Bounding_Boxes_Left_Torso] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Left_Torso, \
    twice_previous_position_Left_Torso, previous_certainty[3], 'Left_Torso', \
    frame_count, 'None', left_torso_size_average, capture_width, capture_height, 'None', engarde_length, camera_steady, camera_motion_threshold)
  Tracking_Bounding_Boxes_Temp[3] = Tracking_Bounding_Boxes_Left_Torso
  Left_Torso_Position = current_position

  # Right_Torso Position
  [current_position, certainty[4], Tracking_Bounding_Boxes_Right_Torso] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Right_Torso, \
    twice_previous_position_Right_Torso, previous_certainty[4], "Right_Torso", \
    frame_count, 'None', right_torso_size_average, capture_width, capture_height, 'None', engarde_length, camera_steady, camera_motion_threshold)
  Tracking_Bounding_Boxes_Temp[4] = Tracking_Bounding_Boxes_Right_Torso
  Right_Torso_Position = current_position

  fencer_data = fencer_data_verification(Left_Torso_Position, left_torso_size_average, Right_Torso_Position, right_torso_size_average, \
    fencer_data, frame_count)

  # Left Position
  [current_position, certainty[0], Tracking_Bounding_Boxes_Left] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Left, \
    twice_previous_position_Left, previous_certainty[0], 'Left_BellGuard', \
    frame_count, Left_Torso_Position, left_torso_size_average, capture_width, \
    capture_height, fencer_data, engarde_length, camera_steady, camera_motion_threshold)
  Tracking_Bounding_Boxes_Temp[0] = Tracking_Bounding_Boxes_Left
  Left_Position = current_position

  #  Right Position
  [current_position, certainty[1], Tracking_Bounding_Boxes_Right] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Right, \
    twice_previous_position_Right, previous_certainty[1], 'Right_BellGuard', \
    frame_count, Right_Torso_Position, right_torso_size_average, capture_width, \
    capture_height, fencer_data, engarde_length, camera_steady, camera_motion_threshold)
  Tracking_Bounding_Boxes_Temp[1] = Tracking_Bounding_Boxes_Right
  Right_Position = current_position

  # Scoring_Box Position
  [current_position, certainty[2], Tracking_Bounding_Boxes_Scoring_Box] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Scoring_Box, \
    twice_previous_position_Scoring_Box, previous_certainty[2], 'Scoring_Box', \
    frame_count, 'None', left_torso_size_average, capture_width, capture_height, 'None', engarde_length, camera_steady, camera_motion_threshold)
  Tracking_Bounding_Boxes_Temp[2] = Tracking_Bounding_Boxes_Scoring_Box
  Scoring_Box_Position = current_position

  Tracking_Bounding_Boxes = Tracking_Bounding_Boxes_Temp

  display(f'The Length of the Left and Right Positions after the Position Finding are: {len(Left_Position)} and {len(Right_Position)}.')

  display(f'At frame {frame_count} the certainty and previous certainty before linear approx analysis is:')
  display(f'{certainty} and {previous_certainty}')

  return (Left_Position, Right_Position, Scoring_Box_Position, Tracking_Bounding_Boxes, Left_Torso_Position, Right_Torso_Position, engarde_length, certainty)

def Bell_Guard_Position_Tracking(boxes, previous_position, twice_previous_position, certainty, tracked_item, frame, Torso_Position, Torso_Size, capture_width, capture_height, fencer_data, engarde_length, camera_steady, camera_motion_threshold):
  # Tracks the position of items
  # tracked_item is needed since boxes only has the class of the item tracked, not the Left or Right
  # tracked_item Format: [0,1,2,3] = [Background, Bell_Guard, Score_Box, Torso]

  #Assumed inherent uncertainty
  certainty_default = int(capture_width/16)
  certainty_multiplier = int(capture_width/80)

  #Reduces the max value of y as compared to x
  y_limiter = 24

  boxes_temp = []
  #Filters out potential boxes based on Tracked Item, Confidence and Saturation of the Box
  if (tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard'):

    #Uses only the Fencer_Data and Certainty for the Appropriate side.
    if tracked_item == 'Left_BellGuard':
      fencer_data = fencer_data[0]
      # display(f'The certainty here is {certainty}.')
      bell_certainty = certainty
    else:
      #Assumes the Right BellGuard
      fencer_data = fencer_data[1]
      bell_certainty = certainty

    for j in range(len(boxes)):
      #The minimum required certainty for a bellguard box
      if ((boxes[j][2] == 1) and (boxes[j][1] > (bellguard_confidence - bellguard_tracking_det_offset))):
        boxes_temp.append(boxes[j])

  elif (tracked_item == 'Left_Torso' or tracked_item == 'Right_Torso'):
    for j in range(len(boxes)):
      if ((boxes[j][2] == 3) and (boxes[j][1] > min_torso_confidence)):
        #Bypasses the Saturation Test
        # test_result = saturation_test(boxes[j], frame)
        test_result = True
        if test_result == True:
          boxes_temp.append(boxes[j])
        else:
          if verbose == True:
            display(f'The saturation test failed at frame {frame_count}.')
          else:
            pass
  elif (tracked_item == 'Scoring_Box'):
    for j in range(len(boxes)):
      if (boxes[j][2] == 2):
        boxes_temp.append(boxes[j])

  # Assigns boxes_temp to boxes
  boxes = boxes_temp

  # Creates points at the centers of the bounding boxes that are in this frame
  x_center = []
  y_center = []
  for i in range(len(boxes)):
    x_center.append(int((boxes[i][0][1] + boxes[i][0][3])/2))
    y_center.append(int((boxes[i][0][0] + boxes[i][0][2])/2))

  # Max allowed speed of a bellguard in a single frame
  # Accounts for a position jump following the engarde positioning
  if frame < engarde_length + 3:
    max_speed = int(capture_width/64)
  else:
    max_speed = int(capture_width/24)

  # Converts previous position into a speed
  x_pos = int(previous_position[0])
  if verbose == True:
    display(f'previous_position is {previous_position} and twice_previous_position is {twice_previous_position}.')
  x_speed = int(min(previous_position[0] - twice_previous_position[0], max_speed))
  y_pos = int(previous_position[1])
  y_speed = int(min(previous_position[1] - twice_previous_position[1], int(max_speed/y_limiter)))
  y_speed = int(max(y_speed, int(max_speed*(-1)/y_limiter)))

  if (frame - 1)  == engarde_length and verbose == True:
      display(f'THe x_speed is {x_speed} and the y_speed is {y_speed} at the engarde length, frame {frame - 1}.')

  # Flips the tracking box to be between the two fencers
  if tracked_item == 'Left_BellGuard' or tracked_item == 'Left_Torso':
    horiz_flip = False
    if verbose == True:
      display(f'The horizontal flip is {horiz_flip} for the {tracked_item} at frame {frame - 1}.')
  elif tracked_item == 'Right_BellGuard' or tracked_item == 'Right_Torso':
    horiz_flip = True
    if verbose == True:
      display(f'The horizontal flip is {horiz_flip} for the {tracked_item} at frame {frame - 1}.')
  else:
    horiz_flip = False


  # Defines the tracking box
  expected_position = [(x_pos + x_speed),(y_pos + y_speed)]
  padding = int(certainty*certainty_multiplier + certainty_default)
  boundary_box_for_tracking = [padding, padding, padding, padding]
  tracking_box = create_boundary_box(expected_position, boundary_box_for_tracking, horiz_flip)
  positions = []

  if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
    # Sets the values for the torso boundary box, limits Bellguard distance from Torso center
    boundary_box_for_torso = [int(Torso_Size[0]*0.20), int(Torso_Size[0]*3.25), int(Torso_Size[1]*.75), int(Torso_Size[1]*1.0)]
    # Uses the boundary box to create a box based on Left/Right and expected/previous position
    torso_box = create_boundary_box(Torso_Position, boundary_box_for_torso, horiz_flip)
    # Finds the overlap of multiple boxes to satisy multiple restrictions
    [x_min, x_max, y_min, y_max] = boundary_box_overlap(tracking_box, torso_box)
    if verbose == True:
      display(f'The Torso_Size[0] is {Torso_Size[0]}, the Horizontal Flip is {horiz_flip} and Torso_Position is {Torso_Position}.')
  else:
    [x_min, x_max, y_min, y_max] = tracking_box

  if verbose == True:
    display(f'The tracking box for the {tracked_item} at frame {frame - 1} is: {tracking_box}.')

  if (tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard') and verbose == True:
    display(f'The torso box for the {tracked_item} at frame {frame - 1} is: {torso_box}.')
    display(f'The overlapping tracking box for the {tracked_item} at frame {frame - 1} is: {[x_min, x_max, y_min, y_max]}.')

  # Creates a list of positions within the bounding boxes
  for i in range(len(boxes)):
    center = [x_center[i], y_center[i]]
    tracking_result = boundary_box_test(center,tracking_box)
    # If the center point is within both boxes for Bellguards or tracking box for other items, then it is appended to positions
    if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
      torso_result = boundary_box_test(center,torso_box)
      # Allows for an incorrect engarde position for the bellguard
      if (frame - 1) > engarde_length + 3:
        if tracking_result == True and torso_result == True:
          positions.append([x_center[i],y_center[i], boxes[i][1]])
      else:
        # Only the torso results is required for the engarde positioning
        if torso_result == True:
          positions.append([x_center[i],y_center[i], boxes[i][1]])
    else:
      if tracking_result == True:
        positions.append([x_center[i],y_center[i], boxes[i][1]])

  # Maximum distance only applies if there are multiple bounding boxes within the tracking box
  maximum_distance_from_expected = int(capture_width/24)
  # Expected Position [x,y], Limits expected position in front of the fencer
  if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
    display(f'The expected position is {expected_position} and Torso Position and size is {Torso_Position[0]} and {Torso_Size[0]}.')
    if (expected_position[0] > Torso_Position[0] + Torso_Size[0]*2.0) and tracked_item == 'Left_BellGuard':
      if verbose == True:
        display(f'At frame {frame - 1} the expected position of the {tracked_item} was too far in front of the Torso, adjusting expected.')
      expected_position = [int(Torso_Position[0] + Torso_Size[0]*2.0), y_pos]
    if (expected_position[0] < Torso_Position[0]) and tracked_item == 'Left_BellGuard':
      if verbose == True:
        display(f'At frame {frame - 1} the expected position of the {tracked_item} was behind the Torso, adjusting expected.')
      expected_position = [int(Torso_Position[0]), y_pos]
    if expected_position[0] < Torso_Position[0] - Torso_Size[0]*2.0 and tracked_item == 'Right_BellGuard':
      if verbose == True:
        display(f'At frame {frame - 1} the expected position of the {tracked_item} was too far from the Torso, adjusting expected.')
        display(f'Torso_Position[0] is {Torso_Position[0]}, Torso_Size[0] is {Torso_Size[0]}, y_pos is {y_pos}.')
      expected_position = [int(Torso_Position[0] - Torso_Size[0]*2.0), y_pos]
    if (expected_position[0] > Torso_Position[0]) and tracked_item == 'Right_BellGuard':
      if verbose == True:
        display(f'At frame {frame - 1} the expected position of the {tracked_item} was behind the Torso, adjusting expected.')
      expected_position = [int(Torso_Position[0]), y_pos]

  #Assumed maximum distance from wrist to bellguard
  wrist_to_bellguard_max = int(Torso_Size[0]/8)

  #Sets Initial Conditions for Type of Tracking
  using_human_pose = False
  using_difference_images = False
  using_expected = False
  using_position = False

  if verbose == True:
    display(f'The camera steady value for frame {frame - 1} is {camera_steady[frame - 1]}.')
    if camera_steady[frame - 1] >= camera_motion_threshold:
      display(f'The camera is in motion and motion detection is less reliable.')

  # Determines the Bellguard Position based on number of detections, confidence, box location and motion
  if (len(positions)) == 0:
    display(f'There where no positions found for the {tracked_item} at frame {frame - 1}.')
    if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
      #Uses a larger boundary box if high confidence in wrist position
      if fencer_data[0][2] > wrist_conf_very_high:
        display(f'The wrist confidence is very high, using a larger human pose boundary.')
        human_pose_boundary = [int(Torso_Size[0]*1.5), int(Torso_Size[0]*1.75), int(Torso_Size[0]), int(Torso_Size[0])]
      else:
        display(f'The wrist confidence is not very high, using a smaller human pose boundary.')
        human_pose_boundary = [int(Torso_Size[0]/2), int(Torso_Size[0]*1.25), int(Torso_Size[0]*3/4), int(Torso_Size[0]*3/4)]
      wrist_position = [fencer_data[0][0], fencer_data[0][1]]
      display(f'Attempting Human Pose Approximation for the {tracked_item} at frame {frame - 1}.')
      if tracked_item == 'Left_BellGuard':
        boundary_box = create_boundary_box(expected_position, human_pose_boundary, False)
        box_test = boundary_box_test(wrist_position, boundary_box)
      else:
        boundary_box = create_boundary_box(expected_position, human_pose_boundary, True)
        box_test = boundary_box_test(wrist_position, boundary_box)
      if verbose == True:
        display(f'{tracked_item} : wrist conf:{fencer_data[0][2]}, box_test:{box_test}.')
      if fencer_data[0][2] > wrist_conf_min and box_test:
        #Wrist Pose Approximation
        if verbose == True:
          display(f'Using the Wrist Approximation for the {tracked_item} at frame {frame - 1}.')
          display(f'The fencer data for frame {frame - 1} is:')
          display(fencer_data)
        using_human_pose = True
        if tracked_item == 'Left_BellGuard':
          position = [fencer_data[0][0] + int(Torso_Size[0]/8), fencer_data[0][1] - int(Torso_Size[0]/12)]
        elif tracked_item == 'Right_BellGuard':
          #Right_Bellguard is assumed
          position = [fencer_data[0][0] - int(Torso_Size[0]/8), fencer_data[0][1] - int(Torso_Size[0]/12)]
        else:
          display(f'The tracked item was not a Bell Guard at frame {frame - 1}.')
      #If the Human Pose is outside of bounds then motion difference is tried
      else:
        motion_difference_boundary = [int(Torso_Size[0]/4), int(Torso_Size[0]), int(Torso_Size[0]/3), int(Torso_Size[0]/3)]
        if tracked_item == 'Left_BellGuard' and camera_steady[frame - 1] < camera_motion_threshold:
          boundary_box = create_boundary_box(expected_position, motion_difference_boundary, False)
          position = motion_difference_tracking(frame, 'Left', boundary_box, capture_width, capture_height, 0.5, 3, 4)
          if position == 'None':
            display(f'Attempting to use a smaller kernel for motion difference tracking.')
            position = motion_difference_tracking(frame, 'Left', boundary_box, capture_width, capture_height, 1, 3, 4)
            if position == 'None':
              display(f'Attempting to use a smallest kernel for motion difference tracking.')
              position = motion_difference_tracking(frame, 'Left', boundary_box, capture_width, capture_height, 3, 3, 4)
          if verbose == True:
            display(f'The position for motion difference frame {frame - 1} is ({position})')
            display(f'The boundary box test limits are {motion_difference_boundary} for frame {frame - 1}.')
          boundary_box = create_boundary_box(expected_position, motion_difference_boundary, False)
          box_test = boundary_box_test(position, boundary_box)
          #Uses the Expected position if the motion difference is out of bounds
          if box_test == False:
            display(f'Motion difference failed, using the Expected Position for the {tracked_item} for frame {frame - 1}.')
            position = expected_position
            using_expected = True
          else:
            display(f'The motion difference position was used for the {tracked_item} at frame {frame - 1}.')
            using_difference_images = True
        elif tracked_item == 'Right_BellGuard' and camera_steady[frame - 1] < camera_motion_threshold:
          boundary_box = create_boundary_box(expected_position, motion_difference_boundary, True)
          position = motion_difference_tracking(frame, 'Right', boundary_box, capture_width, capture_height, 0.5, 3, 4)
          if position == 'None':
            display(f'Attempting to use a smaller kernel for motion difference tracking.')
            position = motion_difference_tracking(frame, 'Right', boundary_box, capture_width, capture_height, 1, 3, 4)
            if position == 'None':
              display(f'Attempting to use a smallest kernel for motion difference tracking.')
              position = motion_difference_tracking(frame, 'Right', boundary_box, capture_width, capture_height, 3, 3, 4)
          if verbose == True:
            display(f'The position for motion difference frame {frame - 1} is ({position})')
            display(f'The boundary box test limits are {motion_difference_boundary} for frame {frame - 1}.')
          boundary_box = create_boundary_box(expected_position, motion_difference_boundary, True)
          box_test = boundary_box_test(position, boundary_box)
          # box_test = False
          if box_test == False:
            display(f'Motion difference failed, using the Expected Position for the {tracked_item} for frame {frame - 1}.')
            position = expected_position
            using_expected = True
          else:
            display(f'The motion difference position was used for the {tracked_item} at frame {frame - 1}.')
            using_difference_images = True
        else:
          display(f'Too much camera motion, using expected position')
          position = expected_position
          using_expected = True
    else:
      position = expected_position    

    # Criteria for Setting Certainty to zero preventing a linear appoximation adjustment of this point
    if (using_human_pose == True and fencer_data[0][2] > wrist_conf_high) or \
    (using_difference_images == True and position[1] < Torso_Position[1] + Torso_Size[1]/2 and camera_steady[frame - 1] < camera_motion_threshold):
      if using_difference_images == True:
        display(f'Using difference images for frame {frame - 1} with no detected positions')
      certainty = 0
    else:
      certainty = certainty + 1

  # For a single detected Bellguard Position
  elif (len(positions)) == 1:
    if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
      display(f'There is one possible position, {positions[0]} for {tracked_item} in the tracking box for frame {frame - 1}.')
      # single_position_box = [int(Torso_Size[0]/2*(1+bell_certainty/4)), int(Torso_Size[0]*(1+bell_certainty/4)), int((Torso_Size[0]/2)*(1+bell_certainty/4)), int((Torso_Size[0]/2)*(1+bell_certainty/4))]
      single_position_box = [int(Torso_Size[0]*3/4*(1+bell_certainty/4)), int(Torso_Size[0]*(1+bell_certainty/4)), int(Torso_Size[0]), int(Torso_Size[0])]
      if tracked_item == 'Left_BellGuard':
        boundary_box = create_boundary_box(expected_position, single_position_box, False)
      else:
        boundary_box = create_boundary_box(expected_position, single_position_box, True)
      box_test = boundary_box_test(positions[0], boundary_box)
      display(f'The expected position for frame {frame - 1} is {expected_position}.')
      display(f'The single_position_box is {single_position_box} and the boundary box is {boundary_box}.')
      # Requires Box Boundary and Human Pose Wrist confidence less than high confidence.
      # if box_test == True and fencer_data[0][2] < wrist_conf_high:
      if box_test == True and positions[0][2] > bellguard_confidence_high:
        display(f'The detected position was used for the {tracked_item} at frame {frame - 1}.')
        position = positions[0]
        using_position = True
      else:
        #Human Pose
        display(f'Attempting to use Human Pose for the {tracked_item} at frame {frame - 1}')
        if fencer_data == 'None':
          fencer_data = [[0,0,0],[0,0,0],[0,0,0]]
        human_pose_boundary = [int(Torso_Size[0]*3/4), int(Torso_Size[0]), int(Torso_Size[0]/2), int(Torso_Size[0]/2)]
        display(f'Fencer data for frame {frame - 1} is: {fencer_data}.')
        wrist_position = [fencer_data[0][0], fencer_data[0][1]]
        if tracked_item == 'Left_BellGuard':
          boundary_box = create_boundary_box(expected_position, human_pose_boundary, False)
        else:
          boundary_box = create_boundary_box(expected_position, human_pose_boundary, True)
        box_test = boundary_box_test(wrist_position, boundary_box)
        if fencer_data[0][2] > wrist_conf_min and box_test:
          if verbose == True:
            display(f'{tracked_item}: wrist conf:{fencer_data[0][2]}, box_test:{box_test}.')
            display(f'Using the Wrist Approximation for the {tracked_item} at frame {frame - 1}.')
            display(f'The fencer data for frame {frame - 1} is:')
            using_human_pose = True
          if tracked_item == 'Left_BellGuard':
            position = [fencer_data[0][0] + int(Torso_Size[0]/8), fencer_data[0][1] - int(Torso_Size[0]/12)]
          else:
            #Right_Bellguard is assumed
            position = [fencer_data[0][0] - int(Torso_Size[0]/8), fencer_data[0][1] - int(Torso_Size[0]/12)]
        else:
          #Image Difference
          display(f'Attempting to use Image Difference for the {tracked_item} at frame {frame - 1}')
          motion_difference_boundary = [int(Torso_Size[0]/8), int(Torso_Size[0]/2), int(Torso_Size[0]/4), int(Torso_Size[0]/4)]
          if tracked_item == 'Left_BellGuard':
            boundary_box = create_boundary_box(expected_position, motion_difference_boundary, False)
            diff_position = motion_difference_tracking(frame, 'Left', [x_min, x_max, y_min, y_max], capture_width, capture_height, 1, 1, 2)
            if diff_position == 'None':
              diff_position = motion_difference_tracking(frame, 'Left', [x_min, x_max, y_min, y_max], capture_width, capture_height, 2, 1, 2)
          else:
            #Right Bellguard is assumed
            boundary_box = create_boundary_box(expected_position, motion_difference_boundary, True)
            diff_position = motion_difference_tracking(frame, 'Right', [x_min, x_max, y_min, y_max], capture_width, capture_height, 1, 1, 2)
            if diff_position == 'None':
              diff_position = motion_difference_tracking(frame, 'Right', [x_min, x_max, y_min, y_max], capture_width, capture_height, 2, 1, 2)
          box_test = boundary_box_test(diff_position, motion_difference_boundary)
          if box_test == True and diff_position != 'None':
            position = diff_position
            using_difference_images = True
          else:
            #Expected Position
            position = expected_position
            using_expected = True
          if verbose == True:
            display(f'The position for motion difference frame {frame - 1} is ({position})')
            display(f'The motion_difference_boundary test limits are {motion_difference_boundary} for frame {frame - 1}.')

      # Designed to catch an engarde position that is outside the tracking box
      if frame < (engarde_length + 3) and position == twice_previous_position:
        position = positions[0]

      #Sets Certainty Box
      if (using_human_pose == True and fencer_data[0][2] > wrist_conf_high) or (using_position == True):
        certainty = 0
        display(f'Certainty set to zero for frame {frame - 1} for the {tracked_item}.')
      else:
        certainty = certainty + 1

    else:
      position = positions[0]

  # Multiple bounding boxes within the tracking box
  elif (len(positions)) > 1:
    display(f'Multiple Bounding Boxes Detected for the {tracked_item} at frame {frame - 1}')
    # One set of conditions is used for Bell_Guards and another for all else
    if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
      # If the fencer_data wrist is confident, then it is used for Bell_Guards
      if fencer_data[0][2] > wrist_conf_min:
        display(f'Wrist Confidence Greater than Minimum for the {tracked_item} at frame {frame - 1}.')
        display(f'The Pose Confidence is {fencer_data[0][2]} with a required minimum of {wrist_conf_min}.')
        human_pose_boundary = [int(Torso_Size[0]/4), int(Torso_Size[0]/2), int(Torso_Size[0]/2), int(Torso_Size[0]/2)]
        wrist_position = [fencer_data[0][0], fencer_data[0][1]]
        if tracked_item == 'Left_BellGuard':
          boundary_box = create_boundary_box(expected_position, human_pose_boundary, False)
        else:
          boundary_box = create_boundary_box(expected_position, human_pose_boundary, True)
        box_test = boundary_box_test(wrist_position, boundary_box)
        if tracked_item == 'Left_BellGuard' and box_test == True:
          position = [fencer_data[0][0] + int(Torso_Size[0]/8), fencer_data[0][1] - int(Torso_Size[0]/6)]
          using_human_pose = True
          display(f'Using the wrist position of {position} for the {tracked_item} at frame {frame - 1}.')
        elif tracked_item == 'Right_BellGuard' and box_test == True:
          position = [fencer_data[0][0] - int(Torso_Size[0]/8), fencer_data[0][1] - int(Torso_Size[0]/6)]
          using_human_pose = True
          display(f'Using the wrist position of {position} for the {tracked_item} at frame {frame - 1}.')
        else:
          # Tests if the Position Confidence is High for the Bellguard
          if positions[0][2] > bellguard_confidence_high:
            position = multiple_box_determination(expected_position, positions, [human_pose_boundary[0], human_pose_boundary[1]], bellguard_confidence, horiz_flip)
            using_position = True
          else:
            position = expected_position
            using_expected = True
            display(f'The Human Pose Box Test failed for the {tracked_item} at frame {frame - 1}, using expected position.')
          display(f'The point tested is {wrist_position} and the box is {boundary_box} for human pose at for the {tracked_item} at frame {frame - 1}')
        # If the wrist confidence is not High, while the bellguard is, then uses the Bellguard Position
        if fencer_data[0][2] < wrist_conf_high and positions[0][2] > bellguard_confidence_high:
          single_position_box = [int(Torso_Size[0]/2), int(Torso_Size[0]), int(Torso_Size[0]/2), int(Torso_Size[0]/2)]
          if tracked_item == 'Left_BellGuard':
            boundary_box = create_boundary_box(expected_position, single_position_box, False)
          else:
            boundary_box = create_boundary_box(expected_position, single_position_box, True)
          box_test = boundary_box_test(positions[0], boundary_box)
          if box_test:
            display(f'Using the High Confidence Bellguard for Multiple Boxes for the {tracked_item} at frame {frame - 1}.')
            position = positions[0]
            using_position = True

      # If the fencer_data wrist is not confident
      else:
        display(f'Insufficient Pose Confidence for the {tracked_item} at frame {frame - 1}.')
        if verbose == True:
          display(f'The Pose Confidence is {fencer_data[0][2]} with a required minimum of {wrist_conf_min}.')
          display(f'The x value is  {fencer_data[0][0]} with a minimum of {x_min} and a maximum of {x_max}.')
          display(f'The x value is  {fencer_data[0][1]} with a minimum of {y_min} and a maximum of {y_max}.')
        # Excludes Positions too far from expected but still within the tracking box
        within_distance_from_expected = []
        for i in range(len(positions)):
          expected_box = [int(Torso_Size[0]/2*(1+bell_certainty/4)), int(Torso_Size[0]*(1+bell_certainty/4)), int(Torso_Size[0]/6*(1+bell_certainty/4)), int(Torso_Size[0]/6)]
          if tracked_item == 'Left_BellGuard':
            boundary_box = create_boundary_box(expected_position, expected_box, False)
          else:
            boundary_box = create_boundary_box(expected_position, expected_box, True)
          box_test = boundary_box_test(positions[i], boundary_box)
          if box_test:
            within_distance_from_expected.append(positions[i])

        # Uses the most confident, i.e. the first position in the list
        if len(within_distance_from_expected) > 0:
          position_boundary = [int(Torso_Size[0]/4), int(Torso_Size[0]/2), int(Torso_Size[0]/2), int(Torso_Size[0]/2)]
          position = multiple_box_determination(expected_position, positions, [position_boundary[0], position_boundary[1]], bellguard_confidence, horiz_flip)
          certainty = 0
          using_position = True
        else:
        # If the length of within_distance_from_expected is zero
          display(f'Error occured finding a position within the required distance and the {tracked_item} set to expeced position at frame {frame - 1}.')
          display(f'The expected position is {expected_position}, while the expected box is {expected_box}.')
          position = [(x_pos + x_speed),(y_pos + y_speed)]
          using_expected = True

      #Sets Certainty Box
      if (using_human_pose == True and fencer_data[0][2] > wrist_conf_high) or (using_position == True):
        if verbose == True:
          display(f'Confidence for the {tracked_item} is High so the certainty is set to zero.')
        certainty = 0
      else:
        if verbose == True:
          display(f'Confidence for the {tracked_item} is Low so the certainty is incremented higher.')
        certainty = certainty + 1

    # If the tracked item is not a bell_guard
    else:
      #Uses the most confident position within the tracking box
      position = positions[0]


  #Prevents the Bellguard being hidden behind the knee by setting a bellguard position behind the knee to the knee position.
  if (tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard') and fencer_data[1][2] > knee_conf_min:
    distance_from_position_to_knee = abs(int(((position[0] - fencer_data[1][0])**2 + (position[1] - fencer_data[1][1])**2)**(0.5)))
    
    display(f'The distance for the {tracked_item} from the knee is {distance_from_position_to_knee} and the min is {(Torso_Size[0]/2)} at frame {frame - 1}.')
    if tracked_item == 'Left_BellGuard':
      if distance_from_position_to_knee < (Torso_Size[0]/2) and (position[0] < fencer_data[1][0]) and (fencer_data[0][2] > wrist_conf_min):
        position = [fencer_data[1][0] + int(Torso_Size[0]/8), fencer_data[1][1] - int(Torso_Size[0]/12)]
        display(f'The {tracked_item} is near the knee at frame {frame - 1}.')
    else:
      #Assumes Right_BellGuard
      if distance_from_position_to_knee < (Torso_Size[0]/2) and (position[0] > fencer_data[1][0]) and (fencer_data[0][2] > wrist_conf_min):
        position = [fencer_data[1][0] - int(Torso_Size[0]/8), fencer_data[1][1] - int(Torso_Size[0]/12)]
        display(f'The {tracked_item} is near the knee at frame {frame - 1}.')

  if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
    display(f'The position of the {tracked_item} at frame {frame - 1} is {position}.')

  return (position, certainty, [x_min, x_max, y_min, y_max])

def weight_average_list(List):

  #Prevents division by zero
  try:
    value_sum = 0
    value_weight = 0
    for i in range(len(List)):
      value_sum = (value_sum + List[i][0]) * List[i][1]
      value_weight = value_weight + List[i][1]
    weighted_average = value_sum/value_weight
  except:
    weighted_average = 0

  return (weighted_average)

def average_list(List):
  try:
    average = sum(List) / len(List)
  except:
    average = 0
  return (average)

def color_tester(box, frame):
  #Takes a given box and tests for a specific color range

  path = r'/content/Mask_RCNN/videos/save/'
  file_name = str(frame) + '.jpg'
  name = os.path.join(path, file_name)
  img = cv2.imread(name)

  display(f'The file names to be color tested is {name}.')
  # box[0] are the coordinates ([y1,x1,y2,x2]), box[1] is confidence and box[2] is object
  # Tests if Bellguard is the correct color
  if box[2] == 1:
    blue_range = [50, 150]
    green_range = [50, 150]
    red_range = [50, 160]
    max_delta = 25
  elif box[2] == 3:
    blue_range = [60, 150]
    green_range = [60, 150]
    red_range = [60, 160]
    max_delta = 30
  else:
    display(f'The object to test does not have a color profile.')

  # OpenCV uses Blue, Green, Red order
  b, g, r = 0, 0, 0

  width = (box[0][3]-box[0][1])
  height = (box[0][2]-box[0][0])

  #i is the x value of the image
  for i in range(width):
    #j is y value of the image
    for j in range(height):
      #color channel of the image [B,G,R]
      #image, img, is of format [y,x] 
      b = b + img[box[0][0] + j, box[0][1] + i, 0]
      g = g + img[box[0][0] + j, box[0][1] + i, 1]
      r = r + img[box[0][0] + j, box[0][1] + i, 2]

  # Finds the Color Averages
  b_average = int(b/(width*height))
  g_average = int(g/(width*height))
  r_average = int(r/(width*height))

  # Finds maximum differences between colors
  max_1 = abs(b_average - g_average)
  max_2 = abs(b_average - r_average)
  max_3 = abs(g_average - r_average)
  max_delta = max(max_1, max_2, max_3)

  if test_result == False:
    display(f'The Color Test Result Failed for object {box[2]}.')

  return (test_result)

def symmetry_test(width, height, left_x, left_y, right_x, right_y):

  # Tests the potential left and right positions for left/right symmetry and removes outlier points
  display(f'Commencing Symmetry Test...')

  # Sets how large the allowable band is with respect to height or width
  band_width_ratio_x = 8
  band_width_ratio_y = 8

  all_positions_x = left_x + right_x
  all_positions_y = left_y +right_y
  if len(all_positions_x) != len(all_positions_y):
    display(f'ERROR...The length of the x and y positions are different.')


  # Keeps track of which positions are most in line with the other positions
  # Finds the X Band
  x_distances_from_center = []
  x_distances_from_other_points_score = []
  for i in range(len(all_positions_x)):
    #Determines the x_min band for each position by distance from center
    x_distances_from_center.append(abs(int((width/2)-all_positions_x[i])))
  #Creates an iterator that determines which x_point is close to the most other points and finds its index
  for j in range(len(x_distances_from_center)):
    score = 0
    for k in range(len(x_distances_from_center) - 1):
      if abs(x_distances_from_center[j] - x_distances_from_center[k+1]) < width/band_width_ratio_x:
        score = score + 1
      else:
        pass
    x_distances_from_other_points_score.append(score)
  x_index_band = x_distances_from_other_points_score.index(max(x_distances_from_other_points_score))

  x_min = abs(int(all_positions_x[x_index_band] - width/band_width_ratio_x))
  x_max = abs(int(all_positions_x[x_index_band] + width/band_width_ratio_x))

  # Finds the Y Band
  y_distances_from_center = []
  y_distances_from_other_points_score = []
  for i in range(len(all_positions_y)):
    y_distances_from_center.append(abs(int((height/2)-all_positions_y[i])))
  for j in range(len(y_distances_from_center)):
    score = 0
    for k in range(len(y_distances_from_center) - 1):
      if abs(y_distances_from_center[j] - y_distances_from_center[k+1]) < width/band_width_ratio_y:
        score = score + 1
      else:
        pass
    y_distances_from_other_points_score.append(score)
  y_index_band = y_distances_from_other_points_score.index(max(y_distances_from_other_points_score))

  y_min = abs(int(all_positions_y[y_index_band] - width/band_width_ratio_y))
  y_max = abs(int(all_positions_y[y_index_band] + width/band_width_ratio_y))

  # Cycles through the positions and keeps values that are in the horizontal x band
  positionsx_temp = []
  positionsy_temp = []

  display(f'The x_min/max is {x_min}/{x_max}, the band width is {width/band_width_ratio_x} and the center is {width/2}.')

  for i in range(len(all_positions_x)):
    if ((all_positions_x[i] < (width/2 - x_min)) and (all_positions_x[i] > (width/2 - x_max))) or ((all_positions_x[i] < (width/2 + x_max)) and (all_positions_x[i] > (width/2 + x_min))):
      positionsx_temp.append(all_positions_x[i])
      positionsy_temp.append(all_positions_y[i])
    else:
      pass

  # Replaces the all position x and y lists with the temp list limited by the bands
  all_positions_x = positionsx_temp
  all_positions_y = positionsy_temp

  #Cycles through the positions and keeps values that are in the vertical y band
  positionsx_temp = []
  positionsy_temp = []

  if verbose == True:
    display(f'The y_min/max is {y_min}/{y_max}, the band width is {height/band_width_ratio_y} and the center is {height/2}.')

  for i in range(len(all_positions_y)):
    if ((all_positions_y[i] > (y_min)) and (all_positions_y[i] < (y_max))):
      positionsx_temp.append(all_positions_x[i])
      positionsy_temp.append(all_positions_y[i])
    else:
      pass

  # Replaces the all position x and y lists with the temp list limited by the bands
  all_positions_x = positionsx_temp
  all_positions_y = positionsy_temp

  if verbose == True:
    display(f'There were originaly {len(left_x) + len(right_x)} values and {len(all_positions_x) - (len(left_x) + len(right_x))} were removed.')

  # Returns the x and y values to left and right positions
  ret_left_x, ret_left_y, ret_right_x, ret_right_y = [],[],[],[]

  
  for i in range(len(all_positions_x)):
    # Tests if the x value is on the left or right side
    if all_positions_x[i] < width/2:
      ret_left_x.append(all_positions_x[i])
      ret_left_y.append(all_positions_y[i])
    else:
      ret_right_x.append(all_positions_x[i])
      ret_right_y.append(all_positions_y[i])
  # Prevents an off center camera from removing all engarde points
  if (len(ret_left_x) == 0) or (len(ret_left_y) == 0) or (len(ret_right_x) == 0) or (len(ret_right_y) == 0):
    ret_left_x = left_x
    ret_left_y = left_y
    ret_right_x = right_x
    ret_right_y = right_y

  return (ret_left_x, ret_left_y, ret_right_x, ret_right_y)

def list_threshold_test(threshold, list_to_test):
  #Determines if a list meets a minimum threshold
  threshold_met = False

  for k in range(len(list_to_test)):
    if list_to_test[k][1] > threshold:
      threshold_met = True
      break
    else:
      pass

  return(threshold_met)

def multiple_box_determination(expected_position, positions, x_boundaries, min_conf, horiz_flip):

  confidence_weighting = .9

  delta_x_forward = x_boundaries[1]
  delta_x_backward = x_boundaries[0]

  if horiz_flip == True:
    delta_temp = delta_x_forward
    delta_x_forward = delta_x_backward
    delta_x_backward = delta_temp

  position_ratings = []

  display(f'There are {len(positions)} positions available.')
  display(f'The positions are:')
  display(positions)  

  for i in range(len(positions)):
    delta_position = positions[i][0] - expected_position[0]
    if verbose == True:
      display(f'The positions{i}[0] is {positions[i][0]} and the expected_position[0] is {expected_position[0]} therefore delta position is {delta_position}.')
    if delta_position > 0:
      if verbose == True:
        display(f'Position {i} is forward of the expected position.')
      position_ratings.append(abs((delta_position/delta_x_forward)*(1-positions[i][2])**confidence_weighting))
      display(f'delta_position is {delta_position}.')
      display(f'delta_x_forward is {delta_x_forward}.')
      display(f'positions[i][2] is {positions[i][2]}.')
    else:
      if verbose == True:
        display(f'Position {i} is behind the expected position.')
      position_ratings.append(abs((delta_position/delta_x_backward)*(1-positions[i][2])**confidence_weighting))
      display(f'delta_position is {delta_position}.')
      display(f'delta_x_backward is {delta_x_backward}.')
      display(f'positions[i][2] is {positions[i][2]}.')

  if verbose == True:
    display(position_ratings)

  position = positions[position_ratings.index(min(position_ratings))]

  return (position)

def boundary_box_overlap(box1, box2):
  #Finds the overlap of two boxes assume (x_min, x_max, y_min, y_max)
  
  box_overlap = [max(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), min(box1[3], box2[3])]

  return(box_overlap)

def create_boundary_box(center, padding, horiz_flip):
  # Center is [x,y]
  # Padding is [Left, Right, Top, Bottom]
  # horiz_flip is True or False

  if horiz_flip == False:
    left = center[0] - padding[0]
    right = center[0] + padding[1]
  elif horiz_flip == True:
    left = center[0] - padding[1]
    right = center[0] + padding[0]
  else:
    display(f'ERROR Horiz Flip not True or False.')

  top = center[1] - padding[2]
  bottom = center[1] + padding[3]

  return ([left, right, top, bottom])

def boundary_box_test(test_point, boundary):
  #Format Test_Point is of the form (x,y)
  #Format Boundary is of the form (x_min, x_max, y_min, y_max)
  #Format Boundary is of the form (behind the fencer, in front of the fencer, above the fencer, below the fencer)

  if verbose == True:
    display(test_point)
    display(boundary)

  if test_point != 'None':
    if test_point[0] > boundary[0] and test_point[0] < boundary[1] and test_point[1] > boundary[2] and test_point[1] < boundary[3]:
      box_test = True
    else:
      box_test = False
  else:
    box_test = False

  return (box_test)

def engarde_failure_test(bbox, bellguard_confidence, x_max, y_max, side):
  # Tests for reasons the engarde positioning failed to detect a BellGuard

  display(f'The {side} engarde position failed due to...')

  if side == 'Left':
    oppside = 'Right'
    k = 0
  else:
    oppside = 'Left'
    k = 1


  for j in range(len(bbox)):
    if bbox[j][1] < bellguard_confidence:
      display(f'The confidence in the {side} bellguard is too low at {bellguard_confidence}.')
    else: 
      pass
    if side == 'Left':
      if bbox[j][k] > x_max:
        display(f'The {side} bellguard was too far {oppside} at {bbox[j][0]} while the maximum is {x_max}.')
      else:
        pass
    else:
      display(f'bbox at this point is: {bbox}. J is {j} and k is {k}.')
      display(bbox[j])
      display(bbox[j][k])
      if bbox[j][k] < x_max:
        display(f'The {side} bellguard was too far {oppside} at {bbox[j][0]} while the maximum is {x_max}.')
    if bbox[j][k] > y_max:
      display(f'The {side} bellguard was too low at {bbox[j][0]} while the maximum allowed is {y_max}.')
    else:
      pass

  return

def torso_failure_test(bbox, capture_width, capture_height, y_average, Bell_Guard_Size_average, side, frame_count, min_torso_confidence):
  # Tests for reasons the engarde positioning failed to detect a Torso
  # Is tested at finding tracking boxes

  display(f'The {side} Torso failed due to...') 
  for j in range(len(bbox)):
    if bbox[j][1] > min_torso_confidence:
      pass
    else:
      display(f'The confidence is of the box is too low at only {int(bbox[j][1]*100)}% at frame {frame_count}.')
    if bbox[j][0][2] > y_average:
      pass
    else:
      display(f'The Torso was not lower than the Bell Guard with a lower height of {bbox[j][0][2]} with a max value of {y_average} at frame {frame_count}.')
    if bbox[j][0][2] < (y_average + 3*Bell_Guard_Size_average[1]):
      pass
    else:
      display(f'The bottom of the torso box was too low at {bbox[j][0][2]} with a max value of {int(y_average + 3*Bell_Guard_Size_average[1])} at frame {frame_count}.')

  display(f'y_average is {y_average}.')
  display(f'Bell_Guard_Size_average[1] is {Bell_Guard_Size_average[1]}.')

  return

def torso_position_failure_test(bbox, engarde_length, x_min_torso, x_max_torso, y_min_torso, y_max_torso, y_average, side, frame_count):
  # Tests for reasons the engarde positioning failed to detect a Torso
  # Is tested at torso positions

  confidence = min_torso_confidence

  display(f'Analyzing the Torso Position Failure at frame {frame_count} for the {side} side...')
  count = 0
  
  for k in range(len(bbox)):

    if bbox[k][2] == 3 and bbox[k][1] > confidence:
      count = count + 1
  display(f'There are {len(bbox)} ROIs, {count} of them are Torsos with greater than {confidence}%.')

  for j in range(len(bbox)):
    y_center = int((bbox[j][0][0] + bbox[j][0][2])/2)
    x_center = int((bbox[j][0][1] + bbox[j][0][3])/2)
    if bbox[j][2] == 3 and bbox[j][1] > confidence:
      if x_center > x_min_torso:
        pass
      else:
        display(f'The Torso center at {x_center} is to the Left of the Box side at {x_min_torso} at frame {frame_count}.')
      if x_center < x_max_torso:
        pass
      else:
        display(f'The Torso center at {x_center} is to the Right of the Box side at {x_max_torso} at frame {frame_count}.')
      if y_center > y_min_torso:
        pass
      else:
        display(f'The Torso center at {y_center} is Above the Box at {y_min_torso} at frame {frame_count}.')
      if y_center < y_max_torso:
        pass
      else:
        display(f'The Torso center at {y_center} is Below the Box at {y_max_torso} at frame {frame_count}.')
      if bbox[j][0][2] > y_average:
        pass
      else:
        display(f'The Torso center is Below the Bell Guard at frame {frame_count}.')
      if bbox[j][2] == 3:
        pass
      else:
        display(f'The Torso is not labelled as a Torso at frame {frame_count}.')
    else:
      pass

  return

def engarde_position(bbox, capture_width, capture_height, engarde_length, frame_count):
  #Finds the initial positions to start tracking
  #Format of bbox[frame][roi], ([y1,x1,y2,x2], percent certainty, type)

  # Initializes the Bell Guard Positions
  # Position format [x,y]
  # Size format [[Width],[Height]]
  Left_Position = []
  Right_Position = []
  Bell_Guard_Size = [[],[]]
  Scoring_Box_Position = []
  Scoring_Box_Size = [[],[]]
  Left_Torso_Position = []
  Left_Torso_Size = [[],[]]
  Right_Torso_Position = []
  Right_Torso_Size = [[],[]]

  display(f'The bbox for the engarde capture at frame {frame_count} is:')
  display(bbox)

  #sum_of_boxes is used to average the Left (x,y)(0), Right (x,y)(1), ScoreBox (x,y)(2), Left_Torso (x,y)(3), Right_Torso(x,y)(4) values
  sum_of_boxes = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]

  # j represents the rois(specific bounding box) within the frame sorted by confidence score
  for j in range(len(bbox)):
    # The percent confidence for each roi is [i][j][1]
    # This uses the minimum value of the bbox (top-left) to determine Left, Right, Scorebox
    # The Bellguards must be centered within the frame, classified as Bellguards with a minimum confidence and have the correct color saturation
    # Adds values to the Left engarde box
    if (bbox[j][1] > bellguard_confidence and bbox[j][0][1] < int(capture_width*2/5) and bbox[j][0][0] < int(capture_height*3/4) and bbox[j][0][0] > int(capture_height*1/4) and bbox[j][2] == 1):
      test_result = saturation_test(bbox[j], frame_count)
      if verbose == True:
        display(f'The result of the saturation test for the Left Engarde Position is {test_result} at frame {frame_count}.')
      if test_result == True:
        #Appends x value:
        # sum_of_boxes[0][0].append(bbox[j][0][1])
        sum_of_boxes[0][0].append([bbox[j][0][1], bbox[j][1]])
        #Appends y value:
        sum_of_boxes[0][1].append([bbox[j][0][0], bbox[j][1]])
        #Appends x width value:
        Bell_Guard_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
        #Appends y width value:
        Bell_Guard_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
    #Adds values to the Right engarde box
    elif (bbox[j][1] > bellguard_confidence and bbox[j][0][1] > int(capture_width*3/5) and bbox[j][0][0] < int(capture_height*3/4) and bbox[j][0][0] > int(capture_height*1/4) and bbox[j][2] == 1):
      # test_result = color_tester(bbox[i][j], i)
      test_result = saturation_test(bbox[j], frame_count)
      if verbose == True:
        display(f'The result of the saturation test for the Right Engarde Position is {test_result} at frame {frame_count}.')
      if test_result == True:
        #Appends x value:
        sum_of_boxes[1][0].append([bbox[j][0][1], bbox[j][1]])
        #Appends y value:
        sum_of_boxes[1][1].append([bbox[j][0][0], bbox[j][1]])
        #Appends x width value:
        Bell_Guard_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
        #Appends y width value:
        Bell_Guard_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
    #Adds values to the ScoreBox Position
    elif (bbox[j][1] > 0.50 and bbox[j][0][1] > int(capture_width/3) and bbox[j][0][1] < int(capture_width*(2/3)) and bbox[j][2] == 2):
      #Appends x value:
      sum_of_boxes[2][0].append([bbox[j][0][1], bbox[j][1]])
      #Appends y value:
      sum_of_boxes[2][1].append([bbox[j][0][0], bbox[j][1]])  
      #Appends x width value:
      Scoring_Box_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
      #Appends y width value:
      Scoring_Box_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
    else:
      pass

  try:
    # Tests for cause of Left Engarde Position Failure
    if len(sum_of_boxes[0][0]) == 0:
      engarde_failure_test(bbox[j], bellguard_confidence, int(capture_width*2/5), int(capture_height*2/3), 'Left')
    # Tests for cause of Right Engarde Position Failure
    if len(sum_of_boxes[1][0]) == 0:
      engarde_failure_test(bbox[j], bellguard_confidence, int(capture_width*3/5), int(capture_height*3/4), 'Right')
  except:
    display(f'There was an error in the engarde failure test and it was skipped.')

  # Finds the center point
  x_average_left = weight_average_list(sum_of_boxes[0][0])
  y_average_left = weight_average_list(sum_of_boxes[0][1])
  x_average_right = weight_average_list(sum_of_boxes[1][0])
  y_average_right = weight_average_list(sum_of_boxes[1][1])
  x_average_scorebox = weight_average_list(sum_of_boxes[2][0])
  y_average_scorebox = weight_average_list(sum_of_boxes[2][1])

  # Prevents a failure to detect the bellguard from failing to detect the torso
  # If the bellguard is unusually high or low then it is set to the height of the opposing BellGuard
  if (y_average_left < capture_height/5) or (y_average_left > capture_height*4/5):
    if verbose == True:
      display(f'The y_average_left was too high or low and was set to y_average_right.')
    y_average_left = y_average_right
  if (y_average_right < capture_height/5) or (y_average_right > capture_height*4/5):
    if verbose == True:
      display(f'The y_average_right was too high or low and was set to y_average_left.')
    y_average_right = y_average_left

  if verbose == True:
    display(f'The average left position is ({x_average_left},{y_average_left}).')
    display(f'The average right position is ({x_average_right},{y_average_right}).')

  # Bell_Guard_Size_average [Width, Height]
  Bell_Guard_Size_average = []
  # Appends the average scoring box width
  Bell_Guard_Size_average.append(average_list(Bell_Guard_Size[0]))
  # Appends the average scoring box height
  Bell_Guard_Size_average.append(average_list(Bell_Guard_Size[1]))

  # Finds the Torso Position After the Bell_Guard Position because the Bell_Guard is used as a constraint
  # j represents the rois(specific bounding box) within the frame sorted by confidence score
  for j in range(len(bbox)):
    # Adds values to the Left_Torso Position, similar requirements to Left guard
    # Minimum Torso confidence, on the left half of the screen, bottom of the box is below the bellguard, but also above 3 three times the bellguard height and is labeled torso
    if (bbox[j][1] > min_torso_confidence and bbox[j][0][1] < int(capture_width/2) and bbox[j][0][2] > y_average_left \
        and bbox[j][0][2] < (y_average_left + 3*Bell_Guard_Size_average[1]) and bbox[j][2] == 3):
      
      # Tests the Torso Color Saturation
      test_result = saturation_test(bbox[j], frame_count)
      if test_result == True:
        # Appends x value:
        sum_of_boxes[3][0].append(bbox[j][0][1])
        #Appends y value:
        sum_of_boxes[3][1].append(bbox[j][0][0])
        #Appends x width value:
        Left_Torso_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
        #Appends y width value:
        Left_Torso_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
      else:
        if verbose == True:
          display(f'The saturation test failed at frame {frame_count}.')
        else:
          pass
    # Adds values to the Right_Torso Position, similar requirements to Right guard
    elif (bbox[j][1] > min_torso_confidence and bbox[j][0][1] > int(capture_width/2) and \
          bbox[j][0][2] > (y_average_right) and bbox[j][0][2] < (y_average_right + 3*Bell_Guard_Size_average[1]) and bbox[j][2] == 3):
      test_result = saturation_test(bbox[j], frame_count)
      if test_result == True:
        #Appends x value:
        sum_of_boxes[4][0].append(bbox[j][0][1])
        #Appends y value:
        sum_of_boxes[4][1].append(bbox[j][0][0])
        #Appends x width value:
        Right_Torso_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
        #Appends y width value:
        Right_Torso_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
      else:
        if verbose == True:
          display(f'The saturation test failed at frame {frame_count}.')
        else:
          pass
    else:
      pass

  if len(sum_of_boxes[3][0]) == 0:
    torso_failure_test(bbox, capture_width, capture_height, y_average_left, Bell_Guard_Size_average, 'Left', frame_count, min_torso_confidence)
  if verbose == True:
    display(f'Prior to torso failure test for right torso the y_average_left is {y_average_left}.')

  if len(sum_of_boxes[4][0]) == 0:
    torso_failure_test(bbox, capture_width, capture_height, y_average_right, Bell_Guard_Size_average, 'Right', frame_count, min_torso_confidence) 
  if verbose == True:
    display(f'Prior to torso failure test for left torso the y_average_right is {y_average_right}.')

  #Finds the top left corner then moves the average point to the center
  x_average_left_torso = average_list(sum_of_boxes[3][0]) + average_list(Left_Torso_Size[0])/2
  y_average_left_torso = average_list(sum_of_boxes[3][1]) + average_list(Left_Torso_Size[1])/2
  x_average_right_torso = average_list(sum_of_boxes[4][0]) + average_list(Right_Torso_Size[0])/2
  y_average_right_torso = average_list(sum_of_boxes[4][1]) + average_list(Right_Torso_Size[1])/2

  if verbose == True:
    display(f'The average left engarde position is:({x_average_left},{y_average_left})')
    display(f'The average right engarde position is:({x_average_right},{y_average_right})')

    display(f'The average left torso is:({int(x_average_left_torso)},{int(y_average_left_torso)})')
    display(f'The average right torso is:({int(x_average_right_torso)},{int(y_average_right_torso)})')

  # scoring_box_size_average [Width, Height]
  scoring_box_size_average = []
  # Appends the average scoring box width
  scoring_box_size_average.append(average_list(Scoring_Box_Size[0]))
  # Appends the average scoring box height
  scoring_box_size_average.append(average_list(Scoring_Box_Size[1]))

  # left_torso_size_average [Width, Height]
  left_torso_size_average = []
  # Appends the average scoring box width
  left_torso_size_average.append(average_list(Left_Torso_Size[0]))
  # Appends the average scoring box height
  left_torso_size_average.append(average_list(Left_Torso_Size[1]))

  # right_torso_size_average [Width, Height]
  right_torso_size_average = []
  # Appends the average scoring box width
  right_torso_size_average.append(average_list(Right_Torso_Size[0]))
  # Appends the average scoring box height
  right_torso_size_average.append(average_list(Right_Torso_Size[1]))

  #Creates Padding for the EnGarde Tracking Box
  engarde_box_padding = int(capture_width/15)
  torso_padding = int(capture_width/20)

  x_min_engardeL = int(x_average_left - engarde_box_padding)
  x_max_engardeL = int(x_average_left + engarde_box_padding)
  y_min_engardeL = int(y_average_left - engarde_box_padding)
  y_max_engardeL = int(y_average_left + engarde_box_padding)

  x_min_engardeR = int(x_average_right - engarde_box_padding)
  x_max_engardeR = int(x_average_right + engarde_box_padding)
  y_min_engardeR = int(y_average_right - engarde_box_padding)
  y_max_engardeR = int(y_average_right + engarde_box_padding)

  x_min_engardeScore = int(x_average_scorebox - engarde_box_padding)
  x_max_engardeScore = int(x_average_scorebox + engarde_box_padding)
  y_min_engardeScore = int(y_average_scorebox - engarde_box_padding)
  y_max_engardeScore = int(y_average_scorebox + engarde_box_padding)

  x_min_torsoL = int(x_average_left_torso - torso_padding)
  x_max_torsoL = int(x_average_left_torso + torso_padding)
  y_min_torsoL = int(y_average_left_torso - torso_padding*3/2)
  y_max_torsoL = int(y_average_left_torso + torso_padding*3/2)

  x_min_torsoR = int(x_average_right_torso - torso_padding)
  x_max_torsoR = int(x_average_right_torso + torso_padding)
  y_min_torsoR = int(y_average_right_torso - torso_padding*3/2)
  y_max_torsoR = int(y_average_right_torso + torso_padding*3/2)

  #Iterates through the first engarde_length frames and checks if there are rois in the expected engarde position
  for j in range(len(bbox)):
    y_center = int((bbox[j][0][0] + bbox[j][0][2])/2)
    x_center = int((bbox[j][0][1] + bbox[j][0][3])/2)
    # Checks for rois in the Left Engarde Position
    if (x_center > x_min_engardeL and x_center < x_max_engardeL and y_center > y_min_engardeL and y_center < y_max_engardeL and bbox[j][2] == 1):
      # display(f'The roi is in the left en garde position')
      Left_Position.append([x_center, y_center])
    # Checks for rois in the Right Engarde Position
    if (x_center > x_min_engardeR and x_center < x_max_engardeR and y_center > y_min_engardeR and y_center < y_max_engardeR and bbox[j][2] == 1):
      # display(f'The roi is in the right en garde position')
      Right_Position.append([x_center, y_center])
    # Checks for rois in the Scoring Box Position
    if (x_center > x_min_engardeScore and x_center < x_max_engardeScore and y_center > y_min_engardeScore and y_center < y_max_engardeScore and bbox[j][2] == 2):
      Scoring_Box_Position.append([x_center, y_center])
    # Checks for rois in the Left Torso Position
    if (x_center > x_min_torsoL and x_center < x_max_torsoL and y_center > y_min_torsoL and y_center < y_max_torsoL and bbox[j][0][2] > y_average_left and bbox[j][2] == 3):
      Left_Torso_Position.append([x_center, y_center])
    # Checks for rois in the Right Torso Position 
    if (x_center > x_min_torsoR and x_center < x_max_torsoR and y_center > y_min_torsoR and y_center < y_max_torsoR and bbox[j][0][2] > y_average_right and bbox[j][2] == 3):
      Right_Torso_Position.append([x_center, y_center])

    Tracking_Bounding_Boxes_Temp = [[],[],[]]

    Tracking_Bounding_Boxes_Temp[0].append(x_min_engardeL)
    Tracking_Bounding_Boxes_Temp[0].append(x_max_engardeL)
    Tracking_Bounding_Boxes_Temp[0].append(y_min_engardeL)
    Tracking_Bounding_Boxes_Temp[0].append(y_max_engardeL)

    Tracking_Bounding_Boxes_Temp[1].append(x_min_engardeR)
    Tracking_Bounding_Boxes_Temp[1].append(x_max_engardeR)
    Tracking_Bounding_Boxes_Temp[1].append(y_min_engardeR)
    Tracking_Bounding_Boxes_Temp[1].append(y_max_engardeR)

    Tracking_Bounding_Boxes_Temp[2].append(x_min_engardeScore)
    Tracking_Bounding_Boxes_Temp[2].append(x_max_engardeScore)
    Tracking_Bounding_Boxes_Temp[2].append(y_min_engardeScore)
    Tracking_Bounding_Boxes_Temp[2].append(y_max_engardeScore)

    Tracking_Bounding_Boxes = Tracking_Bounding_Boxes_Temp

  # Tests for why a Torso Position is not Found
  if (len(Left_Torso_Position) == 0):
    torso_position_failure_test(bbox, engarde_length, x_min_torsoL, x_max_torsoL, y_min_torsoL, y_max_torsoL, y_average_left, 'Left', frame_count)    
  if (len(Right_Torso_Position) == 0):
    torso_position_failure_test(bbox, engarde_length, x_min_torsoR, x_max_torsoR, y_min_torsoR, y_max_torsoR, y_average_right, 'Right', frame_count)


  # Averages the Left and Right x,y positions for engarde
  # Left Bell Guard engarde position
  if verbose == True:
    display(f'The length of the built Tracking Bounding Boxes is {len(Tracking_Bounding_Boxes[0])}.')
  x = 0
  y = 0
  if len(Left_Position) > 0:
    for i in range(len(Left_Position)):
      x = x + Left_Position[i][0]
      y = y + Left_Position[i][1]
    x = int(x/(len(Left_Position)))
    y = int(y/(len(Left_Position)))
    Left_Position = [x,y]

  if verbose == True:
    display(f'Left_Position at Engarde is:')
    display(Left_Position)

  # Right Bell Guard engarde position
  x = 0
  y = 0
  if len(Right_Position) > 0:
    for i in range(len(Right_Position)):
      x = x + Right_Position[i][0]
      y = y + Right_Position[i][1]
    x = int(x/(len(Right_Position)))
    y = int(y/(len(Right_Position)))
    Right_Position = [x,y]

  if verbose == True:
    display(f'Right_Position at Engarde is:')
    display(Right_Position)

  # Scoring_Box engarde position
  x = 0
  y = 0
  if len(Scoring_Box_Position) > 0:
    for i in range(len(Scoring_Box_Position)):
      x = x + Scoring_Box_Position[i][0]
      y = y + Scoring_Box_Position[i][1]
    x = int(x/(len(Scoring_Box_Position)))
    y = int(y/(len(Scoring_Box_Position)))
    Scoring_Box_Position = [x,y]

  if Scoring_Box_Position == [0,0]:
    Tracking_Bounding_Boxes_Temp[2] = [0,0,0,0]

  if verbose == True:
    display(f'Scoring_Box_Position at Engarde is:')
    display(Scoring_Box_Position)

  # Left_Torso engarde position
  x = 0
  y = 0
  if len(Left_Torso_Position) > 0:
    for i in range(len(Left_Torso_Position)):
      x = x + Left_Torso_Position[i][0]
      y = y + Left_Torso_Position[i][1]
    x = int(x/(len(Left_Torso_Position)))
    y = int(y/(len(Left_Torso_Position)))
    Left_Torso_Position = [x,y]

  if verbose == True:
    display(f'Left_Torso_Position at Engarde is:')
    display(Left_Torso_Position)

  # Right_Torso engarde position
  x = 0
  y = 0
  if len(Right_Torso_Position) > 0:
    for i in range(len(Right_Torso_Position)):
      x = x + Right_Torso_Position[i][0]
      y = y + Right_Torso_Position[i][1]
    x = int(x/(len(Right_Torso_Position)))
    y = int(y/(len(Right_Torso_Position)))
    Right_Torso_Position = [x,y]

  if verbose == True:
    display(f'Right_Torso_Position at Engarde is:')
    display(Right_Torso_Position)

  return (Left_Position, Right_Position, Scoring_Box_Position, scoring_box_size_average, Tracking_Bounding_Boxes, Left_Torso_Position, Right_Torso_Position, left_torso_size_average, right_torso_size_average)

def draw_Bell_Guard_Position(Left_Position, Right_Position, Scoring_Box_Position, scoring_box_size_average, Left_Torso_Position, Right_Torso_Position, frame_count, Tracking_Bounding_Boxes, video_filename, capture_width, capture_height, engarde_length, keypoints, score_box_empty, camera_steady, camera_motion_threshold):
  #Adds an overlay on the image to visualize the location of tracked objects

  path = r'/content/Mask_RCNN/videos/'
  capture = cv2.VideoCapture(os.path.join(path, video_filename))

  capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)

  #Color format is [B,G,R]
  left_light_color_default = [[],[],[]]
  right_light_color_default = [[],[],[]]
  left_light_color = []
  right_light_color = []

  # Creates a list of Files from a Directory
  path = r'/content/Mask_RCNN/videos/save/'
  path_orig = r'/content/Mask_RCNN/videos/original/'
  files = [i for i in os.listdir(path)]
  # Sorts the Files after cropping '.jpg'
  files.sort(key=lambda x: int(x[:-4]))

  left_light_comparison, right_light_comparison, default_color = [], [], []

  for i, file in enumerate(files):
    # Reads the image
    name = os.path.join(path_orig, file)
    img = cv2.imread(name)

    # OpenCV uses Blue, Green, Red order
    # Light_Color is of the format [[[B0],[G0],[R0]],[[B1],[G1],[R1]],[[B2],[G2],[R2]],...]
    
    if i <= engarde_length:
      if scoring_box_size_average == [0,0]:
        scoring_box_size_average = [int(capture_width/5), int(capture_height/5)]
      if verbose == True:
        display(f'The average scoring box size is {scoring_box_size_average}.')
      # Uses a comparison of frames and scoring box position to determine the light off colors
      [left_light_comparison_temp, right_light_comparison_temp, defualt_color_temp] = scoring_box_lights(img, Scoring_Box_Position[i], scoring_box_size_average, [], i, score_box_empty)
      left_light_comparison.append(left_light_comparison_temp)
      right_light_comparison.append(right_light_comparison_temp)
      default_color.append(defualt_color_temp)
      # Averages the Default Color on the Last iteration
      if i == engarde_length:
        b_temp = int(sum(default_color[0])/len(default_color[0]))
        g_temp = int(sum(default_color[1])/len(default_color[1]))
        r_temp = int(sum(default_color[2])/len(default_color[2]))
        default_color = [b_temp,g_temp,r_temp]
    elif i > engarde_length:
      try:
        [left_light_comparison_temp, right_light_comparison_temp, defualt_color_temp] = scoring_box_lights(img, Scoring_Box_Position[i], scoring_box_size_average, default_color, i, score_box_empty)
      except:
        display(f'Light Comparison Failed due to Error at frame {i}.')
        [left_light_comparison_temp, right_light_comparison_temp, defualt_color_temp] = [0,0,[]]
      left_light_comparison.append(left_light_comparison_temp)
      right_light_comparison.append(right_light_comparison_temp)

    if verbose == True:
      display(f'Frame Count is {frame_count}.')

    #Creates the dots on the Bell Guards
    frame = cv2.circle(img, (Left_Position[i][0], Left_Position[i][1]), 4, (255, 0, 0), -1)
    frame = cv2.circle(frame, (Right_Position[i][0], Right_Position[i][1]), 4, (0, 255, 0), -1)
    frame = cv2.circle(frame, (Scoring_Box_Position[i][0], Scoring_Box_Position[i][1]), 4, (255, 255, 0), -1)
    frame = cv2.circle(frame, (Left_Torso_Position[i][0], Left_Torso_Position[i][1]), 4, (0, 255, 0), -1)
    frame = cv2.circle(frame, (Right_Torso_Position[i][0], Right_Torso_Position[i][1]), 4, (255, 255, 0), -1)

    # Adds Frame Number to the Image
    text = 'Frame' + str(i)
    frame = cv2.putText(frame, text, (int(capture_width*7/8), int(capture_height*1/16)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, )
    # Adds if the Camera Motion is detected by difference images
    if camera_steady[i] > camera_motion_threshold:
      text = 'Camera'
      frame = cv2.putText(frame, text, (int(capture_width*7/8), int(capture_height*3/16)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, )
      text = 'Motion'
      frame = cv2.putText(frame, text, (int(capture_width*7/8), int(capture_height*4/16)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, )
    
    # Sets BellGuard Position Colors
    left_color = (255, 0, 0)
    right_color = (0, 255, 0)

    #Creates the Tracking Boxes
    frame = cv2.putText(frame, 'Tracking Box', (Tracking_Bounding_Boxes[i][0][0], Tracking_Bounding_Boxes[i][0][2]), cv2.FONT_HERSHEY_COMPLEX, 0.7, left_color, 2)
    frame = cv2.rectangle(frame, (Tracking_Bounding_Boxes[i][0][0], Tracking_Bounding_Boxes[i][0][2]),(Tracking_Bounding_Boxes[i][0][1], Tracking_Bounding_Boxes[i][0][3]),left_color, 2)
    frame = cv2.putText(frame, 'Tracking Box', (Tracking_Bounding_Boxes[i][1][0], Tracking_Bounding_Boxes[i][1][2]), cv2.FONT_HERSHEY_COMPLEX, 0.7, right_color, 2)
    frame = cv2.rectangle(frame, (Tracking_Bounding_Boxes[i][1][0], Tracking_Bounding_Boxes[i][1][2]),(Tracking_Bounding_Boxes[i][1][1], Tracking_Bounding_Boxes[i][1][3]),right_color, 2)

    [frame, none] = overlay_keypoints(frame, keypoints[i][0], keypoints[i][1], True)

    #Saves the image frame overwriting the original image
    cv2.imwrite(name, frame)

  #Releases capture so that other files can be used
  capture.release()

  return (left_light_comparison, right_light_comparison)

def mask_image(frame, width, height, masking_box):
  # Used to Mask parts of the image that are not of interest

  if verbose == True:
    display(f'The masking box is:')
    display(masking_box)

  #Create the Mask
  mask = np.zeros((height, width, 3), dtype = np.uint8);
  for i in range(len(masking_box)):
    mask = cv2.rectangle(mask, (masking_box[i][0], masking_box[i][2]) ,(masking_box[i][1], masking_box[i][3]), (255,255,255), -1)

  #Applies the mask to Frame
  frame = cv2.bitwise_and(mask, frame)

  return (frame)

def create_representative_image(clip_vector, capture_width, capture_height):
  # Allows for an overlay that represents the bellguard horizontal motion and box lights

  #Creates a Folder to save the images and removes previous version
  os.chdir('/content/Mask_RCNN/videos')
  # !mkdir save_white_dot
  # os.mkdir('save_white_dot')
  # !rm -r /content/Mask_RCNN/videos/save_white_dot
  # os.rmdir('/content/Mask_RCNN/videos/save_white_dot')
  # Removes and Recreates the Save_White_Dot to ensure the directory is empty
  try:
    shutil.rmtree('save_white_dot') 
  except:
    display(f'ERROR removing the Save_White_Dot folder.')
  # os.rmdir('/save_white_dot')
  # !mkdir save_white_dot
  os.mkdir('save_white_dot')

  rect_size = int(capture_width/40)

  #Defines the File Path
  path = r'/content/Mask_RCNN/videos/save_white_dot/'
  
  for i in range(len(clip_vector)):
    img = np.zeros((capture_height,capture_width,3), np.uint8)

    #Creates the Left Bell_Guard
    img = cv2.circle(img, (clip_vector[i][0], int(capture_height/2)), 20, (118, 37, 217), -1)
    #Creates the Right Bell_Guard
    img = cv2.circle(img, (clip_vector[i][1], int(capture_height/2)), 20, (157, 212, 19), -1)

    if (clip_vector[i][2] == 1):
      #Creates the Left Score Light
      img = cv2.rectangle(img, (rect_size, rect_size), (rect_size*5, rect_size*3), (0, 0, 255), -1)
    if (clip_vector[i][3] == 1):
      #Creates the Right Score Light
      img = cv2.rectangle(img, (capture_width - rect_size, rect_size), (capture_width - rect_size*5, rect_size*3), (0, 255, 0), -1)

    name = str(i) + '.jpg'
    name = os.path.join(path, name)

    cv2.imwrite(name, img)

  return

def create_overlay_image(frame_count):
  # Allows for an overlay that represents the bellguard horizontal motion and box lights

  #Creates a Folder to save the images and removes previous version
  os.chdir('/content/Mask_RCNN/videos/')
  # !rm -r /content/Mask_RCNN/videos/overlay
  # Attempts to remove the Overlay folder and recreate it to ensure that it is empty
  try:
    shutil.rmtree('overlay')
  except:
    display(f'ERROR removing the Overlay folder.')
  # !mkdir overlay
  os.mkdir('overlay')


  #Defines the File Path
  path = r'/content/Mask_RCNN/videos/overlay/'
  path_background = r'/content/Mask_RCNN/videos/save/'
  path_overlay = r'/content/Mask_RCNN/videos/save_white_dot/'
  for i in range(frame_count):
    background_name = str(i) + '.jpg'
    background_name = os.path.join(path_background, background_name)

    overlay_name = str(i) + '.jpg'
    overlay_name = os.path.join(path_overlay, overlay_name)
    
    background = cv2.imread(background_name)
    overlay = cv2.imread(overlay_name)

    added_image = cv2.addWeighted(background,0.8,overlay,1.0,0)

    combined_name = str(i) + '.jpg'
    combined_name = os.path.join(path, combined_name)

    cv2.imwrite(combined_name, added_image)

  return

def light_color_comparison(light_color, light_color_default, color):
  # Deterines if a light turned on based on a default color, an input color and expected color

  light_comparison = []
  # A high max distance is less sensitive and a lower max distance is more sensitive
  max_distance_total = 180
  max_distance_specific_color = 90

  if color == 'Red':
    color_specific = 2
  elif color == 'Green':
    color_specific = 1
  else:
    pass

  if verbose == True:
    display(f'The Color being analyzed is {color}.')
    display(f'The default color is:')
    display(light_color_default)
    display(f'With the specific color being {light_color_default[color_specific]}')
    display(f'The max distance total is {max_distance_total}.')
    display(f'The max distance for a specific color is {max_distance_specific_color}.')

  #i cycles through each light value corresponding to each frame
  for i in range(len(light_color)):
    distance = 0
    for j in range(3):
      distance = distance + (light_color[i][j] - light_color_default[j])**2

    distance_specific_color = abs(light_color[i][color_specific] - light_color_default[color_specific])

    distance = int((distance)**(0.5))
    if vebose == True:
      display(f'The distance is {distance} and the color specific distance is {distance_specific_color} for frame {i}.')
    #0 is no color change from the default color)
    if (distance > max_distance_total and distance_specific_color > max_distance_specific_color):
      light_comparison.append(1)
      if verbose == True:
        display(f'The light is ON.')
    #1 is a color change from the default color
    else:
      light_comparison.append(0)
      if verbose == True:
        display(f'The light is OFF.')

  return (light_comparison)

def clip_vector_generator(Left_Position, Right_Position, left_light_comparison, right_light_comparison, clip_vector_previous, width):
  #Compiles the clip_vector that is used for the action analysis

  # Allows for the assumption that both lights are on if the positions are close to each other.
  # Useful if there is difficulty detecting the scoring box.
  close_bellguards = False
  # Once lights turn on it is assumed the lights stay on for the rest of the action
  light_assumption = False

  if len(Left_Position) != len(Right_Position):
    display(f'The Left and Right Positions do not match up')
  else:
    pass

  # This is either [] or the Previously saved Clip_Vector
  clip_vector = clip_vector_previous

  for i in range(len(Left_Position)):  
    # Checks the lights should be assumed on if they are not already
    # Determines if the bellguards are close to each other
    if (abs(Left_Position[i][0] - Right_Position[i][0]) < width*.050) and (light_assumption == False):
      close_bellguards = True

    # Adjusts the clip vector to reflect scoring box light assumptions
    clip_vector_temp = [[],[],[],[]]
    clip_vector_temp[0] = Left_Position[i][0]
    clip_vector_temp[1] = Right_Position[i][0]
    if (assume_lights == True and close_bellguards == True) or light_assumption == True:
      clip_vector_temp[2] = 1
      clip_vector_temp[3] = 1
      light_assumption = True
    else:
      if ignore_box_lights == True:
        clip_vector_temp[2] = 0
        clip_vector_temp[3] = 0
      else:
        clip_vector_temp[2] = left_light_comparison[i]
        clip_vector_temp[3] = right_light_comparison[i]

    clip_vector.append(clip_vector_temp)

  return (clip_vector)

def clip_vector_np_save(clip_call, file_number, clip_vector):
  # Saves the clip vector for future use
  # Clip_Call Left_Touch, Right_Touch, Simul

  # Generates the clip_vector speed based on the clip_vector
  clip_vector_speed = []
  for i in range(len(clip_vector)-1):
    clip_vector_speed.append([])
    clip_vector_speed[i].append(clip_vector[i+1][0]-clip_vector[i][0])
    # Reverses the Right Fencers position so that positive is towards the opponent
    clip_vector_speed[i].append(clip_vector[i][1]-clip_vector[i+1][1])
    clip_vector_speed[i].append(clip_vector[i+1][2])
    clip_vector_speed[i].append(clip_vector[i+1][3])

  # Generates the clip_vector acceleration based on the clip_vector
  clip_vector_acceleration = []
  for i in range(len(clip_vector_speed)-1):
    clip_vector_acceleration.append([])
    clip_vector_acceleration[i].append(clip_vector[i+1][0]-clip_vector[i][0])
    # Reverses the Right Fencers position so that positive is towards the opponent
    clip_vector_acceleration[i].append(clip_vector[i][1]-clip_vector[i+1][1])
    clip_vector_acceleration[i].append(clip_vector[i+1][2])
    clip_vector_acceleration[i].append(clip_vector[i+1][3])

  path = '/content/drive/My Drive/projects/fencing/Fencing Clips/'

  # Saves the clip_vector as a numpy array
  clip_vector_np = np.asarray(clip_vector)
  name = os.path.join(path, clip_call)
  name_2 = clip_call + '_Vector_Clips'
  name = os.path.join(name, name_2)
  if verbose == True:
    display(f'The name of the path for clip vectors to be saved is:')
    display(name)

  # Changes the directory to the sub folder of the fenncing clip
  # %cd $name
  os.chdir(name)

  clip_vector_np_name = 'clip_vector_np' + str(file_number) + '.csv'
  # Saves to the current directory
  np.savetxt(clip_vector_np_name, clip_vector_np, delimiter=',')

  # Saves the clip_vector_speed
  clip_vector_speed_np = np.asarray(clip_vector_speed)
  name = os.path.join(path, clip_call)
  name_2 = clip_call + '_Vector_Clips_Speed'
  name = os.path.join(name, name_2)

  # Changes the directory to the sub folder for the speed fencing clip
  # %cd $name
  os.chdir(name)

  clip_vector_speed_np_name = 'clip_vector_speed_np' + str(file_number) + '.csv'
  # Saves to the current directory
  np.savetxt(clip_vector_speed_np_name, clip_vector_speed_np, delimiter=',')

  # Saves the clip_vector_acceleration
  clip_vector_acceleration_np = np.asarray(clip_vector_acceleration)
  name = os.path.join(path, clip_call)
  name_2 = clip_call + '_Vector_Clips_Acceleration'
  name = os.path.join(name, name_2)

  # Changes the directory to the sub folder for the acceleration fencning clip
  # %cd $name
  os.chdir(name)

  clip_vector_acceleration_np_name = 'clip_vector_acceleration_np' + str(file_number) + '.csv'
  #Saves to the current directory
  np.savetxt(clip_vector_acceleration_np_name, clip_vector_acceleration_np, delimiter=',')

  return

def Left_Right_Test(Left_Position, Right_Position):
  # Requires that the Left and Right BellGuards be on the Left and Right sides respectively

  #Left_Position is chosen arbitrarily for length
  for i in range(len(Left_Position)):
    if Left_Position[i][0] > Right_Position[i][0]:
      display(f'The Left and Right were swapped on frame {i} and are now corrected.')
      position_temp = Left_Position[i]
      Left_Position[i] = Right_Position[i]
      Right_Position[i] = position_temp
    else:
      pass

  return (Left_Position, Right_Position)

def camera_motion_adjustment(Position, Score_Box_Position):
  # Takes a Position as an input and adjusts the position to compensate for camera motion
  # Uses solely the x position of the scoring box to calculate motion
  # Ignores the change in angle as the camera is rotated
  # This is only used when it is assumed that the Scoring Box is well detected and tracked

  Score_Box_Position_Temp = []
  #Converts Scoring Box Positions to solely x value
  #Scoring Box Position is of the format [x0,x1,x2...]
  for i in range(len(Score_Box_Position)):
    Score_Box_Position_Temp.append(Score_Box_Position[i][0])

  for j in range(len(Position)):
    score_box_delta = Score_Box_Position_Temp[j] - Score_Box_Position_Temp[0]
    Position[j][0] = Position[j][0] - score_box_delta

  return (Position)

def position_down_scale(Position1, Position2, capture_width, capture_height):
  # Scales the Position Down to the Capture Width in the x axis if required for visualization convenience
  # Does not alter the Clip Vector Data
  
  position_temp = []

  for i in range(len(Position1)):
    position_temp.append(Position1[i][0])

  for j in range(len(Position2)):
    position_temp.append(Position2[j][0])

  min_x_position = min(position_temp)
  max_x_position = max(position_temp)

  if min_x_position < 0:
    #Shifts the bellguards to the right for the camera moving to the left
    for i in range(len(Position1)):
      Position1[i][0] = int(Position1[i][0] - min_x_position)

    for j in range(len(Position2)):
      Position2[j][0] = int(Position2[j][0] - min_x_position)

  # Absolute Pixel
  if max_x_position > capture_width:
    #Scales the max x position if greater than the screen
    for i in range(len(Position1)):
      Position1[i][0] = int(Position1[i][0] * capture_width / max_x_position)

    for j in range(len(Position2)):
      Position2[j][0] = int(Position2[j][0] * capture_width / max_x_position)

  return (Position1, Position2)

if train_model == True:
  # Transfers the Bell_Guard set from Google Drive, specifically images and annotations
  # %cd /content/drive/My Drive/projects/fencing/Bell_Guard
  os.chdir('/content/drive/My Drive/projects/fencing/Bell_Guard')
  # !cp -r /content/drive/My\ Drive/projects/fencing/Bell_Guard/ /content/Mask_RCNN/
  source = '/content/drive/My Drive/projects/fencing/Bell_Guard/'
  destination = '/content/Mask_RCNN/Bell_Guard/'
  shutil.copytree(source, destination)

  #Finds the number of images used in training and testing
  # %cd /content/drive/My Drive/projects/fencing/Bell_Guard/images
  os.chdir('/content/drive/My Drive/projects/fencing/Bell_Guard/images')

  number_of_images = len(os.listdir())
  train_set_number = int(0.8*number_of_images)

  # %cd /content/Mask_RCNN/
  os.chdir('/content/Mask_RCNN/')

class Bell_GuardDataset(Dataset):
  #class that defines and loads the Bell_Guard dataset
	# load the dataset definitions
  def load_dataset(self, dataset_dir, is_train=True):
		# define the two classes
    self.add_class("dataset", 1, "bellguard")
    self.add_class("dataset", 2, "scorebox")
    self.add_class("dataset", 3, "torso")
    # self.add_class("dataset", 4, "person")

    # Adds the Images to be Analyzed
		# Defines data locations
    images_dir = dataset_dir + '/images/'
    annotations_dir = dataset_dir + '/annots/'
		# Finds all images
    for filename in listdir(images_dir):
      # Extracts image id
      image_id = filename[5:-4]
			# Skips all images after the training set
      if is_train and int(image_id) >= train_set_number:
        continue
			# Skips all images before the training set
      if not is_train and int(image_id) < train_set_number:
        continue
      img_path = images_dir + filename
      display(img_path)
      ann_path = annotations_dir + 'Image' + image_id + '.xml'
      display(ann_path)
			# Add to dataset
      self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
	# Extract bounding boxes from an annotation file
  def extract_boxes(self, filename):
		# load and parse the file
    tree = ElementTree.parse(filename)
		# get the root of the document
    root = tree.getroot()
		# extract each bounding box
    boxes = list()

    objects = root.findall('.//object')
    # Adds the object found to the end of boxes. Boxes now has 5 values instead of 4.
    objects_to_find = ['bellguard', 'scorebox', 'torso']
    for object_to_find_iterator in objects_to_find:
      # display(f'The object iterator is: {object_to_find_iterator}')
      for obj in objects:
        if (obj.find('.name').text) == object_to_find_iterator:
          xmin = obj.find('.bndbox/xmin')
          ymin = obj.find('.bndbox/ymin')
          xmax = obj.find('.bndbox/xmax')
          ymax = obj.find('.bndbox/ymax')
          coors = [int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text), object_to_find_iterator]
          # display(coors)
          boxes.append(coors)
        else:
          pass

		# Extracts image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height
 
	# Loads the masks for an image
  def load_mask(self, image_id):
		# Gets details of image
    info = self.image_info[image_id]
		# Defines box file location
    path = info['annotation']
		# Loads XML
    object_to_find = 'bellguard'
    # boxes, w, h = self.extract_boxes(path, object_to_find)
    display(f'The image_id is: {image_id}')

    boxes, w, h = self.extract_boxes(path)
		# Creates one array for all masks, each on a different channel
    masks = zeros([h, w, len(boxes)], dtype='uint8')
		# Creates masks
    class_ids = list()
    # len(boxes) is the number of 5 value lists within the list of boxes
    for i in range(len(boxes)):
      box = boxes[i]
      row_s, row_e = box[1], box[3]
      col_s, col_e = box[0], box[2]
      masks[row_s:row_e, col_s:col_e, i] = 1
      class_ids.append(self.class_names.index(box[4]))
      # class_ids.append(self.class_names.index('Bell_Guard'))

    # display(f'The class_ids in load_mask are: {asarray(class_ids, dtype="int32")}')
    return masks, asarray(class_ids, dtype='int32')
 
	# load an image reference
  def image_reference(self, image_id):
    info = self.image_info[image_id]
    return info['path']

if train_model == True:
  # Training set
  train_set = Bell_GuardDataset()
  train_set.load_dataset('Bell_Guard', is_train=True)
  train_set.prepare()
  if verbose == True:
    print('Train: %d' % len(train_set.image_ids))
  
  # Testing/Evaluation set
  test_set = Bell_GuardDataset()

  test_set.load_dataset('Bell_Guard', is_train=False)
  test_set.prepare()
  if verbose == True:
    print('Test: %d' % len(test_set.image_ids))

  # enumerate all images in the dataset
  for image_id in train_set.image_ids:
  	# load image info
  	info = train_set.image_info[image_id]

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)

if train_model == True:
  # define image id
  image_id = 1
  # load the image
  image = train_set.load_image(image_id)
  # load the masks and the class ids
  mask, class_ids = train_set.load_mask(image_id)
  # extract bounding boxes from the masks
  bbox = extract_bboxes(mask)
  # display_instances(image, bbox, mask, class_ids, train_set.class_names)
