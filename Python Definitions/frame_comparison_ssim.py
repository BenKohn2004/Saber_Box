def frame_comparison_ssim(frame_number, frame, width, height, engarde_length):

  # Determines if the subsequent frame is identical to the current or if there was camera motion
  # Uses Mean Square Error and Structural Similarity Index

  save_path = r'/content/Mask_RCNN/videos/original/'
  # image_num = frame_number
  image_name1 = str(frame_number-1) + '.jpg'
  file_name1 = os.path.join(save_path, image_name1)
  # file_name2 = os.path.join(save_path, image_name2)

  image1 = cv2.imread(file_name1)
  image2 = frame

  # Uses a tighter crop for engarde positioning to minimize motion outside the bout
  if frame_number <= engarde_length:
    crop_image1 = image1[int(height*1/5):int(height*3/4), 0:width]
    crop_image2 = image2[int(height*1/5):int(height*3/4), 0:width]
  else:
    # Removes the bottom of the frame to minimize the effect of overlays and shadowing in the foreground
    crop_image1 = image1[int(height*0):int(height*2/4), 0:width]
    crop_image2 = image2[int(height*0):int(height*2/4), 0:width]

  # Calculate MSE
  m = np.linalg.norm(image1 - image2)
  
  # # If GrayScale
  # s = ssim(imageA, imageB)
  # If Color
  s = ssim(crop_image1, crop_image2, multichannel=True)

  if verbose == True:
    display(f'The Mean Square Error of frame {frame_number} is {m}.')
    display(f'The Structural Similarity Index of frame {frame_number} is {s}.')

  return (m, s)

def frame_comparison(frame_number, frame, width, height, engarde_length, original_image_list):
  # Determines if the subsequent frame is identical to the current or if there was camera motion
  # By calculating an average Hue from an HSV image. The Hue is then correlated to an average 
  # color difference between frames.

  image1 = original_image_list[frame_number - 1]
  image2 = frame

  # Uses a tighter crop for engarde positioning to minimize motion outside the bout
  if frame_number <= engarde_length:
    crop_image1 = image1[int(height*1/5):int(height*3/4), 0:width]
    crop_image2 = image2[int(height*1/5):int(height*3/4), 0:width]
  else:
    # Removes the bottom of the frame to minimize the effect of overlays and shadowing in the foreground
    crop_image1 = image1[int(height*0):int(height*3/4), 0:width]
    crop_image2 = image2[int(height*0):int(height*3/4), 0:width]

  #Convert to Grayscale and find the Difference
  image1_gray = cv2.cvtColor(crop_image1, cv2.COLOR_BGR2GRAY)
  image2_gray = cv2.cvtColor(crop_image2, cv2.COLOR_BGR2GRAY)
  
  # Finds the HSV of image2
  image2_HSV = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
  h_average = np.average(image2_HSV[0])

  # Uses Uncropped Frames
  # image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  # image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  # image_diff = cv2.absdiff(image1_gray,image2_gray)
  image_diff = cv2.absdiff(crop_image1,crop_image2)
  # image_diff_color = cv2.absdiff(image1,image2)

  if verbose == True:
    display(f'The max for the difference of frame {frame_number} is {np.amax(image_diff)}.')
    display(f'The average/median for the difference of frame {frame_number} is {np.average(image_diff)}/{np.median(image_diff)}.')

  if verbose == True:
    display(f'The shape of image_diff is {image_diff.shape}.')

  average_image_diff = np.average(image_diff)
  if verbose == True:
    display(f'The image difference average is {np.average(image_diff)} for frame {frame_number}.')

  # average_diff = np.average(image_diff)
  # average_diff = np.average(image_diff_color)

  return (average_image_diff, h_average)