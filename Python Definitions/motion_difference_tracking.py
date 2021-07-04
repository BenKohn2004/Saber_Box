def motion_difference_tracking(frame, side, Bounding_Box, width, height, kernel_scaling, erosion_iterations, dilation_iterations, orig_img_worpt_starting_list):

  # Ensures Bounding_Box is not negative
  for i in range(len(Bounding_Box)):
    if Bounding_Box[i] < 0:
      Bounding_Box[i] = 0

  if verbose == True:
    display(f'The original difference tracking bounding box at frame {frame - 1} is:')
    display(Bounding_Box)

  # Requires the Bounding Box to have a width and be on the image
  if Bounding_Box[1] - Bounding_Box[0] != 0 and Bounding_Box[0] < width and Bounding_Box[0] > 0:
    Position_y_Orig = int((Bounding_Box[3]+Bounding_Box[2])/2)

    # Reads the images
    image1 = orig_img_worpt_starting_list[frame - 1]
    image2 = orig_img_worpt_starting_list[frame - 2]

    # image1 = cv2.imread(file_name1)
    # image2 = cv2.imread(file_name2)

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
    
    if crop_img.shape[0] != 0 and crop_img.shape[1] != 0:

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
      
      if cnts != []:
        c = max(cnts, key=cv2.contourArea)

        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])

        if verbose == True:
          display(f'Left/Right is {left[0]}/{right[0]}.')
          display(f'The int(left[0]) is {int(left[0])} and of type {type(int(left[0]))} while left[0] is {type(left[0])}.')
          display(f'Bounding_Box[0] is {Bounding_Box[0]} and is of {type(Bounding_Box[0])}.')

        if side == 'Left':
          # Obtain outer left coordinate of the contour
          # right = tuple(c[c[:, :, 0].argmin()][0])
          position = [int(right[0]) + Bounding_Box[0], Position_y_Orig]
        elif side == 'Right':
          # left = tuple(c[c[:, :, 0].argmax()][0])
          position = [int(left[0]) + Bounding_Box[0], Position_y_Orig]
        else:
          if verbose == True:
            display(f'Side is not given')

      
      else:
        if verbose == True:
          display(f'There is no data from difference imaging on the {side} side.')
        position = 'None'

    else:
      if verbose == True:
        display(f'The crop image is null on the {side} side.')
      position = 'None'
      cnts = []

    if verbose == True:
      display(f'The kernel number for frame {frame} is {kernel_number}, the number of errosions/dilations are {erosion_iterations}/{dilation_iterations}.')
    if cnts != []:
      if verbose == True:
        display(f'The resulting position is {position} and the boundary box is {Bounding_Box}. The Left/Right limits of the contour are {int(left[0]) + Bounding_Box[0]}/{int(right[0]) + Bounding_Box[0]}.')
  else:
    position = 'None'
    if verbose == True:
      display(f'The bounding box given had a width of zero.')

  return (position)