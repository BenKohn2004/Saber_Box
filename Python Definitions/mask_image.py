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