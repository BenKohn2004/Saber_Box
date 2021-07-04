def create_representative_image(clip_vector, capture_width, capture_height):
  # Allows for an overlay that represents the bellguard horizontal motion and box lights

  #Creates a Folder to save the images and removes previous version
  os.chdir('/content/Mask_RCNN/videos')
  # Removes and Recreates the Save_White_Dot to ensure the directory is empty
  try:
    shutil.rmtree('save_white_dot')
    if verbose == True:
      display(f'Removed the Save_White_Dot folder.')
  except:
    if verbose == True:
      display(f'ERROR removing the Save_White_Dot folder.')
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