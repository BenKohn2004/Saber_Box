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

  #Defines the File Path
  path = r'/content/Mask_RCNN/videos/overlay/'
  path_background = r'/content/Mask_RCNN/videos/save/'
  path_foreground = r'/content/Mask_RCNN/videos/save_white_dot/'
  for i in range(frame_count):
    background_name = str(i) + '.jpg'
    background_name = os.path.join(path_background, background_name)

    foreground_name = str(i) + '.jpg'
    foreground_name = os.path.join(path_foreground, foreground_name)
    
    background = cv2.imread(background_name)
    foreground = cv2.imread(foreground_name)

    added_image = cv2.addWeighted(background,0.8,foreground,1.0,0)

    combined_name = str(i) + '.jpg'
    combined_name = os.path.join(path, combined_name)

    if verbose == True:
      display(f'The file added image is saved at {combined_name}.')

    cv2.imwrite(combined_name, added_image)

  return