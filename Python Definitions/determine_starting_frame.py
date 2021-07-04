def determine_starting_frame(original_image_worpt_list, width, height):
  # Determines the first usable frame in a clip

  # Sets up the clip directories
  [ROOT_DIR, VIDEO_DIR, VIDEO_SAVE_DIR, VIDEO_ORIG_DIR] = setup_clip_directories()

  os.chdir('/content/drive/My Drive/')  

  total_frames = len(original_image_worpt_list)
  if total_frames == 0:
    display(f'ERROR: The Video Clip selected has no frames.')
  display(f'The total number of frames in the video are: {total_frames}')

  # frame_check keeps a running account of detected object in each frame
  frame_check = []
  # Determines when the first complete frame of a clip is found
  test_frame_check = False
  # Initialize the frame iterator, counts the number of frames analyzed
  k = 0
  # Number of Frames to Sample
  group_size = 10
  # Number of Frames that must be true
  min_positives = 2
  # min_positives = 1

  while test_frame_check == False and k < len(original_image_worpt_list):

    # display(f'k is {k} and the length of orig_worpt is {len(original_image_worpt_list)}.')

    frame = original_image_worpt_list[k]
    # Runs the Detection Model
    [bbox, none] = yolov4_run_image(frame, False)

    if bbox != []:

      if verbose_starting:
        display(f'The bbox at frame {k}:')
        display(bbox)

      # Tests if there is a viable Left and Right Torso in the Frame
      if verbose_starting:
        display(f'Torso Detection for frame {k}.')
      [Left_Torso_Detection, Left_Torso_Box_Max] = torso_detection_test(bbox, 'Left', width, height)
      [Right_Torso_Detection, Right_Torso_Box_Max]  = torso_detection_test(bbox, 'Right', width, height)

      # Tests if there is a viable Left and Right Bell Guard in the Frame
      if verbose_starting:
        display(f'Bell Guard Detection for frame {k}.')
      Left_Bell_Guard_Detection = bell_guard_detection_test(bbox, Left_Torso_Box_Max, 'Left', width, height)
      Right_Bell_Guard_Detection  = bell_guard_detection_test(bbox, Right_Torso_Box_Max, 'Right', width, height)

      if verbose_starting:
        display(f'At frame {k} the L/R Torso is {Left_Torso_Detection}/{Right_Torso_Detection} and L/R Bellguard is {Left_Bell_Guard_Detection}/{Right_Bell_Guard_Detection}. ')

      # Checks if all the detections are True and appends 1 or 0 to the frame_check.
      if Left_Bell_Guard_Detection == Right_Bell_Guard_Detection == Left_Torso_Detection == Right_Torso_Detection == True:
        frame_check.append(1)
      else:
        frame_check.append(0)

    # Increments the frame iterator
    k = k + 1

    # Tests the frame_check to determine if there are sufficient positives
    if len(frame_check) > group_size:
      sum_of_values = 0
      for i in range(group_size):
        sum_of_values = sum_of_values + frame_check[len(frame_check) - i - 1]
  
      if verbose_starting == True:
        display(f'The number of positives is {sum_of_values}.')
      if sum_of_values >= min_positives:
        if verbose_starting == True:
          display(f'The threshold has been met. Success.')
        test_frame_check = True
      else:
        if verbose_starting == True:
          display(f'The minimum has not been achieved. Failure.')

  starting_frame = k - group_size

  return (starting_frame, total_frames)