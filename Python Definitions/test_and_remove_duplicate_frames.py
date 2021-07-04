def test_and_remove_duplicate_frames(original_image_list, engarde_length, remove_duplicate_frames):
  # Creates a List of unique frames with by comparing the previous and current frames
  # This compensates for video compression that may give duplicate frames when FPS is changed

  camera_steady = []
  engarde_diff_average_arr = np.array([])
  engarde_hue_average_arr = np.array([])

  original_image_worpt_list = []

  #Tests for the Same Frames and removes identical frames
  num_of_deleted_frames = 0
  frame_test = True

  for i in range(len(original_image_list)):
    # #Saves an original version of the frame without Regions of Interest

    if i > 0:
      if verbose == True:
        display(f'Performing Difference Check for Frame {i}')
      # True implies a unique frame while False is a repeat
      # [average_diff, h_average] = frame_comparison(frame_count, frame, width, height, frame_count, engarde_length, original_image_list)
      [average_diff, h_average] = frame_comparison(i, original_image_list[i], width, height, engarde_length, original_image_list)

      # Uses the Engarde Positioning to determine a baseline difference level between frames
      if i < engarde_length:
        # engarde_diff_average_list.append(average_diff)
        engarde_diff_average_arr = np.append(engarde_diff_average_arr, average_diff)
        engarde_hue_average_arr = np.append(engarde_hue_average_arr, h_average)

      elif i == engarde_length:

        engarde_diff_average_arr = np.append(engarde_diff_average_arr, average_diff)
        engarde_hue_average_arr = np.append(engarde_hue_average_arr, h_average)

        # Emperically Derived Threshold Based on Hue
        duplicate_threshold = np.average(engarde_hue_average_arr) * -.037+3.8663
        # duplicate_threshold = np.average(engarde_hue_average_arr) * -.037+ 1.5663
        camera_motion_threshold = np.percentile(engarde_diff_average_arr, 40) * camera_motion_threshold_factor
                
        if verbose == True:
          display(f'The Duplicate Threshold at the Engarde Length is {duplicate_threshold}.')
          display(f'The Camera Motion Threshold at the Engarde Length is {camera_motion_threshold}.')

      elif i > engarde_length:
        if average_diff < duplicate_threshold:
          if verbose == True:
            display(f'The frame {i} is identical to frame {i - 1}.')
          frame_test = False
        else:
          if verbose == True:
            display(f'Frame {i} is unique.')
          frame_test = True
        # display(f'{frame_count} greater than engarde length')

    # Saves the Image in Either Original or Original and Without Repeat
    # Excludes frames that are Part of the Engarde Positioning
    if (frame_test == True) or (i <= engarde_length) or (remove_duplicate_frames == False):
      # cv2.imwrite(name_orig, frame)
      # cv2.imwrite(name_orig_worpt, frame)

      # original_image_list.append(frame)
      original_image_worpt_list.append(original_image_list[i])

      if i > 0:
        camera_steady.append(average_diff)
      else:
        camera_steady.append(0)
    else:
      # cv2.imwrite(name_orig, frame)
      # original_image_list.append(frame)
      num_of_deleted_frames += 1

  return (camera_steady, duplicate_threshold, camera_motion_threshold, original_image_list, original_image_worpt_list)