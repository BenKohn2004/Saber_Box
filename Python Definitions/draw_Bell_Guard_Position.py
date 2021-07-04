def draw_Bell_Guard_Position(Left_Position, Right_Position, Scoring_Box_Position, scoring_box_size_average, Left_Torso_Position, Right_Torso_Position, Left_Foot_Position, Right_Foot_Position, frame_count, Tracking_Bounding_Boxes, capture_width, capture_height, engarde_length, score_box_empty, camera_steady, camera_motion_threshold, Exclusion_Areas, simple_clip_vector, save_image_list):
  #Adds an overlay on the image to visualize the location of tracked objects

  #Color format is [B,G,R]
  left_light_color_default = [[],[],[]]
  right_light_color_default = [[],[],[]]
  left_light_color = []
  right_light_color = []

  # Creates a list of Files from a Directory
  path = r'/content/Mask_RCNN/videos/save/'


  left_light_comparison, right_light_comparison, default_color = [], [], []

  # for i, file in enumerate(files):
  for i in range(len(save_image_list)):
    # Reads the image
    # name = os.path.join(path, file)
    img = save_image_list[i]

    # OpenCV uses Blue, Green, Red order
    # Light_Color is of the format [[[B0],[G0],[R0]],[[B1],[G1],[R1]],[[B2],[G2],[R2]],...]
    
    # if i <= engarde_length:
    #   if scoring_box_size_average == [0,0]:
    #     scoring_box_size_average = [int(capture_width/5), int(capture_height/5)]
    #   if verbose == True:
    #     display(f'The average scoring box size is {scoring_box_size_average}.')
    #   # Uses a comparison of frames and scoring box position to determine the light off colors

    #   if verbose == True:
    #     display(f'The index i is {i}.')
    #     display(f'The Score Box Position is: {Scoring_Box_Position}.')
    #     display(f'The Score Box Position at i is {Scoring_Box_Position[i]}.')
    #   [left_light_comparison_temp, right_light_comparison_temp, defualt_color_temp] = scoring_box_lights(img, Scoring_Box_Position[i], scoring_box_size_average, [], i, score_box_empty)
    #   left_light_comparison.append(left_light_comparison_temp)
    #   right_light_comparison.append(right_light_comparison_temp)
    #   default_color.append(defualt_color_temp)
    #   # Averages the Default Color on the Last iteration
    #   if i == engarde_length:
    #     b_temp = int(sum(default_color[0])/len(default_color[0]))
    #     g_temp = int(sum(default_color[1])/len(default_color[1]))
    #     r_temp = int(sum(default_color[2])/len(default_color[2]))
    #     default_color = [b_temp,g_temp,r_temp]
    # elif i > engarde_length:
    #   try:
    #     [left_light_comparison_temp, right_light_comparison_temp, defualt_color_temp] = scoring_box_lights(img, Scoring_Box_Position[i], scoring_box_size_average, default_color, i, score_box_empty)
    #   except:
    #     if verbose == True:
    #       display(f'Light Comparison Failed due to Error at frame {i}.')
    #     [left_light_comparison_temp, right_light_comparison_temp, defualt_color_temp] = [0,0,[]]
    #   left_light_comparison.append(left_light_comparison_temp)
    #   right_light_comparison.append(right_light_comparison_temp)

    if verbose == True:
      display(f'The Left Position is: {Left_Position}.')
      display(f'The iterator i is {i}.')

    #Creates the dots on the Bell Guards
    frame = cv2.circle(img, (Left_Position[i][0], Left_Position[i][1]), int(4*(capture_width/1280)), (118, 37, 217), -1)
    frame = cv2.circle(frame, (Right_Position[i][0], Right_Position[i][1]), int(4*(capture_width/1280)), (157, 212, 19), -1)
    frame = cv2.circle(frame, (Scoring_Box_Position[i][0], Scoring_Box_Position[i][1]), int(4*(capture_width/1280)), (255, 255, 0), -1)
    frame = cv2.circle(frame, (Left_Torso_Position[i][0], Left_Torso_Position[i][1]), int(4*(capture_width/1280)), (0, 255, 0), -1)
    frame = cv2.circle(frame, (Right_Torso_Position[i][0], Right_Torso_Position[i][1]), int(4*(capture_width/1280)), (255, 255, 0), -1)
    # frame = cv2.circle(frame, (Left_Foot_Position[i][0], Left_Foot_Position[i][1]), int(4*(capture_width/1280)), (118, 37, 217), -1)
    # frame = cv2.circle(frame, (Right_Foot_Position[i][0], Right_Foot_Position[i][1]), int(4*(capture_width/1280)), (157, 212, 19), -1)
    
    # Creates the Representative Bell Guard Position
    frame = cv2.circle(img, (Left_Position[i][0], int(capture_height/2)), int(20*(capture_width/1280)), (118, 37, 217), -1)
    frame = cv2.circle(frame, (Right_Position[i][0], int(capture_height/2)), int(20*(capture_width/1280)), (157, 212, 19), -1)
    # frame = cv2.circle(frame, (Left_Foot_Position[i][0], int(capture_height*5/8)), int(10*(capture_width/1280)), (118, 37, 217), -1)
    # frame = cv2.circle(frame, (Right_Foot_Position[i][0], int(capture_height*5/8)), int(10*(capture_width/1280)), (157, 212, 19), -1)

    # Creates the Light Indicators
    rect_size = int(capture_width/40)
    if (simple_clip_vector[i][2] == 1):
      #Creates the Left Score Light
      frame = cv2.rectangle(frame, (rect_size, int(rect_size*1.5)), (rect_size*5, int(rect_size*4.5)), (0, 0, 255), -1)
    if (simple_clip_vector[i][3] == 1):
      #Creates the Right Score Light
      frame = cv2.rectangle(frame, (capture_width - rect_size, int(rect_size*1.5)), (capture_width - rect_size*5, int(rect_size*4.5)), (0, 255, 0), -1)

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

    # [frame, none] = overlay_keypoints(frame, keypoints[i][0], keypoints[i][1], True)

    #Draws the Exclusion Areas
    for j in range(len(Exclusion_Areas)):
      frame = cv2.circle(frame, (Exclusion_Areas[j][0],Exclusion_Areas[j][1]), int(capture_width/80), (144,238,144), 2)

    if verbose == True:
      display(f'The Tracking Box for the Left Fencer at frame {i} is:')
      display(f'{Tracking_Bounding_Boxes[i][0][0]},{Tracking_Bounding_Boxes[i][0][2]}')
      display(f'The Tracking Box for the Right Fencer at frame {i} is:')
      display(f'{Tracking_Bounding_Boxes[i][1][0]},{Tracking_Bounding_Boxes[i][1][2]}')

    #Saves the image frame overwriting the original image
    # name = os.path.join(path, file)
    file = str(i) + '.jpg'
    name = os.path.join(path, file)
    # cv2.imwrite(name, frame)
    save_image_list[i] = frame

    if verbose == True:
      display(f'The Draw Bell Guard frame {i} is being saved at {name}.')

  # # Releases capture so that other files can be used
  # capture.release()

  return (left_light_comparison, right_light_comparison, save_image_list)