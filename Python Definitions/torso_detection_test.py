def torso_detection_test(bbox, side, width, height):

  # Initializes the detection variable
  detection = False

  y_min = int(height*1/6)
  y_max = int(height*2/3)
  if side == 'Left':
    x_min = 0
    x_max = width*1/2
  elif side == 'Right':
    x_min = width*1/2
    x_max = width

  for i in range(len(bbox)):
    # Defines the Center of the BBox
    x_avg = int((bbox[i][0][1]+bbox[i][0][3])/2)
    y_avg = int((bbox[i][0][0]+bbox[i][0][2])/2)
    # Used in Bell Guard to verify bellguard is above the bottom of the Torso Detection
    y_box_max = int(bbox[i][0][2])
    # Checks if the BBox is within the Bounds for plausible Bell Guard
    if  bbox[i][2] == 3 and x_min < x_avg and x_avg < x_max and y_min < y_avg and y_avg < y_max:
      detection = True
      break

  return (detection, y_box_max)