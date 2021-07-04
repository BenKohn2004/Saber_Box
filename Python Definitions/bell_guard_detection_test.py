def bell_guard_detection_test(bbox, Torso_Box_Max, side, width, height):

  # Initializes the detection variable
  detection = False

  y_min = int(height*1/6)
  y_max = min(int(height*2/3), Torso_Box_Max)
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

    if verbose_starting == True and bbox[i][2] == 1:
      display(f'The bellguard is at [{x_avg},{y_avg}] on the {side} side with box:"')
      # display(f'bbox[i] is {bbox[i]}.')
      # display(f'bbox[i][2] is {bbox[i][2]}.')
      display(f'The x_min,x_max,y_min,y_max is {x_min},{x_max},{y_min},{y_max}')

    # Checks if the BBox is within the Bounds for plausible Bell Guard
    if  bbox[i][2] == 1 and x_min < x_avg and x_avg < x_max and y_min < y_avg and y_avg < y_max:
      detection = True
      break

  return (detection)