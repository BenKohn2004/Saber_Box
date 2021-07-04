def box_size_finder(bbox, capture_width, capture_height, object_to_size):

  Box_Size = [[],[]]
  sum_of_boxes = [[],[]]
  frame_multiplier = 1

  if object_to_size == 'score_box':
    x_min = int(capture_width/4)
    x_max = int(capture_width*3/4)
    bbox_category = 2
  elif object_to_size == 'left':
    x_min = 0
    x_max = int(capture_width/2)
    bbox_category = 3
  elif object_to_size == 'right':
    x_min = int(capture_width/2)
    x_max = int(capture_width)
    bbox_category = 3

  # i represents the frame, minimum of 50 frames or len(bbox)
  for i in range(min(50*frame_multiplier, len(bbox))):
    # j represents the rois(specific bounding box) within the frame sorted by confidence score
    for j in range(len(bbox[i])):
      if (bbox[i][j][1] > 0.90 and bbox[i][j][0][1] > x_min and bbox[i][j][0][1] < x_max and bbox[i][j][2] == bbox_category):
        #Appends x value:
        sum_of_boxes[0].append(bbox[i][j][0][1])
        #Appends y value:
        sum_of_boxes[1].append(bbox[i][j][0][0])  
        #Appends x width value:
        Box_Size[0].append(bbox[i][j][0][3] - bbox[i][j][0][1])
        #Appends y width value:
        Box_Size[1].append(bbox[i][j][0][2] - bbox[i][j][0][0])

  x_average = average_list(sum_of_boxes[0])
  y_average = average_list(sum_of_boxes[1])

  # scoring_box_size_average [Width, Height]
  box_size_average = []
  # Appends the average scoring box width
  box_size_average.append(int(average_list(Box_Size[0])))
  # Appends the average scoring box height
  box_size_average.append(int(average_list(Box_Size[1])))

  if verbose == True:
    display(f'The Average Box Size for {object_to_size} is {box_size_average}')

  return (box_size_average)