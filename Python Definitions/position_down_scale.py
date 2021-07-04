def position_down_scale(Position1, Position2, capture_width, capture_height):
  # Scales the Position Down to the Capture Width in the x axis if required for visualization convenience
  # Does not alter the Clip Vector Data
  
  position_temp = []

  for i in range(len(Position1)):
    position_temp.append(Position1[i][0])

  for j in range(len(Position2)):
    position_temp.append(Position2[j][0])

  min_x_position = min(position_temp)
  max_x_position = max(position_temp)

  if min_x_position < 0:
    #Shifts the bellguards to the right for the camera moving to the left
    for i in range(len(Position1)):
      Position1[i][0] = int(Position1[i][0] - min_x_position)

    for j in range(len(Position2)):
      Position2[j][0] = int(Position2[j][0] - min_x_position)

  # Absolute Pixel
  if max_x_position > capture_width:
    #Scales the max x position if greater than the screen
    for i in range(len(Position1)):
      Position1[i][0] = int(Position1[i][0] * capture_width / max_x_position)

    for j in range(len(Position2)):
      Position2[j][0] = int(Position2[j][0] * capture_width / max_x_position)

  return (Position1, Position2)