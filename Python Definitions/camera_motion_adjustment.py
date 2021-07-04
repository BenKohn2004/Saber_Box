def camera_motion_adjustment(Position, Score_Box_Position):
  # Takes a Position as an input and adjusts the position to compensate for camera motion
  # Used solely the x position of the scoring box to calculate motion
  # Ignores the change in angle as the camera is rotated
  # This is only used when it is assumed that the Scoring Box is well detected and tracked

  Score_Box_Position_Temp = []
  #Converts Scoring Box Positions to solely x value
  #Scoring Box Position is of the format [x0,x1,x2...]
  for i in range(len(Score_Box_Position)):
    Score_Box_Position_Temp.append(Score_Box_Position[i][0])

  for j in range(len(Position)):
    score_box_delta = Score_Box_Position_Temp[j] - Score_Box_Position_Temp[0]
    Position[j][0] = Position[j][0] - score_box_delta

  return (Position)