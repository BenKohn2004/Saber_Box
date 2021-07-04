def tracking_box_default(Left, Right, Score_Box, x_padding, y_padding, engarde_length):
  # Creates a default tracking box

  Tracking_Bounding_Boxes_Temp = [[],[],[]]
  Tracking_Bounding_Boxes = []

  for i in range(engarde_length):
    Tracking_Bounding_Boxes_Temp[0].append(Left[0] - x_padding)
    Tracking_Bounding_Boxes_Temp[0].append(Left[0] + x_padding)
    Tracking_Bounding_Boxes_Temp[0].append(Left[1] - y_padding)
    Tracking_Bounding_Boxes_Temp[0].append(Left[1] + y_padding)

    Tracking_Bounding_Boxes_Temp[1].append(Right[0] - x_padding)
    Tracking_Bounding_Boxes_Temp[1].append(Right[0] + x_padding)
    Tracking_Bounding_Boxes_Temp[1].append(Right[1] - y_padding)
    Tracking_Bounding_Boxes_Temp[1].append(Right[1] + y_padding)

    Tracking_Bounding_Boxes_Temp[2].append(Score_Box[0] - x_padding)
    Tracking_Bounding_Boxes_Temp[2].append(Score_Box[0] + x_padding)
    Tracking_Bounding_Boxes_Temp[2].append(Score_Box[1] - y_padding)
    Tracking_Bounding_Boxes_Temp[2].append(Score_Box[1]+ y_padding)

    Tracking_Bounding_Boxes.append(Tracking_Bounding_Boxes_Temp)

  return (Tracking_Bounding_Boxes)