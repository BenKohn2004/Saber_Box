def export_scorebox(img, bbox_list, Scoring_Box_Position):
  # Returns the image of a scorebox that contains the Scoring Box Position

  # Initializes the screbox_img
  scorebox_img = []

  # Cycles through the img_bbox to find a bbox that contains the ScoreBox Position, may cycle multiple times overwriting if multiple are possible
  for i in range(len(bbox_list)):
    # Sets the bbox to the i bbox in the list of bbox
    bbox = bbox_list[i]
    # display(f'The bbox range for {i} is x{bbox[0]}:{bbox[2]},y{bbox[1]}:{bbox[3]}.')
    # Checks if Scoring Position is within the bbox, if not then it is skipped.
    if Scoring_Box_Position[1] > bbox[0] and Scoring_Box_Position[1] < bbox[2] and Scoring_Box_Position[0] > bbox[1] and Scoring_Box_Position[0] < bbox[3]:
      # Crops the ScoreBox image
      scorebox_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]

  return (scorebox_img)