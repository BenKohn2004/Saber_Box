def scorebox_bbox(bbox):
  # Creates a list of scorebox bbox

  scorebox_bbox_list = []

  # The scorebox is the second item in the bbox
  tracked_item = 2

  for i in range(len(bbox)):
    if bbox[i][2] == tracked_item:
      # Stores just the position of BBoxes, ignoring percent confidence and tracked item
      scorebox_bbox_list.append(bbox[i][0])

  return (scorebox_bbox_list)