def boundary_box_overlap(box1, box2):
  #Finds the overlap of two boxes assume (x_min, x_max, y_min, y_max)
  
  box_overlap = [max(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), min(box1[3], box2[3])]

  return (box_overlap)