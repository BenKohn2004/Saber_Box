def yolov4_run_image(image, return_image):

  # %cd '/content/darknet/'

  os.chdir('/content/darknet/')

  # run test on person.jpg image that comes with repository

  detections, width_ratio, height_ratio = darknet_helper(image, width, height)

  bbox_temp = []
  names_of_objects = ['BG', 'bellguard', 'scorebox', 'torso', 'foot']

  if return_image:
    for label, confidence, bbox in detections:
      left, top, right, bottom = bbox2points(bbox)
      left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
      cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
      cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        class_colors[label], 2)
  else:
    image = 'None'

  for label, confidence, bbox in detections:
    left, top, right, bottom = bbox2points(bbox)
    left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
    arr = np.array([top, left, bottom, right])

    # change the dtype to 'float64' 
    arr = arr.astype('int32')
    percentage = float(math.floor(float(confidence))/100)
    class_id = names_of_objects.index(label)

    bbox_temp.append([arr, percentage, class_id])

  return (bbox_temp, image)