def saturation_test(box, frame, save_image_list):
  # Test is a True/False return
  # Takes an image and tests it for the expected saturation

  img = save_image_list[frame]
  # Converts from BGR to HSV
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  
  # Tests Bellguard
  if box[2] == 1:
    blue_range = [50, 150]
    green_range = [50, 150]
    red_range = [50, 160]
    max_delta = 25
    # saturation_range = [0, 20]
    saturation_range = [0, 70]
    object_tested = 'Bellguard'
  # Tests Torso
  elif box[2] == 3:
    blue_range = [60, 150]
    green_range = [60, 150]
    red_range = [60, 160]
    max_delta = 30
    saturation_range = [0, 20]
    object_tested = 'Torso'
  else:
    if verbose == True:
      display(f'The object to test does not have a color/saturation profile.')

  width = (box[0][3]-box[0][1])
  height = (box[0][2]-box[0][0])

  s_temp = []

  #i is the x value of the image
  for i in range(width):
    #j is y value of the image
    for j in range(height):
      s = img[box[0][0] + j, box[0][1] + i, 1]
      s_temp.append(s)

    #Sorts the distances and keeps the top quarter then finds the average
    s_temp.sort()
    #Truncates to the least saturated/most gray values
    s_temp = s_temp[:(int(len(s_temp)/2)*-1)]
    s_temp = s_temp[:(int(len(s_temp)*3/4)*-1)]
    #Averages the saturation values
    s_average = int(sum(s_temp)/len(s_temp))

  if s_average < saturation_range[1]:
    test_result = True
  else:
    test_result = False

  if verbose == True:
    display(f'The test result for the {object_tested} saturation is {test_result} with a saturation of {s_average}.')

  return (test_result)