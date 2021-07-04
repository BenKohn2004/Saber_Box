def color_tester(box, frame):
  #Takes a given box and tests for a specific color range

  path = r'/content/Mask_RCNN/videos/save/'
  file_name = str(frame) + '.jpg'
  name = os.path.join(path, file_name)
  img = cv2.imread(name)

  if verbose == True:
    display(f'The file names to be color tested is {name}.')
  # box[0] are the coordinates ([y1,x1,y2,x2]), box[1] is confidence and box[2] is object
  # Tests if Bellguard is the correct color
  if box[2] == 1:
    blue_range = [50, 150]
    green_range = [50, 150]
    red_range = [50, 160]
    max_delta = 25
  elif box[2] == 3:
    blue_range = [60, 150]
    green_range = [60, 150]
    red_range = [60, 160]
    max_delta = 30
  else:
    if verbose == True:
      display(f'The object to test does not have a color profile.')

  # OpenCV uses Blue, Green, Red order
  b, g, r = 0, 0, 0

  width = (box[0][3]-box[0][1])
  height = (box[0][2]-box[0][0])

  #i is the x value of the image
  for i in range(width):
    #j is y value of the image
    for j in range(height):
      #color channel of the image [B,G,R]
      #image, img, is of format [y,x] 
      b = b + img[box[0][0] + j, box[0][1] + i, 0]
      g = g + img[box[0][0] + j, box[0][1] + i, 1]
      r = r + img[box[0][0] + j, box[0][1] + i, 2]

  # Finds the Color Averages
  b_average = int(b/(width*height))
  g_average = int(g/(width*height))
  r_average = int(r/(width*height))

  # Finds maximum differences between colors
  max_1 = abs(b_average - g_average)
  max_2 = abs(b_average - r_average)
  max_3 = abs(g_average - r_average)
  max_delta = max(max_1, max_2, max_3)

  if test_result == False:
    if verbose == True:
      display(f'The Color Test Result Failed for object {box[2]}.')

  return (test_result)