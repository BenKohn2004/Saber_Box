def scoring_box_lights(img, Scoring_Box_Position, scoring_box_size_average, default_color, frame, score_box_empty):

  # A high max distance is less sensitive and a lower max distance is more sensitive
  max_distance_total = 200
  max_distance_specific_color = 100

  # Defines the region of the top_left position of a 5x3 grid of the score_box, [xmin,ymin,xmax,ymax]
  # Extends the Light Search Position outside of the detected box
  xmin = Scoring_Box_Position[0] - int(scoring_box_size_average[0]/2) - int(scoring_box_size_average[0]/8)
  xmax = Scoring_Box_Position[0] - int(scoring_box_size_average[0]/2) + int(scoring_box_size_average[0]/4)
  ymin = Scoring_Box_Position[1] - int(scoring_box_size_average[1]/2)
  ymax = Scoring_Box_Position[1] - int(scoring_box_size_average[1]/2) + int(scoring_box_size_average[1]/3)
  left_light_position = [xmin, xmax, ymin, ymax]

  # Defines the region of the top_right position of a 5x3 grid of the score_box, [xmin,ymin,xmax,ymax]
  xmin = Scoring_Box_Position[0] + int(scoring_box_size_average[0]/2) - int(scoring_box_size_average[0]/4)
  xmax = Scoring_Box_Position[0] + int(scoring_box_size_average[0]/2) + int(scoring_box_size_average[0]/8)
  ymin = Scoring_Box_Position[1] - int(scoring_box_size_average[1]/2)
  ymax = Scoring_Box_Position[1] - int(scoring_box_size_average[1]/2) + int(scoring_box_size_average[1]/3)
  right_light_position = [xmin, xmax, ymin, ymax]

  if default_color != []:
    distance_temp, distance_specific_color_temp = [], []

    width = left_light_position[1]-left_light_position[0]
    height = left_light_position[3]-left_light_position[2]

    #i is the x value of the image for the Left Side/Red
    for i in range(width):
      #j is y value of the image
      for j in range(height):
        #color channel of the image [B,G,R]
        #image, img, is of format [y,x]
        pixel_position_y = left_light_position[2] + j
        pixel_position_x = left_light_position[0] + i
        b = (img[pixel_position_y, pixel_position_x, 0] - default_color[0])
        g = (img[pixel_position_y, pixel_position_x, 1] - default_color[1])
        r = (img[pixel_position_y, pixel_position_x, 2] - default_color[2])
        distance_temp.append(int((b**2 + g**2 + r**2)**(0.5)))
        distance_specific_color_temp.append(abs(r))

    #Sorts the distances and keeps the top quarter then finds the average
    distance_temp.sort()
    distance_temp = distance_temp[(int(len(distance_temp)/4)*-1):]
    distance = int(sum(distance_temp)/len(distance_temp))
    distance_specific_color_temp.sort()
    distance_specific_color_temp = distance_specific_color_temp[(int(len(distance_specific_color_temp)/4)*-1):]
    distance_specific_color = int(sum(distance_specific_color_temp)/len(distance_specific_color_temp))

    #0 is no color change from the default color)
    if distance > max_distance_total and distance_specific_color > max_distance_specific_color and score_box_empty == False:
      left_light_comparison = 1
    #1 is a color change from the default color
    else:
      left_light_comparison = 0

    #Resets b,g,r for the Right Side
    distance_temp, distance_specific_color_temp= [], []
    width = right_light_position[1]-right_light_position[0]
    height = right_light_position[3]-right_light_position[2]

    #i is the x value of the image
    for i in range(width):
      #j is y value of the image
      for j in range(height):
        #kcolor channel of the image [B,G,R]

        # pixel_position = right_light_position[2] + j,right_light_position[0] + i
        pixel_position_y = right_light_position[2] + j
        pixel_position_x = right_light_position[0] + i
        b = (img[pixel_position_y, pixel_position_x, 0] - default_color[0])
        g = (img[pixel_position_y, pixel_position_x, 1] - default_color[1])
        r = (img[pixel_position_y, pixel_position_x, 2] - default_color[2])
        distance_temp.append(int((b**2 + g**2 + r**2)**(0.5)))
        distance_specific_color_temp.append(abs(g))

    #Sorts the distances and keeps the top sixth then finds the average
    distance_temp.sort()
    distance_temp = distance_temp[(int(len(distance_temp)/6)*-1):]
    distance = int(sum(distance_temp)/len(distance_temp))
    distance_specific_color_temp.sort()
    distance_specific_color_temp = distance_specific_color_temp[(int(len(distance_specific_color_temp)/4)*-1):]
    distance_specific_color = int(sum(distance_specific_color_temp)/len(distance_specific_color_temp))

    #0 is no color change from the default color)
    if (distance > max_distance_total and distance_specific_color > max_distance_specific_color):
      right_light_comparison = 1
    #1 is a color change from the default color
    else:
      right_light_comparison = 0

  #Finds the Defualt Color
  else:
    b, g, r = 0, 0, 0
    # Cycles through the Left and Right Light Positions to determine a default color for the frame
    width = left_light_position[1]-left_light_position[0]
    height = left_light_position[3]-left_light_position[2]
    for i in range(width):
      for j in range(height):
        pixel_position_y = left_light_position[2] + j
        pixel_position_x = left_light_position[0] + i
        b = b + img[pixel_position_y, pixel_position_x, 0]
        g = g + img[pixel_position_y, pixel_position_x, 1]
        r = r + img[pixel_position_y, pixel_position_x, 2]
        default_color_left_temp = [int(b/(width*height)),int(g/(width*height)),int(r/(width*height))]
    width = right_light_position[1]-right_light_position[0]
    height = right_light_position[3]-right_light_position[2]
    for i in range(width):
      for j in range(height):
        # pixel_position = right_light_position[2] + j,right_light_position[0] + i
        pixel_position_y = left_light_position[2] + j
        pixel_position_x = left_light_position[0] + i
        b = b + img[pixel_position_y, pixel_position_x, 0]
        g = g + img[pixel_position_y, pixel_position_x, 1]
        r = r + img[pixel_position_y, pixel_position_x, 2]
        default_color_right_temp = [int(b/(width*height)),int(g/(width*height)),int(r/(width*height))]
    #Combines the Left and Right Default Colors for B,G,R
    for i in range(3):
      default_color.append((default_color_left_temp[i] + default_color_right_temp[i])/2)

    # Assumes that the lights are off during the engarde phase.
    left_light_comparison = 0
    right_light_comparison = 0

  return (left_light_comparison, right_light_comparison, default_color)