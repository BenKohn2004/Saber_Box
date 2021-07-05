def Analyze_ScoreBox_Lights(file_name, ScoreBox_Default_Green, ScoreBox_Default_Red, image_size_default, img):
  # Importing an Image and Creating a Default Hist

  # Takes a scorebox image as an array and returns it as a histogram of the appropriate corner.
  [histr_green, histr_red, image_size] = Load_ScoreBox_Image(img)

  # Normalizes the Histogram by the number of pixels in the image
  max_shape = int((max(image_size, image_size_default))/50)

  # Creates a difference array
  difference_array_Green = np.array(create_difference_array(histr_green, ScoreBox_Default_Green, max_shape))
  difference_array_Red = np.array(create_difference_array(histr_red, ScoreBox_Default_Red, max_shape))

  # Threshold for light on or off
  weighted_average_threshold_red = 1.2
  weighted_average_threshold_green = 1.0

  if verbose_det == True:

    g_d_avg = sum(ScoreBox_Default_Green[1])/len(ScoreBox_Default_Green[1])
    r_d_avg = sum(ScoreBox_Default_Red[2])/len(ScoreBox_Default_Red[2])

    g_w=weighted_average_array(difference_array_Green[1])
    r_w=weighted_average_array(difference_array_Red[2])

    display(f'G/R def_avg {g_d_avg}/{r_d_avg}.')
    display(f'Weighted Average g/r is {g_w}/{r_w}.')

  # Creates the calculated light based on an emperical threshold
  if weighted_average_array(difference_array_Green[1]) > weighted_average_threshold_green:
    green_light_calc = 1
  else:
    green_light_calc = 0

  if weighted_average_array(difference_array_Red[2]) > weighted_average_threshold_red:
    red_light_calc = 1
  else:
    red_light_calc = 0  

  return ([red_light_calc, green_light_calc])