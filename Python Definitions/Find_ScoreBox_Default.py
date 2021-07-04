def Find_ScoreBox_Default(img_temp):
  # Uses a List of Arrays to create an Average Default Lights Off Histogram
  # Input is a Python List of Arrays for each ScoreBox Image

  # Initializes the Green and Red Histograms
  histr_green_list = []
  histr_red_list = []
  image_size_list = []

  for i in range(len(img_temp)):
    img = img_temp[i]
    if img != []:
      height, width, channel = img.shape
      image_size = height*width
      image_size_list.append(image_size)

      # Creates the Green Image using the Top Right Corner
      mask = np.zeros(img.shape[:2], dtype="uint8")
      cv2.rectangle(mask, (int(width/2), 0), (width,int(height/2)), 255, -1)
      img_green = cv2.bitwise_and(img, img, mask=mask)

      # Colors are (H L S) but (b g r) are used for graphing convenience
      color = ('b','g','r')

      histr_green = []
      for j,col in enumerate(color):
        histr_green_temp = cv2.calcHist([img_green],[j],None,[254],[1,255])
        histr_green.append(histr_green_temp)

      histr_green_list.append(histr_green)

      # Creates the Red Image using the Top Left Corner
      mask = np.zeros(img.shape[:2], dtype="uint8")
      cv2.rectangle(mask, (0,0), (int(width/2),int(height/2)), 255, -1)
      img_red = cv2.bitwise_and(img, img, mask=mask)

      histr_red = []
      for j,col in enumerate(color):
          histr_red_temp = cv2.calcHist([img_red],[j],None,[254],[1,255])
          histr_red.append(histr_red_temp)

      histr_red_list.append(histr_red)
    
  array_sum = np.zeros((3, 256))
  for i in range(len(histr_green_list)): # 10
    for j in range(len(histr_green_list[i])): # 3
      for k in range(histr_green_list[i][j].shape[0]): # 256
        array_sum[j][k] = array_sum[j][k] + histr_green_list[i][j][k][0]

  ScoreBox_Default_Green = [[],[],[]]
  for i in range(3):
    # Averages the H L S, giving a default ScoreBox with lights 'Off'
    ScoreBox_Default_Green[i] = array_sum[i]/len(histr_green_list)
    ScoreBox_Default_Green[i] = ScoreBox_Default_Green[i].astype('float32')

  array_sum = np.zeros((3, 256))
  for i in range(len(histr_red_list)): # 10
    for j in range(len(histr_red_list[i])): # 3
      for k in range(histr_red_list[i][j].shape[0]): # 256
        array_sum[j][k] = array_sum[j][k] + histr_red_list[i][j][k][0]

  ScoreBox_Default_Red = [[],[],[]]
  for i in range(3):
    # Averages the H L S, giving a default ScoreBox with lights 'Off'
    ScoreBox_Default_Red[i] = array_sum[i]/len(histr_red_list)
    ScoreBox_Default_Red[i] = ScoreBox_Default_Red[i].astype('float32')

    image_size_default = int(sum(image_size_list)/len(image_size_list))

  return (ScoreBox_Default_Green, ScoreBox_Default_Red, image_size_default)