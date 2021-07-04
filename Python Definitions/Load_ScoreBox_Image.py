def Load_ScoreBox_Image(img):
  # Loads an Array and preprocesses it so that it can be compared to a defualt image array
  # Returns two Arrays, one representing the Green Light Location and the other the Red Light Location

  # Input an scorebox image as an array and 

  # Takes an image and returns the histogram of the top left (Red) and top right (green) of the image
  # Format (H,L,S), with H,L,S represented by a numpy array of 256 float 32
  
  height, width, channel = img.shape
  image_size = height*width

  # Creates the Green Image using the Top Right Corner
  mask = np.zeros(img.shape[:2], dtype="uint8")
  cv2.rectangle(mask, (int(width/2), 0), (width,int(height/2)), 255, -1)
  img_green = cv2.bitwise_and(img, img, mask=mask)
  # img_green_hls = cv2.cvtColor(img_green, cv2.COLOR_RGB2HLS)

  color = ('b','g','r')

  histr_green = []
  for j,col in enumerate(color):
    histr_green_temp = cv2.calcHist([img_green],[j],None,[256],[1,255])
    # plt.plot(histr_green_temp,color = col)
    # plt.xlim([1,255])
    # Reshapes (256, 1) to (256)
    histr_green_temp.shape = (256)
    histr_green.append(histr_green_temp)
  # plt.show()

  # Creates the Red Image using the Top Left Corner
  mask = np.zeros(img.shape[:2], dtype="uint8")
  cv2.rectangle(mask, (0,0), (int(width/2),int(height/2)), 255, -1)
  img_red = cv2.bitwise_and(img, img, mask=mask)
  # img_red_hls = cv2.cvtColor(img_red, cv2.COLOR_RGB2HLS)

  histr_red = []
  for j,col in enumerate(color):
    histr_red_temp = cv2.calcHist([img_red],[j],None,[256],[1,255])
    # plt.plot(histr_red_temp,color = col)
    # plt.xlim([1,255])
    histr_red_temp.shape = (256)
    histr_red.append(histr_red_temp)
  # plt.show()

  return (histr_green, histr_red, image_size)