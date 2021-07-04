def create_title_card(width, height, results, pred_total, name, img_directory):

  # Tests for and creates if needed, a folder for the font
  os.chdir('/content/')
  if not exists('/content/font'):
    os.mkdir('font')
  os.chdir('/content/font')
  if not exists('NotoSerif-hinted.zip'):
    !wget https://noto-website-2.storage.googleapis.com/pkgs/NotoSerif-hinted.zip
    !unzip "NotoSerif-hinted.zip"
  os.chdir('/content/')

  # create same size image of background color
  bg_color = (0,0,0)
  # bg = np.full((img.shape), bg_color, dtype=np.uint8)
  image = np.full((height, width,3), bg_color, dtype=np.uint8)

  # draw text on bg
  text_color = (255,255,255)
  font = cv2.FONT_HERSHEY_TRIPLEX

  scaling = width/1280
  fontScale_header= 3.8*scaling
  fontScale = 3.6*scaling
  thickness = int(4*scaling)
  certainty_cutoff = 0.1
  

  from PIL import Image

  if (pred_total * 100) > certainty_cutoff:
    # Creates the Text for the Title Card
    text1 = "Predicted Result"
    text2 = 'Left: ' + str(results[0]) + '%'
    text3 = 'Right: ' + str(results[1]) + '%'
    text4 = 'Simul: ' + str(results[2]) + '%'

    # Gets the boundary of the text. It is approximate since it uses Hershey Text
    text1size = cv2.getTextSize(text1, font, fontScale_header, thickness)[0]
    text2size = cv2.getTextSize(text2, font, fontScale, thickness)[0]
    text3size = cv2.getTextSize(text3, font, fontScale, thickness)[0]
    text4size = cv2.getTextSize(text4, font, fontScale, thickness)[0]

    # Defines the x and y values of the Text Blocks
    text1x = int((image.shape[1] - text1size[0]) / 2)
    text1y = int(height*0/10)
    text2x = int((image.shape[1] - text2size[0]) / 2)
    text2y = int(height*2/10)
    text3x = int((image.shape[1] - text3size[0]) / 2)
    text3y = int(height*4/10)
    text4x = int((image.shape[1] - text4size[0]) / 2)
    text4y = int(height*6/10)

    # Uses PIL for a so that a custom font can be used
    im = Image.fromarray(np.uint8(image))
    font_pil_header = PIL.ImageFont.truetype('/content/font/NotoSerif-SemiBold.ttf', int(scaling*140))
    font_pil = PIL.ImageFont.truetype('/content/font/NotoSerif-SemiBold.ttf', int(scaling*130))
    draw  = PIL.ImageDraw.Draw(im)

    # Draws the Text
    draw.text((text1x, text1y), text1, fill=(255,255,255,255), font=font_pil_header)
    draw.text((text2x, text2y), text2, fill=(255,255,255,255), font=font_pil)
    draw.text((text3x, text3y), text3, fill=(255,255,255,255), font=font_pil)
    draw.text((text4x, text4y), text4, fill=(255,255,255,255), font=font_pil)

  else:
    # Creates the Text for an Uncertain Result
    text1 = "Certainty is"
    text2 =  "too Low"
    # Gets the boundary of the text. It is approximate since it uses Hershey Text
    text1size = cv2.getTextSize(text1, font, fontScale, thickness)[0]
    text2size = cv2.getTextSize(text2, font, fontScale, thickness)[0]
    text1x = int((image.shape[1] - text1size[0]) / 2)
    text1y = int(height*0/10)
    text2x = int((image.shape[1] - text2size[0]) / 2)
    text2y = int(height*2/10)

    im = Image.fromarray(np.uint8(image))
    font_pil_header = PIL.ImageFont.truetype('/content/font/NotoSerif-SemiBold.ttf', int(scaling*140))
    font_pil = PIL.ImageFont.truetype('/content/font/NotoSerif-SemiBold.ttf', int(scaling*130))
    draw  = PIL.ImageDraw.Draw(im)
    draw.text((text1x, text1y), text1, fill=(255,255,255,255), font=font_pil)
    draw.text((text2x, text2y), text2, fill=(255,255,255,255), font=font_pil)

  # Converts the PIL image back into Numpy
  image = np.array(im)

  # Finds the Number of the Last frame
  ROOT_DIR = '/content/Mask_RCNN/'
  VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
  VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, img_directory)
  images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
  # Sorts the images by integer index
  images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

  last_iterator = int(images[-1].split(img_directory + '/')[1][:-4])
  display(f'The last image is {last_iterator}.')

  # Saves the image
  for i in range(25):
    save_name = name + str(i + last_iterator + 1) + '.jpg'
    save_name = os.path.join(VIDEO_SAVE_DIR, save_name)
    cv2.imwrite(save_name, image)

  return