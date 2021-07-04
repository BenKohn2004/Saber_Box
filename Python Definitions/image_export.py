def image_export(file_name, img_bbox, frame_count, export_scorebox_image, Scoring_Box_Position):
  # Exports an Image of the Scorebox given an image and a bounding box
  # When export_scorebox_image == True

  # The scorebox is the second item in the bbox
  tracked_item = 2


  # bbox_list = []
  # scorebox_img_list = []
  scorebox_list = []

  # export_scorebox_image == True: 
  counter = 0
  # Cycles through the bounding box and saves an image for each scoring box detection
  # for i in range(len(bbox)):
  for i in range(len(img_bbox)):
    # Sets the bbox to bbox pair of the img_bbox
    bbox = img_bbox[i][1]
    if bbox[i][2] == tracked_item:
      # Crops the ScoreBox image
      scorebox_img = frame[bbox[i][0][0]:bbox[i][0][2], bbox[i][0][1]:bbox[i][0][3]]
      file_name_scorebox = file_name[:-4] + '_score_box_' + str(frame_count) + '_' + str(counter) + '.jpg'
      score_box_name = os.path.join('/content/drive/My Drive/Sync/ScoreBox Images/',file_name_scorebox)

      # bbox_list.append(bbox[i])
      # scorebox_img_list.append(scorebox_img)
      scorebox_list.append([scorebox_img,bbox[i]])

      # Saves file if exporting scorebox images else returns scorebox image array
      if export_scorebox_image == True:
        # Saves the ScoreBox image
        cv2.imwrite(score_box_name, scorebox_img)

        # Saves ScoreBox Data on the ScoreBox csv
        ScoreBox_DataFrame_path = '/content/drive/My Drive/Sync/ScoreBox Images/ScoreBox DataFrame.csv'
        df = pd.read_csv(ScoreBox_DataFrame_path)
        df = df.append({'File Name': file_name, 'Frame': frame_count, 'Iteration': counter, 'Red Light': np.nan, 'Green Light': np.nan,}, ignore_index=True)
        df.to_csv(ScoreBox_DataFrame_path, index = False)

      # Iterates for multiple ScoreBoxes in a single frame
      counter = counter + 1

      # Scorebox_List is of the format, [[scorebox_img, bbox],[scorebox_img, bbox],[scorebox_img, bbox]]
      # For each scorebox detection in a frame

  return (scorebox_img)