def bbox_output(video_name, bbox_clip, output_foot_data):
  # Creates a CSV file of each detection of the format:
  # [frame, x_center, y_center, detection confidence]

  tracked_item = 0

  if output_foot_data == True:
    tracked_item = 4

  if tracked_item != 0:
    clip_items = []

    for i in range(len(bbox_clip)):
      for j in range(len(bbox_clip[i])):
        if bbox_clip[i][j][2] == tracked_item:
          x_center = int((bbox_clip[i][j][0][3]+bbox_clip[i][j][0][1])/2)
          y_center = int((bbox_clip[i][j][0][2]+bbox_clip[i][j][0][0])/2)
          clip_items.append([i,x_center,y_center, bbox_clip[i][j][1]])

    df = pd.DataFrame(clip_items, columns=['frame','x', 'y', 'confidence'])
#     %cd '/content/drive/My Drive/Sync/Foot Data'
    name = video_name[:-4] + '_foot.csv'
    df.to_csv(name, index=False, header = True)


  return ()