def create_folder_hierarchy(output_bbox_clip, output_foot_data, export_scorebox_image):

  # Changes the directory to the top level of My Drive
  if not exists('/content/drive/My Drive/Sync/Acceleration'):
    os.makedirs('/content/drive/My Drive/Sync/Acceleration')
    display(f'Created the Sync Acceleration directory.')
  else:
    display(f'The Sync Acceleration directory already exists.')
  # Creates the Tracked Clips Folder
  if not exists('/content/drive/My Drive/Sync/Tracked Clips'):
    os.makedirs('/content/drive/My Drive/Sync/Tracked Clips')
    display(f'Created the Sync Tracked Clips directory.')
  else:
    display(f'The Sync Tracked Clips directory already exists.')

  # Creates the Boundary Box Outputs if required
  if output_bbox_clip:
    if output_foot_data:
      if not exists('/content/drive/My Drive/Sync/Foot Data'):
        os.makedirs('/content/drive/My Drive/Sync/Foot Data')
        display(f'Created the Foot Data directory.')
      else:
        display(f'The Foot Data directory already exists.')

  # Creates the Scorebox Image Output directory and csv if required
  if export_scorebox_image:
    if not exists('/content/drive/My Drive/Sync/ScoreBox Images'):
      os.makedirs('/content/drive/My Drive/Sync/ScoreBox Images')
      display(f'Created the ScoreBox Images directory.')
    else:
      display(f'The ScoreBox Images directory already exists.')
    path = '/content/drive/My Drive/Sync/ScoreBox Images/ScoreBox DataFrame.csv'
    if not exists(path):
      d = {'File Name': [], 'Frame': [], 'Iteration': [], 'Red Light': [], 'Green Light': []}
      df = pd.DataFrame(data=d)
      df.to_csv(path, index = False)

  return ()