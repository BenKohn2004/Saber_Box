def update_scorebox_csv(name, frame_count, light_status):
  # Updates the ScoreBox csv with the Lights
  # When export_scorebox_image == True

  path = '/content/drive/My Drive/Sync/ScoreBox Images/ScoreBox DataFrame.csv'
  df = pd.read_csv(path)

  df.loc[(df['File Name'] == name) & (df['Frame']  == frame_count),['Red Light']] = light_status
  df.loc[(df['File Name'] == name) & (df['Frame']  == frame_count),['Green Light']] = light_status

  df.to_csv(path, index = False)

  return ()