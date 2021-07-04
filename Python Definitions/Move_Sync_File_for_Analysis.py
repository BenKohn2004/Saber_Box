def Move_Sync_File_for_Analysis(file_name):

  if not exists('/content/Mask_RCNN/videos'):
    os.mkdir('/content/Mask_RCNN/videos')

  destination = '/content/Mask_RCNN/videos'
  # file_name = clip
  file_source = '/content/drive/My Drive/Sync/'
  file_name_with_source = os.path.join(file_source, file_name)
  file_name_with_destination = os.path.join(destination, '999.mp4')
  if exists(file_name_with_destination):
    os.remove(file_name_with_destination)
    if verbose == True:
      f'{file_name} was removed from the Left Touch Folder.'
  else:
    if verbose == True:
      f'There was no file to remove from the Left Touch Folder.'
  shutil.copy(file_name_with_source, file_name_with_destination)

  if exists(file_name_with_destination):
    display(f'The file {file_name} has been moved to {destination}.')
  else:
    display(f'The file {file_name} was not successfully moved.')

  return