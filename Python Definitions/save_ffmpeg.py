def save_ffmpeg(source, destination, name, fps):

  os.chdir(source)
  file_name_with_destination = os.path.join(destination, name)

  if verbose == True:
    display(f'The source of the files is {source}.')
    display(f'The save destination is {destination}.')
    display(f'Name with destionation is {file_name_with_destination}.')
    display(f'The name is {name}.')

  !ffmpeg -framerate $fps -i %1d.jpg output.mp4

  file_name_with_destination = os.path.join(destination, name)
  name = name + '.mp4'

  shutil.copy('output.mp4', file_name_with_destination)

  return