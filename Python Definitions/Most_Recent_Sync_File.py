def Most_Recent_Sync_File():

  # Creates a List of Modified File Names and Dates modifies and selects the most recent.
  # Initializes the Lists
  date_modified_list = []
  file_size_list = []
  file_names = []
  path = '/content/drive/My Drive/Sync/'

  # Cycles through the files to generate the lists
  for file in os.listdir(path):
    # Ensures that the File is not a Folder
    if os.path.isfile(os.path.join(path, file)):
      name = os.path.join(path, file)
      file_names.append(file)
      # date_modified_list.append(os.path.getmtime(name))
      date_modified_list.append(os.path.getctime(name))
      file_size_list.append(int(os.path.getsize(name)/1024))
      # display(file)

  # if verbose == True:
  #   display(max(date_modified_list))
  # Finds the index of the most recent file name
  if date_modified_list != []:
    index_of_max = date_modified_list.index(max(date_modified_list))
    if verbose == True:
      display(f'The most recent file is {file_names[index_of_max]}.')

    clip = file_names[index_of_max]
    size = file_size_list[index_of_max]
  else:
    clip = 'None'
    size = 'None'

  return (clip, size)