def number_of_files_in_dir(path):
  # Returns the number of files in a directory
  files = [i for i in os.listdir(path)]
  number_of_files = len(files)

  return (number_of_files)