def setup_clip_directories():
  # Creates the Save, Original and Original without Repeats Folders

  os.chdir('/content/')

  if not exists('/content/Mask_RCNN'):
    os.mkdir('Mask_RCNN')

  os.chdir('/content/Mask_RCNN/')

  display(f'os.getcwd() is: {os.getcwd()}')
  ROOT_DIR = os.getcwd()
  MODEL_DIR = os.path.join(ROOT_DIR, "logs")
  VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
  VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")
  VIDEO_ORIG_DIR = os.path.join(VIDEO_DIR, "original")
  VIDEO_ORIGWORPT_DIR = os.path.join(VIDEO_DIR, "original_without_repeats")
  display(f'The ROOT_DIR is: {ROOT_DIR}')

  # Removes and recreates the save directory effectively emptying the folder
  # Attempts to remove folders to ensure folders are empty
  try:
    shutil.rmtree('videos')
  except:
    display(f'The video directory did not exist.')
  os.mkdir('videos')
  os.chdir('/content/Mask_RCNN/videos')
  os.mkdir('save')
  os.mkdir('original')
  os.mkdir('original_without_repeats')


  return (ROOT_DIR, VIDEO_DIR, VIDEO_SAVE_DIR, VIDEO_ORIG_DIR)