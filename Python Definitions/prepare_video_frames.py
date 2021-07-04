def prepare_video_frames(image_save_directory, video_directory, video_name):
  VIDEO_DIR = video_directory
  VIDEO_SAVE_DIR = image_save_directory
  images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
  # Sort the images by integer index
  images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

  name = str(video_name) + '.mp4'
  outvid = os.path.join(VIDEO_DIR, name)

  return (outvid, images)