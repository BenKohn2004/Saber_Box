def read_video(file_name, folder):
  # Reads a video file and returns the video as a list of arrays for each frame and width, height, fps

  display(f'The file to be read is {os.path.join(folder, file_name)}')

  video_array_list = []
  capture = cv2.VideoCapture(os.path.join(folder, file_name))
  
  fps = int(capture.get(cv2.CAP_PROP_FPS))
  # Prevents unusually large FPS
  if fps > 100:
    fps = 30

  # capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  while True:
    ret, frame = capture.read()
    if not ret:
      break

    video_array_list.append(frame)

  capture.release()

  return (video_array_list, fps)