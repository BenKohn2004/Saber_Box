touch_folders = ['Left_Touch', 'Right_Touch', 'Simul']

root_path = r'/content/drive/My Drive/projects/fencing/Fencing Clips/'
video_dir_clips = root_path + touch_folders[clip_call]
display(f'The video_dir_clips path is: {video_dir_clips}')

path = r'/content/Mask_RCNN/videos/'

[bbox, frame_count, capture_width, capture_height, clip_vector_previous, fencer_data, keypoints, clip_vector, fps] = \
  process_video_clip(video_filename, touch_folders[clip_call], remove_duplicate_frames)

# Removes the .mp4 from the String
if simplified == True:
  iterator = video_filename[:-4]

#Saves the Clip, Speed and Acceleration Vectors
clip_vector_np_save(touch_folders[clip_call], iterator, clip_vector)

#Saves Images for the Representative Video
create_representative_image(clip_vector, capture_width, capture_height)

# Prepares and Downloads videos
if download_videos == True:
  # Prepares Output Video
  [outvid, images] = prepare_video_frames('save', 'out')
  make_video(outvid, images, fps=fps)

  # Downloads Output Video
  name = '/content/Mask_RCNN/videos/' + str(iterator) + '.out.mp4'
  display(name)
  files.download(name)

  #Prepares Representative Video
  [outvid, images] = prepare_video_frames('save_white_dot', 'representative_out')
  make_video(outvid, images, fps=fps)

  # Downloads Representative Video
  name =  '/content/Mask_RCNN/videos/' + str(iterator) + '.representative_out.mp4'
  files.download(name)

  # Prepares Overlay Video
  create_overlay_image(len(clip_vector))
  [outvid, images] = prepare_video_frames('overlay', 'overlay_out')
  make_video(outvid, images, fps=fps)

  # Downloads Overlay Video
  name =  '/content/Mask_RCNN/videos/' + str(iterator) + '.overlay_out.mp4'
  files.download(name)

# Analyzes the Video Clip
engarde_position_buffer = 15
max_length = 103
clip_vector_length = max_length - engarde_position_buffer

if simplified == True:
  save_path = '/content/drive/My Drive/'
else:
  save_path = '/content/drive/My Drive/projects/fencing/Fencing Clips/'
model = load_model(os.path.join(save_path, 'ROW_model.h5'))

display(touch_folders[clip_call])

x = load_clip(touch_folders[clip_call], iterator, max_length)

pred = model.predict(x)
display(f'The predicted touch is Left {int(pred[0][0]*100)}%, Right {int(pred[0][1]*100)}%, Simul {int(pred[0][2]*100)}%.')
pred_total = pred[0][0] + pred[0][1] + pred[0][2]
display(f'The normalized predicted touch is Left {int(pred[0][0]/pred_total*100)}%, Right {int(pred[0][1]/pred_total*100)}%, Simul {int(pred[0][2]/pred_total*100)}%.')
