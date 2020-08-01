#Bell_GuardConfig
class Bell_GuardConfig(Config):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# define the name of the configuration
	NAME = "Bell_Guard_cfg"
	NUM_CLASSES = 1 + 3
	# number of training steps per epoch
	STEPS_PER_EPOCH = 131

# Train_Model
if train_model == True:
	# prepare train set
	train_set = Bell_GuardDataset()
	if verbose == True:
		display(f'Loading the Train DataSet')
	train_set.load_dataset('Bell_Guard', is_train=True)
	#train_set.load_dataset('kangaroo', is_train=True)
	train_set.prepare()
	if verbose == True:
		print('Train: %d' % len(train_set.image_ids))
	
	# prepare test/val set
	test_set = Bell_GuardDataset()
	display(f'Loading the Test DataSet')
	test_set.load_dataset('Bell_Guard', is_train=False)
	test_set.prepare()
	if verbose == True:
		print('Test: %d' % len(test_set.image_ids))
	# Prepare config
	config = Bell_GuardConfig()
	config.display()
 
	# Define the model
	model = MaskRCNN(mode='training', model_dir='./', config=config)
	# Load weights (mscoco) and exclude the output layers
	model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

	# Train weights (output layers or 'heads')

	augmentation = imgaug.augmenters.Fliplr(0.5)
	# # model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
	# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads', augmentation = imgaug.augmenters.Sometimes(0.5, [
	#                     imgaug.augmenters.Fliplr(0.5),
	#                     imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
	# 								]))
	#Bypasses an Error
	model.keras_model.metrics_tensors = []
	# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads', augmentation=augmentation, IMAGE_META_SIZE = 8)
	model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads', augmentation=augmentation)

# Save_Model
if train_model == True:
  # Moves mask_rcnn_bell_guard_cfg_0005.h5 to the Mask_RCNN directory
  file = !find  bell_guard_* -type d
  file = str(file[0])
  file = '/content/Mask_RCNN/' + file + '/mask_rcnn_bell_guard_cfg_0005.h5'
  # !cp {file} /content/Mask_RCNN/
  source = '/content/drive/My Drive/projects/fencing/Bell_Guard/'
  destination = '/content/Mask_RCNN/'
  shutil.copy(file, destination)

else:
  # Load the pre-trained fencing object detection model
  # !cp -r /content/drive/My\ Drive/mask_rcnn_bell_guard_cfg_0005.h5 /content/Mask_RCNN/
  source = '/content/drive/My Drive/mask_rcnn_bell_guard_cfg_0005.h5'
  destination = '/content/Mask_RCNN/'
  shutil.copy(source, destination)

# Evaluate_Model
# Defines the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
  NAME = "bell_guard_cfg"
	# number of classes (background + bell_guard + scorebox + torso)
  NUM_CLASSES = 1 + 3
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
  APs = list()
  for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
		# convert image into one sample
    sample = expand_dims(scaled_image, 0)
		# make prediction
    yhat = model.detect(sample, verbose=0)
		# extract results for first sample
    r = yhat[0]
		# calculate statistics, including AP
    AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
    APs.append(AP)
	# calculate the mean AP across all images
  mAP = mean(APs)
  return mAP

cfg = PredictionConfig()

if train_model == True:
  # Uncomment this section to train a new model
  # load the train dataset
  train_set = Bell_GuardDataset()
  #train_set = KangarooDataset()
  train_set.load_dataset('Bell_Guard', is_train=True)
  #train_set.load_dataset('kangaroo', is_train=True)
  train_set.prepare()
  print('Train: %d' % len(train_set.image_ids))
  # load the test dataset
  test_set = Bell_GuardDataset()
  #test_set = KangarooDataset()
  test_set.load_dataset('Bell_Guard', is_train=False)
  #test_set.load_dataset('kangaroo', is_train=False)
  test_set.prepare()
  print('Test: %d' % len(test_set.image_ids))
  # create config
  cfg = PredictionConfig()
  # define the model
  model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
  # load model weights
  model_path = 'mask_rcnn_bell_guard_cfg_0005.h5'
  #model_path = 'mask_rcnn_kangaroo_cfg_0005.h5'
  model.load_weights(model_path, by_name=True)
  #model.load_weights('mask_rcnn_kangaroo_cfg_0005.h5', by_name=True)
  # evaluate model on training dataset
  train_mAP = evaluate_model(train_set, model, cfg)
  print("Train mAP: %.3f" % train_mAP)
  # evaluate model on test dataset
  test_mAP = evaluate_model(test_set, model, cfg)
  print("Test mAP: %.3f" % test_mAP)

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores, file_name):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        pass

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        # label = boxes[i][4]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        # display(f'The mask is: {mask}')
        # image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

class InferenceConfig(Config):
# class InferenceConfig(coco.CocoConfig):
  GPU_COUNT = 1
  #Same as batch_size
  IMAGES_PER_GPU = 1
  NUM_CLASSES = 4
  KEYPOINT_MASK_POOL_SIZE = 7

def save_clip_progress(bbox, frame_count, capture_width, capture_height, clip_vector_previous):

  # Counts the files in the directory '/content/Mask_RCNN/videos/save/'

  # %cd /content/Mask_RCNN/videos/save/
  os.chdir('/content/Mask_RCNN/videos/save/')
  number_of_stored_frames = len(os.listdir())
  display(f'The number of stored frames in the save folder is {number_of_stored_frames}.')

  if number_of_stored_frames > 500:
    #Passes the Bounding Boxes to Determine position of items of interest
    [Left_Position, Right_Position, Scoring_Box_Position, scoring_box_size_average, Tracking_Bounding_Boxes, \
     Left_Torso_Position, Right_Torso_Position] = Bell_Guard_Position_Finding(bbox, capture_width, capture_height)

    #Draws the Boxes on the image frame and determines scoring lights turned on
    [left_light_comparison, right_light_comparison] = draw_Bell_Guard_Position(Left_Position, Right_Position, \
      Scoring_Box_Position, scoring_box_size_average, Left_Torso_Position, Right_Torso_Position, frame_count, \
      Tracking_Bounding_Boxes, video_filename, capture_width, capture_height, engarde_length)

    #Adjusts the Bellguard Position Based on the Camera motion as determined by the Score_Box Position
    Left_Position = camera_motion_adjustment(Left_Position, Scoring_Box_Position)
    Right_Position = camera_motion_adjustment(Right_Position, Scoring_Box_Position)

    #Adjusts Left and Right Position for convenient visualization
    [Left_Position, Right_Position] = position_down_scale(Left_Position, Right_Position, capture_width, capture_height)

    #Creates a vector representing the clip, format [left_x, right_x, left_lights, right_lights]
    clip_vector = clip_vector_generator(Left_Position, Right_Position, left_light_comparison, right_light_comparison, clip_vector_previous)

    # clip_vector = smooth_clip_vector(clip_vector, engarde_length)

    clip_call = 'Temp_Clip_Vector'
    file_number = '1'

    #Saves the Clip, Speed and Acceleration Vectors
    clip_vector_np_save(clip_call, file_number, clip_vector)

  else:
    pass

  return()

def smooth_clip_vector(clip_vector, engarde_length):
  # Allows for smoothing the clip_vector

  a = []
  b = []
  for i in range(engarde_length, len(clip_vector)):
    a.append(clip_vector[i][0])
    b.append(clip_vector[i][1])

  x = np.linspace(engarde_length,len(clip_vector), len(clip_vector) - engarde_length)

  # sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
  # yhata = signal.sosfilt(sos, a)
  if len(a)%2 == 1:
    yhata = signal.savgol_filter(a, len(a), 11)
    yhatb = signal.savgol_filter(b, len(b), 11)
  else:
    yhata = signal.savgol_filter(a, len(a) - 1, 11)
    yhatb = signal.savgol_filter(b, len(b) - 1, 11)    

  # plt.plot(x,a, color='black')
  # plt.plot(x,yhata, color='red')
  plt.plot(x,b, color='black')
  plt.plot(x,yhatb, color='blue')
  plt.show()

  vector_clip_smooth = []

  for j in range(len(clip_vector)):
    if j <= engarde_length:
      clip_vector_smooth_temp = [clip_vector[j][0], clip_vector[j][1], clip_vector[j][2], clip_vector[j][3]]
    else:
      clip_vector_smooth_temp = [int(yhata[j - engarde_length]), int(yhatb[j - engarde_length]), clip_vector[j][2], clip_vector[j][3]]
    vector_clip_smooth.append(clip_vector_smooth_temp)

  return (vector_clip_smooth)

def load_clip_vector():
  # Only used for large clips

  display(f'Loading the Clip Vector...')
  filename = r'/content/drive/My Drive/projects/fencing/Fencing Clips/Temp_Clip_Vector/Temp_Clip_Vector_Clips/clip_vector_np1.csv'

  display(f'Attempting to load:')
  display(filename)
  try:
    vector_data = pd.read_csv(filename, header=None)
    arr = vector_data.to_numpy(dtype = np.int32)
    clip_vector = arr.tolist()
  except:
    display(f'Load Failure...')
    display(f'The clip_vector did not exist so it is set to []')
    clip_vector = []

  return (clip_vector)

def create_tracking_masks(previous_positions, certainty, frame_count, torso_size, width, height):
  #Creates Tracking Boxes that can be used to mask the image, ignoring parts that are not of interest
  #Format, Tracking_Boxes = [Left, Right, Scorebox], Left = [x_min, x_max, y_min, y_max]
  #Format, Previous Positions
          # previous_positions  = [[Left_Position[-1], Left_Position[-2]], \
          #                      [Right_Position[-1], Right_Position[-2]], \
          #                      [Scoring_Box_Position[-1], Scoring_Box_Position[-2]], \
          #                      [Left_Torso_Position[-1], Left_Torso_Position[-2]], \
          #                      [Right_Torso_Position[-1], Right_Torso_Position[-2]]]

          #Format, positions are [x,y]

  #Format, torso_position = [[Left_x,Lefty],[Right_x,Right_y]]
  #Format, torso_size = [[Lw,Lh], [Rw,Rh]]

  if verbose == True:
    display(f'Creating Tracking Masks...')
    display(f'The Previous Positions are:')
    display(previous_positions)
    display(f'The torso sizes are:')
    display(torso_size)

  frame_mask = []

  #Certainty is the number of times the bellguard has not been detected in previous frames
  #certainty_default is the minimum size of the tracking box
  certainty_default = int(width/16)
  #certainty_multiplier is how much the tracking box enlarges following a missed
  certainty_multiplier = int(width/80)
  y_limiter = 24

  #Max allowed speed of a bellguard in a single frame
  max_speed = int(width/48)

  display(f'The length of the previous positions is: {len(previous_positions)}.')

  for i in range(len(previous_positions)):
    display(f'The masking iteration for frame {frame_count} is {i}.')
    #FINDS THE LEFT MASKING BOX
    x_pos = previous_positions[i][0][0]
    y_pos = previous_positions[i][0][1]
    #Converts previous position into a speed
    x_speed = min(previous_positions[i][0][0] - previous_positions[i][1][0], max_speed)
    # Limits the maximum vertical speed with relation to x
    y_speed = min(previous_positions[i][0][1] - previous_positions[i][1][1], int(max_speed/y_limiter))

    display(f'x and y position is ({x_pos},{y_pos}) and the speeds are ({x_speed},{y_speed}).')

    x_min = x_pos + (x_speed) - (certainty[i]*certainty_multiplier) - certainty_default
    x_max = x_pos + (x_speed) + (certainty[i]*certainty_multiplier) + certainty_default
    y_min = y_pos + (y_speed) - (certainty[i]*certainty_multiplier) - certainty_default
    y_max = y_pos + (y_speed) + (certainty[i]*certainty_multiplier) + certainty_default

    #Appends the mask to collection of tracked areas
    frame_mask.append([x_min, x_max, y_min, y_max])

  display(f'The Frame Mask for frame {frame_count} is:')
  display(frame_mask)

  return(frame_mask)

def make_video(outvid, images=None, fps=25, size=None,
               is_color=True, format="FMP4"):
  
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

def mean_of_a_numpy_percentile(arr, percentile_cutoff):
  # Returns a percentile value of a numpy array

  display(f'The average of arr is {np.average(arr)}.')

  percentile_value = np.percentile(arr, percentile_cutoff)

  # Uses just the percentile without averaging
  array_percentile_mean = percentile_value

  return (array_percentile_mean)

def frame_comparison_ssim(frame_number, frame, width, height, frame_count, engarde_length):

  # Determines if the subsequent frame is identical to the current or if there was camera motion
  # Uses Mean Square Error and Structural Similarity Index

  save_path = r'/content/Mask_RCNN/videos/original/'
  image_num = frame_number
  image_name1 = str(image_num-1) + '.jpg'
  file_name1 = os.path.join(save_path, image_name1)
  # file_name2 = os.path.join(save_path, image_name2)

  image1 = cv2.imread(file_name1)
  image2 = frame

  # Uses a tighter crop for engarde positioning to minimize motion outside the bout
  if frame_number <= engarde_length:
    crop_image1 = image1[int(height*1/5):int(height*3/4), 0:width]
    crop_image2 = image2[int(height*1/5):int(height*3/4), 0:width]
  else:
    # Removes the bottom of the frame to minimize the effect of overlays and shadowing in the foreground
    crop_image1 = image1[int(height*0):int(height*2/4), 0:width]
    crop_image2 = image2[int(height*0):int(height*2/4), 0:width]

  # Calculate MSE
  m = np.linalg.norm(image1 - image2)
  
  # # If GrayScale
  # s = ssim(imageA, imageB)
  # If Color
  s = ssim(crop_image1, crop_image2, multichannel=True)

  if verbose == True:

    display(f'The Mean Square Error of frame {frame_count} is {m}.')
    display(f'The Structural Similarity Index of frame {frame_count} is {s}.')

  return(m, s)

def frame_comparison(frame_number, frame, width, height, frame_count, engarde_length):
  # Determines if the subsequent frame is identical to the current or if there was camera motion
  # By calculating an average Hue from an HSV image. The Hue is then correlated to an average 
  # color difference between frames.

  save_path = r'/content/Mask_RCNN/videos/original/'
  image_num = frame_number
  # image_name2 = str(image_num) + '.jpg'
  image_name1 = str(image_num-1) + '.jpg'
  file_name1 = os.path.join(save_path, image_name1)
  # file_name2 = os.path.join(save_path, image_name2)

  image1 = cv2.imread(file_name1)
  image2 = frame

  # Uses a tighter crop for engarde positioning to minimize motion outside the bout
  if frame_number <= engarde_length:
    crop_image1 = image1[int(height*1/5):int(height*3/4), 0:width]
    crop_image2 = image2[int(height*1/5):int(height*3/4), 0:width]
  else:
    # Removes the bottom of the frame to minimize the effect of overlays and shadowing in the foreground
    crop_image1 = image1[int(height*0):int(height*3/4), 0:width]
    crop_image2 = image2[int(height*0):int(height*3/4), 0:width]

  #Convert to Grayscale and find the Difference
  image1_gray = cv2.cvtColor(crop_image1, cv2.COLOR_BGR2GRAY)
  image2_gray = cv2.cvtColor(crop_image2, cv2.COLOR_BGR2GRAY)
  

  # Finds the HSV of image2
  image2_HSV = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
  h_average = np.average(image2_HSV[0])

  # Uses Uncropped Frames
  # image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  # image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  # image_diff = cv2.absdiff(image1_gray,image2_gray)
  image_diff = cv2.absdiff(crop_image1,crop_image2)
  # image_diff_color = cv2.absdiff(image1,image2)

  if verbose == True:
    display(f'The max for the difference of frame {frame_count} is {np.amax(image_diff)}.')
    display(f'The average/median for the difference of frame {frame_count} is {np.average(image_diff)}/{np.median(image_diff)}.')
    # display(f'The max for the color difference of frame {frame_count} is {np.amax(image_diff_color)}.')
    # display(f'The average/median for the color difference of frame {frame_count} is {np.average(image_diff_color)}/{np.median(image_diff_color)}.')


  display(f'The shape of image_diff is {image_diff.shape}.')

  # display(f'The shape of image_diff is {image_diff_color.shape}.')

  # if frame_number <= engarde_length:
  #   image_percentile_mean = mean_of_a_numpy_percentile(image_diff, image_percentile)
  #   # image_percentile_mean = mean_of_a_numpy_percentile(image_diff_color, image_percentile)
  #   display(f'The image percentile mean is {image_percentile_mean} for frame {frame_count}.')
  # else:
  #   image_percentile_mean = np.average(image_diff)
  #   display(f'The image difference average is {np.average(image_diff)} for frame {frame_count}.')

  average_image_diff = np.average(image_diff)
  display(f'The image difference average is {np.average(image_diff)} for frame {frame_count}.')

  # average_diff = np.average(image_diff)
  # average_diff = np.average(image_diff_color)

  return(average_image_diff, h_average)

def test_and_remove_duplicate_frames(file_name, touch_folder, ROOT_DIR, engarde_length):
  # Creates a List of unique frames with by comparing the previous and current frames
  # This compensates for video compression that may give duplicate frames when FPS is changed

  camera_steady = []
  engarde_diff_average_arr = np.array([])
  engarde_hue_average_arr = np.array([])

  VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
  VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")
  VIDEO_ORIG_DIR = os.path.join(VIDEO_DIR, "original")
  VIDEO_ORIGWORPT_DIR = os.path.join(VIDEO_DIR, "original_without_repeats")

  display(f'The video directory is {VIDEO_DIR}/{file_name}')
  capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, file_name))

  total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames == 0:
    display(f'ERROR: The Video Clip selected has no frames.')
  display(f'The total number of frames in the video are: {total_frames}')

  width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(capture.get(cv2.CAP_PROP_FPS))

  capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  #Tests for the Same Frames and removes identical frames
  frame_count = 0
  num_of_deleted_frames = 0
  frame_test = True

  while True:
    ret, frame = capture.read()
    if not ret:
      break
    display(f'frame_count is {frame_count}.')

    #Saves an original version of the frame without Regions of Interest
    name_orig = '{0}.jpg'.format(frame_count)
    name_orig = os.path.join(VIDEO_ORIG_DIR, name_orig)
    name_orig_worpt = '{0}.jpg'.format(frame_count - num_of_deleted_frames)
    name_orig_worpt = os.path.join(VIDEO_ORIGWORPT_DIR, name_orig_worpt)

    if frame_count > 0:
      if verbose == True:
        display(f'Performing Difference Check for Frame {frame_count}')
      # True implies a unique frame while False is a repeat
      [average_diff, h_average] = frame_comparison(frame_count, frame, width, height, frame_count, engarde_length)

      # [m,s] = frame_comparison_ssim(frame_count, frame, width, height, frame_count, engarde_length)
      # average_diff = s
      # duplicate_threshold = s/10
      # camera_motion_threshold = s*10

      # Uses the Engarde Positioning to determine a baseline difference level between frames
      if frame_count < engarde_length:
        # engarde_diff_average_list.append(average_diff)
        engarde_diff_average_arr = np.append(engarde_diff_average_arr, average_diff)
        engarde_hue_average_arr = np.append(engarde_hue_average_arr, h_average)

      elif frame_count == engarde_length:


        engarde_diff_average_arr = np.append(engarde_diff_average_arr, average_diff)
        engarde_hue_average_arr = np.append(engarde_hue_average_arr, h_average)

        # Emperically Derived Threshold Based on Hue
        duplicate_threshold = np.average(engarde_hue_average_arr) * -.037+3.8663
        camera_motion_threshold = np.percentile(engarde_diff_average_arr, 40) * camera_motion_threshold_factor
                
        if verbose == True:
          display(f'The Duplicate Threshold at the Engarde Length is {duplicate_threshold}.')
          display(f'The Camera Motion Threshold at the Engarde Length is {camera_motion_threshold}.')

      elif frame_count > engarde_length:
        if average_diff < duplicate_threshold:
          if verbose == True:
            display(f'The frame {frame_count} is identical to frame {frame_count - 1}.')
          frame_test = False
        else:
          if verbose == True:
            display(f'Frame {frame_count} is unique.')
          frame_test = True
        # display(f'{frame_count} greater than engarde length')

    # Saves the Image in Either Original or Original and Without Repeat
    # Excludes frames that are Part of the Engarde Positioning
    if (frame_test == True) or (frame_count <= engarde_length):
      cv2.imwrite(name_orig, frame)
      cv2.imwrite(name_orig_worpt, frame)
      if frame_count > 0:
        camera_steady.append(average_diff)
      else:
        camera_steady.append(0)
    else:
      cv2.imwrite(name_orig, frame)
      num_of_deleted_frames += 1
    frame_count += 1

  # Releases the Video Capture
  capture.release()

  # Saves the new video file over the original
  # Directory of images to run detection on
  # %cd /content/Mask_RCNN/
  os.chdir('/content/Mask_RCNN/')
  ROOT_DIR = os.getcwd()
  VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
  VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "original_without_repeats")
  images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
  # Sort the images by integer index
  images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

  # name = str(iterator) + '.mp4'
  # name = str(file_name) + '.mp4'
  name = file_name
  display(f'The iterator file_name is {name}.')
  outvid = os.path.join(VIDEO_DIR, name)
  make_video(outvid, images, fps=fps)

  return (camera_steady, duplicate_threshold, camera_motion_threshold)

def process_video_clip(file_name, touch_folder, remove_duplicate_frames):
  # Processes the video
  display(f'The video file_name is: {file_name}')

  # Initiates Conditions
  score_box_empty = False
  right_torso_empty = False
  left_torso_empty = False
  left_position_empty = False
  right_position_empty = False

  # %cd /content/Mask_RCNN
  os.chdir('/content/Mask_RCNN/')
  # !mkdir videos
  try:
    os.mkdir('videos')
  except:
    display(f'ERROR creating the video directory')
  display(f'os.getcwd() is: {os.getcwd()}')
  ROOT_DIR = os.getcwd()
  MODEL_DIR = os.path.join(ROOT_DIR, "logs")
  VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
  VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")
  VIDEO_ORIG_DIR = os.path.join(VIDEO_DIR, "original")
  VIDEO_ORIGWORPT_DIR = os.path.join(VIDEO_DIR, "original_without_repeats")
  display(f'The ROOT_DIR is: {ROOT_DIR}')

  COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

  display(f'The COCO model path is : {COCO_MODEL_PATH}')

  # Removes and recreates the save directory effectively emptying the folder
  # !rm -r /content/Mask_RCNN/videos/save
  # !rm -r /content/Mask_RCNN/videos/original
  # !rm -r /content/Mask_RCNN/videos/original_without_repeats
  # Attempts to remove folders to ensure folders are empty
  os.chdir('/content/Mask_RCNN/videos')
  try:
    shutil.rmtree('save')
    shutil.rmtree('original')
    shutil.rmtree('original_without_repeats')
  except:
    display(f'Error removing save/original/original_without_repeats folders')
  # %cd /content/Mask_RCNN/videos
  # os.chdir('/content/Mask_RCNN/videos')
  # !mkdir save
  # !mkdir original
  # !mkdir original_without_repeats
  os.mkdir('save')
  os.mkdir('original')
  os.mkdir('original_without_repeats')

  # %cd /content/Mask_RCNN
  os.chdir('/content/Mask_RCNN')

  # Copies the Video from the Video Clip folder 
  # path = r'/content/drive/My\ Drive/projects/fencing/Fencing\ Clips/' + touch_folder + '/' + file_name
  path = '/content/drive/My Drive/projects/fencing/Fencing Clips/' + touch_folder + '/' + file_name
  display(f'The path is: {path}')
  # !cp $path /content/Mask_RCNN/videos
  destination = '/content/Mask_RCNN/videos'
  shutil.copy(path, destination)

  engarde_length = 10

  # Removes Duplicates and Detects Camera motion in Frames
  if remove_duplicate_frames == True:
    [camera_steady, duplicate_threshold, camera_motion_threshold] = test_and_remove_duplicate_frames(file_name, touch_folder, ROOT_DIR, engarde_length)
    display(f'The duplicate frames of video {file_name}.mp4 have been removed')

  display(f'The COCO model path is : {COCO_MODEL_PATH}')

  if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH) 

  config = InferenceConfig()
  config.display()
  model = MaskRCNN(mode='inference', model_dir='./', config=cfg)

  model_path = 'mask_rcnn_bell_guard_cfg_0005.h5'
  model.load_weights(model_path, by_name=True)

  class_names = ['BG', 'Bell_Guard', 'Score_Box', 'Torso']

  capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, file_name))

  total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames == 0:
    display(f'ERROR: The Video Clip selected has no frames.')
  display(f'The total number of frames in the video are: {total_frames}')

  try:
    if not os.path.exists(VIDEO_SAVE_DIR):
          os.makedirs(VIDEO_SAVE_DIR)
  except OSError:
    print ('Error: Creating directory of data')

  frames = []
  frame_count = 0

  width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(capture.get(cv2.CAP_PROP_FPS))

  display(f'The capture width is: {width}')
  display(f'The capture height is: {height}')

  capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # bbox = []
  keypoints = []
  fencer_data = []
  Left_Position = []
  Right_Position = []
  Scoring_Box_Position = []
  Left_Torso_Position = []
  Right_Torso_Position = []
  left_torso_size_average = []
  right_torso_size_average = []
  Tracking_Bounding_Boxes = []
  
  t0 = time.time()

  skip_frames = False
  skip_frame_counter = 0
  frames_to_skip = 300
  number_of_frames_skipped = 0
  # Assumes any video clip of greater than 10000 frames may require multiple runs
  if total_frames > 10000:
    clip_vector_previous = load_clip_vector()
    frame_count = len(clip_vector_previous)
  else:
    clip_vector_previous = []
  
  # Continues until it breaks by not finding a return from attempting to read a frame capture
  while True:
    if (frame_count%20) == 0:
      t1 = time.time()
      display(f'Processing frame {frame_count} of {total_frames}. Time elapsed {hms_string(t1 - t0)}.')
    if (frame_count%2000) == 0 and (frame_count != 0):
      display(f'Saving Clip Progress...')
      save_clip_progress(bbox, frame_count, width, height, clip_vector_previous)
    #Creates Bounding Box List
    ret, frame = capture.read()
    #Excludes the Tracking Boxes for the Engarde Length    
    if not ret:
      break
    # Save each frame of the video to a list
    frame_count += 1
    if skip_frames == False:
      t10 = time.time()
      frames = [frame]

      # Runs the Detection Model for Bellguard, Torso, Scoring_Box
      results = model.detect(frames, verbose=0)
      # Runs the Human Pose Analysis
      [fencer_data_temp, keypoints_temp] = human_pose_analysis(frame)
      fencer_data.append(fencer_data_temp)
      keypoints.append(keypoints_temp)

      for i, item in enumerate(zip(frames, results)):
        frame = item[0]
        r = item[1]        
        name = '{0}.jpg'.format(frame_count + i - 1)
        #Saves an original version of the frame without Regions of Interest
        name_orig = os.path.join(VIDEO_ORIG_DIR, name)
        cv2.imwrite(name_orig, frame)

        # Format: display_instances(image, boxes, masks, ids, names, scores):
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], file_name)

        #Saves the farme as an image with ids overlayed
        name = os.path.join(VIDEO_SAVE_DIR, name)
        cv2.imwrite(name, frame)

        #Captures the Bounding Box data in bbox
        bbox = []
        for j in range(len(r['rois'])):
          bbox.append([r['rois'][j],r['scores'][j],r['class_ids'][j]])
        a = r['class_ids'].tolist()

      t11 = time.time()
      # Displays the time required to process a given frame
      display(f'The time to process frame {frame_count} is {hms_string(t11 - t10)}.')

      if frame_count <= engarde_length:
        if frame_count == 1:
          t2 = time.time()
        [Left_Position_Temp, Right_Position_Temp, Scoring_Box_Position_Temp, scoring_box_size_average, Tracking_Bounding_Boxes_Temp, \
          Left_Torso_Position_Temp, Right_Torso_Position_Temp, left_torso_size_average_Temp, right_torso_size_average_Temp] \
          = engarde_position(bbox, width, height, engarde_length, frame_count-1)

        Left_Position.append(Left_Position_Temp)
        Right_Position.append(Right_Position_Temp)
        Scoring_Box_Position.append(Scoring_Box_Position_Temp)
        Left_Torso_Position.append(Left_Torso_Position_Temp)
        Right_Torso_Position.append(Right_Torso_Position_Temp)
        left_torso_size_average.append(left_torso_size_average_Temp)
        right_torso_size_average.append(right_torso_size_average_Temp)
        display(f'At frame {frame_count} the tracking Bounding Boxes Temp is:')
        display(Tracking_Bounding_Boxes_Temp)
        Tracking_Bounding_Boxes.append(Tracking_Bounding_Boxes_Temp)

      # Displays the time to process the engarde positioning frames
      display(f'Processing frame {frame_count} of {total_frames}. Time elapsed {hms_string(t1 - t0)}.')

      if frame_count == engarde_length:

        display(f'Time elapsed processing the engarde positions: {hms_string(t2 - t0)}.')
        display(f'Commencing the Engarde Length Processing.')
        tracked_items = [Left_Position, Right_Position, Scoring_Box_Position, Left_Torso_Position, Right_Torso_Position, left_torso_size_average, right_torso_size_average]

        if verbose == True:
          display(f'The Scoring Box Position is:')
          display(Scoring_Box_Position)
        if max(Scoring_Box_Position) == []:
          display(f'The Scoring Box Position was empty so a default was used.')
          score_box_empty = True
          tracked_items[2] = [[width/2, height/2], [width/2, height/2], [width/2, height/2]]


        # Tests for empty Positions
        if max(Right_Torso_Position) == []:
          right_torso_empty = True

        if max(Left_Torso_Position) == []:
          left_torso_empty = True

        if max(Left_Position) == [] and max(Left_Torso_Position) != []:
          left_position_empty = True

        if max(Right_Position) == [] and max(Right_Torso_Position) != []:
          right_position_empty = True
        
        # Replaces [0,0] with [] for torso width and height
        for k in range(len(tracked_items)):
          for j in range(len(tracked_items[k])):
            if tracked_items[k][j] == [0,0]:
              tracked_items[k][j] = []

        for k in range(len(tracked_items)):
          if verbose == True:
            display(f'k is {k}.')
            display(tracked_items[k])
          try:
            tracked_items[k] = [item for item in tracked_items[k] if item != []]
            tracked_items[k] = np.hstack(tracked_items[k])
            if verbose == True:
              display(f'The length of len(tracked_items[k]) is {int(len(tracked_items[k])/2)}.')
            tracked_items[k] = tracked_items[k].reshape(int(len(tracked_items[k])/2), 2)
            tracked_items[k] = np.median(tracked_items[k], axis = 0)
            tracked_items[k] = [int(tracked_items[k][0]), int(tracked_items[k][1])]
            if verbose == True:
              display(f'The tracked item position {tracked_items[k]}.')
          except:
            display(f'Failure to detect the tracked item {k} during the engarde positioning.')
            display(tracked_items[k])
            tracked_items[k] = [0,0]

        if right_torso_empty == True:
          display(f'The Right Torso Position was empty so a default based on the Right BellGuard was used.')
          display(f'tracked_items[1] is {tracked_items[1]}, left_torso_size_average is {left_torso_size_average}.')
          torso_position_default = [tracked_items[1][0] + int(left_torso_size_average[0][0]*3/4), tracked_items[1][1] - int(left_torso_size_average[0][1]/4)]
          tracked_items[4] = torso_position_default
          display(f'As a comparison, the Score Box Position is:')
          display(Scoring_Box_Position)
          display(f'The Right Torso Position is:')
          display(Right_Torso_Position)
          display(f'As a comparison, the tracked_items[2] is:')
          display(tracked_items[2])
          display(f'The tracked_items[4] is:')
          display(tracked_items[4])

        if left_torso_empty == True:
          display(f'The Left Torso Position was empty so a default based on the Left BellGuard was used.')
          display(f'tracked_items[0] is {tracked_items[0]}, right_torso_size_average is {right_torso_size_average}.')
          torso_position_default = [tracked_items[0][0] - int(right_torso_size_average[0][0]*3/4), tracked_items[0][1] - int(right_torso_size_average[0][1]/4)]
          tracked_items[3] = torso_position_default
          display(f'As a comparison, the Score Box Position is:')
          display(Scoring_Box_Position)
          display(f'The Right Torso Position is:')
          display(Right_Torso_Position)
          display(f'As a comparison, the tracked_items[2] is:')
          display(tracked_items[2])
          display(f'The tracked_items[4] is:')
          display(tracked_items[4])

        if left_position_empty == True:
          display(f'The Left  Position was empty so a default based on the Left Torso was used.')
          left_position_default = [tracked_items[3][0] + int(left_torso_size_average[0][0]), tracked_items[3][1] + int(left_torso_size_average[0][0]/4)]
          tracked_items[0] = left_position_default

        if right_position_empty == True:
          display(f'The Right  Position was empty so a default based on the Right Torso was used.')
          right_position_default = [tracked_items[4][0] - int(right_torso_size_average[0][0]), tracked_items[4][1] + int(right_torso_size_average[0][0]/4)]
          tracked_items[1] = right_position_default

        [Left_Position, Right_Position, Scoring_Box_Position, Left_Torso_Position, Right_Torso_Position] = [[],[],[],[],[]]

        # Builds the Positions of the Tracked Items
        for i in range(engarde_length):
          Left_Position.append(tracked_items[0])
          Right_Position.append(tracked_items[1])
          Scoring_Box_Position.append(tracked_items[2])
          Left_Torso_Position.append(tracked_items[3])
          Right_Torso_Position.append(tracked_items[4])

        left_torso_size_average = tracked_items[5]
        if right_torso_empty == False:
          right_torso_size_average = tracked_items[6]
        elif right_torso_empty == True:
          right_torso_size_average = tracked_items[5]
        torso_size = [left_torso_size_average, right_torso_size_average]

        if verbose == True:
          display(f'The left_torso_size_average is {left_torso_size_average} and the right_torso_size_average is {right_torso_size_average}.')

      if frame_count > engarde_length:

        positions = [Left_Position, Right_Position, Scoring_Box_Position, Left_Torso_Position, Right_Torso_Position]

        # Sets the certainty following the engarde positioning
        if frame_count == engarde_length + 1:
          t3 = time.time()
          certainty = [0,0,0,0,0]
          if verbose == True:
            display(f'The positions following the engarde positioning are:')
            display(positions)

        if verbose == True:
          display(f'Certainty just prior to Bell Guard Positioning is {certainty}.')

        previous_certainty = certainty

        if right_torso_size_average[0] == 0 and left_torso_size_average[0] == 0:
          display(f'Error, both Torso sizes are zero.')   
        elif right_torso_size_average[0] == 0:
            display(f'Right Torso Size was zero, using the Left as a Defualt.')
            right_torso_size_average = left_torso_size_average
        elif left_torso_size_average[0] == 0:
            display(f'Left Torso Size was zero, using the Right as a Defualt.')
            left_torso_size_average = right_torso_size_average

        # Finds the Tracked Items and Returns their positions
        [Left_Position_Temp, Right_Position_Temp, Scoring_Box_Position_Temp, Tracking_Bounding_Boxes_Temp, \
         Left_Torso_Position_Temp, Right_Torso_Position_Temp, engarde_length, certainty] \
         = Bell_Guard_Position_Finding(bbox, width, height, fencer_data_temp, positions, frame_count, \
         left_torso_size_average, right_torso_size_average, engarde_length, certainty, camera_steady, camera_motion_threshold)

        if verbose == True:
          display(f'Certainty just after to Bell Guard Positioning is {certainty}.')
          display(f'The Left Position at frame {frame_count - 1} is {Left_Position_Temp}.')

        # Appends the Returned Positions
        Left_Position.append(Left_Position_Temp)
        Right_Position.append(Right_Position_Temp)
        Scoring_Box_Position.append(Scoring_Box_Position_Temp)
        Left_Torso_Position.append(Left_Torso_Position_Temp)
        Right_Torso_Position.append(Right_Torso_Position_Temp)
        Tracking_Bounding_Boxes.append(Tracking_Bounding_Boxes_Temp)

        # Tests for a change in certainty to zero from non-zero. If a position has become certain during
        # this frame then it back calculates previous uncertain position up to a certain position.
        if (certainty[0] == 0 and previous_certainty[0] != 0):
          Left_Position = position_linear_approximation(Left_Position, previous_certainty[0])
          if verbose == True:
            display(f'Using a Linear Approximation for frame {frame_count} for the Left Bellguard Position.')
        elif (certainty[1] == 0 and previous_certainty[1] != 0):
          if verbose == True:
            display(f'The Right Position is: {Right_Position}.')
            display(f'The Previous Certainty is: {previous_certainty[1]}')
            display(f'Using a Linear Approximation for frame {frame_count} for the Right Bellguard Position.')
          Right_Position = position_linear_approximation(Right_Position, previous_certainty[1])
        elif (certainty[2] == 0 and previous_certainty[2] != 0):
          Scoring_Box_Position = position_linear_approximation(Scoring_Box_Position, previous_certainty[2])
        elif (certainty[3] == 0 and previous_certainty[3] != 0):
          Left_Torso_Position = position_linear_approximation(Left_Torso_Position, previous_certainty[3])
        elif (certainty[4] == 0 and previous_certainty[4] != 0):
          Right_Torso_Position = position_linear_approximation(Right_Torso_Position, previous_certainty[4])
        else:
          pass

  t4 = time.time()
  if verbose == True:
    display(f'Time elapsed to process the engarde frames is  {hms_string(t3 - t2)}.')
    display(f'Time elapsed to process the post engarde frames is  {hms_string(t4 - t3)}.')
    display(f'Time elapsed processing the clip the total clip is {hms_string(t4 - t0)}.')
    # Reduces the Frame Count to account for skipped frames
    display(f'The original frame count was: {frame_count - 1} and the number of frames skipped is: {number_of_frames_skipped}.')
  frame_count = len(bbox)
  if verbose == True:
    display(f'The length of the frame_count is {frame_count - 1} while the number of bboxes is {len(bbox)}.')

  file_to_remove = r'/Mask_RCNN/videos/' + file_name
  # Removes the File if it already exists
  # !rm $file_to_remove
  try:
    shutil.rmtree(file_to_remove)
  except:
    display(f'ERROR removing the video file to analyze.')

  capture.release()

  if verbose == True:
    display(f'The Left Position just prior to drawing the Bell_Guards is:')
    display(Left_Position)
    display(f'The Right Position just prior to drawing the Bell_Guards is:')
    display(Right_Position)

  t5 = time.time()
  #Draws the Boxes on the image frame and determines scoring lights turned on
  [left_light_comparison, right_light_comparison] = draw_Bell_Guard_Position(Left_Position, Right_Position, Scoring_Box_Position, \
    scoring_box_size_average, Left_Torso_Position, Right_Torso_Position, frame_count, Tracking_Bounding_Boxes, \
    video_filename, width, height, engarde_length, keypoints, score_box_empty, camera_steady, camera_motion_threshold)
  t6 = time.time()
  display(f'Time elapsed while drawing the Bell_Guard positions is {hms_string(t6 - t5)}.')

  if camera_motion_compensate == True and score_box_empty == False:
    #Adjusts the Bellguard Position Based on the Camera motion as determined by the Score_Box Position
    Left_Position = camera_motion_adjustment(Left_Position, Scoring_Box_Position)
    Right_Position = camera_motion_adjustment(Right_Position, Scoring_Box_Position)

  t7 = time.time()
  display(f'Time elapsed for the camera motion adjustment is {hms_string(t7 - t6)}.')

  if camera_motion_compensate == True and score_box_empty == False:
    #Adjusts Left and Right Position for convenient visualization
    [Left_Position, Right_Position] = position_down_scale(Left_Position, Right_Position, width, height)

  t8 = time.time()
  #Creates a vector representing the clip, format [left_x, right_x, left_lights, right_lights]
  clip_vector = clip_vector_generator(Left_Position, Right_Position, left_light_comparison, right_light_comparison, clip_vector_previous, width)

  if smooth_video_clip == True:
    #Smoothes the Clip using Savitzky–Golay filter
    clip_vector = smooth_clip_vector(clip_vector, engarde_length)

  t9 = time.time()
  display(f'Time elapsed to generate the clip vector is {hms_string(t9 - t8)}.')

  if verbose == True:
    display(f'The final clip vector is:')
    display(clip_vector)

  return (bbox, frame_count, width, height, clip_vector_previous, fencer_data, keypoints, clip_vector, fps)

def prepare_video_frames(img_directory, video_title):

  ROOT_DIR = '/content/Mask_RCNN/'
  VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
  VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, img_directory)
  images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
  # Sort the images by integer index
  images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

  name = str(iterator) + '.' + video_title + '.mp4'
  outvid = os.path.join(VIDEO_DIR, name)

  return (outvid, images)

def downsample_fps(a,b):
  # Adjusts the elements of a larger set a to fit into the length of set b

  c = []
  remainder = 0
  for i in range(len(b)):
    c_temp = []
    if verbose == True:
      display(f'The lower range is {math.ceil(len(a)/len(b)*(i+1)-1-remainder)} and the upper range is {math.floor(len(a)/len(b)*(i+1))}.')
    for j in range(math.ceil(len(a)/len(b)*(i)-remainder),math.floor(len(a)/len(b)*(i+1))):
      c_temp.append(a[j])
      if verbose == True:
        display(f'i,j = {i},{j} and c_temp = {c_temp}')
    remainder = (len(a)/len(b))*(i+1) - int(len(a)/len(b)*(i+1))
    c.append(round(sum(c_temp)/len(c_temp)))
    if verbose == True:
      display(f'The remainder at i = {i} and j = {j} is {remainder} and c is {c}.')

  return (c)

def load_clip(folder, clip_number, max_length):
  if folder == 'Left' or folder == 'left' or folder == 'Left_Touch':
    folder = 0
  if folder == 'Right' or folder == 'right'or folder == 'Right_Touch':
    folder = 1
  if folder == 'Simul' or folder == 'simul'or folder == 'Simul':
    folder = 2

  touch_folder = ['Left_Touch', 'Right_Touch', 'Simul']

  i = folder

  file = 'clip_vector_acceleration_np' + str(clip_number) + '.csv'
  path = r'/content/drive/My Drive/projects/fencing/Fencing Clips/' + touch_folder[i] + '/' + touch_folder[i] + '_Vector_Clips_Acceleration/'

  vector_data = pd.read_csv(os.path.join(path, file), header=None)
  clip_vector = vector_data.to_numpy(dtype = np.float32)

  display(os.path.join(path, file))

  # Pads the clip_vector to 103
  # If the clip is greater than Max Length, it is truncated
  if len(clip_vector) > max_length:
    clip_vector = clip_vector[len(clip_vector) - max_length:]
  padding = np.array([0,0,0,0])
  for k in range(max_length - (len(clip_vector))):
    clip_vector = np.vstack((clip_vector, padding))

  #Normalizes the Value by 31
  max_value = 31
  for i in range(len(clip_vector)):
    for j in range(2):
      if clip_vector[i][j] < max_value:
        clip_vector[i][j] = clip_vector[i][j] * (1/max_value)
      else:
        #Preserves the sign of the value
        clip_vector[i][j] = clip_vector[i][j]/(abs(clip_vector[i][j]))

  # Removes the First 15 frames to minimize engarde positioning
  clip_vector = clip_vector[15:]

  # Sets Clip_Vector to Zero if Light is on
  for j in range(len(clip_vector)):
    if clip_vector[j][2] == 1:
      clip_vector[j][0] = 0
    if clip_vector[j][3] == 1:
      clip_vector[j][1] = 0 

  clip_vector = clip_vector.reshape(1,clip_vector_length,4)

  return (clip_vector)

def create_folder_hierarchy(file_name):

  # Creates File Path in Google Drive
  # !mkdir -p '/content/drive/My Drive/projects/fencing/Fencing Clips/Left_Touch/Left_Touch_Vector_Clips'
  try:
    os.makedirs('/content/drive/My Drive/projects/fencing/Fencing Clips/Left_Touch/Left_Touch_Vector_Clips')
  except:
    display(f'ERROR creating the Left_Touch_Vector_Clips')
  # %cd '/content/drive/My Drive/projects/fencing/Fencing Clips/Left_Touch/'
  os.chdir('/content/drive/My Drive/projects/fencing/Fencing Clips/Left_Touch/')
  # !mkdir Left_Touch_Vector_Clips_Speed
  try:
    os.mkdir('Left_Touch_Vector_Clips_Speed')
  except:
    display(f'ERROR creating the Left_Touch_Vector_Clips_Speed')
  # !mkdir Left_Touch_Vector_Clips_Acceleration
  try:
    os.mkdir('Left_Touch_Vector_Clips_Acceleration')
  except:
    display(f'ERROR creating the Left_Touch_Vector_Clips_Speed')

  # Copies Video File to Left_Touch Folder
  file_name = r'/content/drive/My Drive/' + file_name
  destination = r'/content/drive/My Drive/projects/fencing/Fencing Clips/Left_Touch/'
  # !cp {file_name} {destination}
  shutil.copy(file_name, destination)

  return
