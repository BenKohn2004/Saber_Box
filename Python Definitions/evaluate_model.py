def evaluate_model(dataset, model, cfg):
  # calculate the mAP for a model on a given dataset
  APs = list()
  for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
		# convert image into one sample
    sample = expand_dims(scaled_image, 0)
		# make prediction
    yhat = model_detection.detect(sample, verbose=0)
		# extract results for first sample
    r = yhat[0]
		# calculate statistics, including AP
    AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
    APs.append(AP)
	# calculate the mean AP across all images
  mAP = mean(APs)
  return mAP