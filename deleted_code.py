# normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# true_transformer = T.Compose([
#     T.RandomSizedRectCrop(256, 128),
#     T.RandomHorizontalFlip(),
#     T.ToTensor(),
#     normalizer,
# ])
# false_transformer = transformer = T.Compose([
#     T.RectScale(256, 128),
#     T.ToTensor(),
#     normalizer,
# ])
# pre_dataset_trainTure = Preprocessor(dataset_all, root=dataset_all.images_dir,
#                                      num_samples=16 if dataset_all.is_video else 1,
#                                      transform=true_transformer, is_training=True,
#                                      max_frames=args.max_frame)  # video_frames, image_str, pid, index, videoid
# pre_dataset_trainFalse = Preprocessor(dataset_all, root=dataset_all.images_dir,
#                                       num_samples=16 if dataset_all.is_video else 1,
#                                       transform=false_transformer, is_training=True,
#                                       max_frames=args.max_frames)  # video_frames, image_str, pid, index, videoid
# pre_dataset = Preprocessor(dataset_all, root=dataset_all.images_dir, num_samples=16 if dataset_all.is_video else 1,
# #                  transform=transformer, is_training=training, max_frames=self.max_frames),