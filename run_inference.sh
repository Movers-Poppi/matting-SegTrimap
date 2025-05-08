python inference.py \
  --ckpt_path ./weights/p3m_train_crossentropy.pth \
  --data_root ./datasets/AIM-500 \
  --save_dir ./inference/ \
  --infer_img_size 448 \
  --batch_size 20