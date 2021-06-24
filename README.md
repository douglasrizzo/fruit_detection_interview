# 2021 ML and Com. Vis. Engineer, Problem 2

- Training and evaluation report available [here](https://wandb.ai/tetamusha/fruit_detection_torchvision/reports/Detecting-oranges-with-TorchVision--Vmlldzo4MDAyNzM)
- The Jupyter Notebook + PDF contain a dataset study, which allowed me to discover the number of object classes, object annotations, bounding box aspect ratios and size in pixels.
- The code is basically aa usage example of the code in [this other repository](https://github.com/douglasrizzo/fruit_detection).

The best model trained for the problem used the following arguments:

    python train.py --batch_size 4 --num_workers 4 --backbone mobilenet --detector fasterrcnn --augmentations imgaug --epochs 350 --lr_initial 0.0005 --lr_final 0.00001 --lr_updates 50 --eval_size 0.1 --img_min_size 1920 --img_max_size 1920
