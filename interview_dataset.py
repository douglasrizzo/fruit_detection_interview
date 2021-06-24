import json
import os
import zipfile
from glob import glob
from typing import Union

from imgaug.augmenters.meta import Augmenter

import torchvision_detection_utils.transforms as T
from orchard_datasets import AugmentedDataset, DirectlyDownloadableObject


class InterviewDataset(AugmentedDataset, DirectlyDownloadableObject):
  @staticmethod
  def _read_annotation_file(file_path: str, *args):
    with open(file_path) as f:
      json_data = json.load(f)

      image_height = json_data["imageHeight"]
      image_width = json_data["imageWidth"]

      obj_labels = []
      obj_coords = []

      for i in json_data["shapes"]:
        label = i["label"]

        # these classes seem to not be required, as per the exercise
        if "Anomalia" in label:
          continue

        xmin = i["points"][0][0]
        xmax = i["points"][1][0]
        ymin = i["points"][1][1]
        ymax = i["points"][0][1]

        # sanity checks
        if ymin > image_height or ymax > image_height:
          raise ValueError("object y coordinate larger than image height")
        if xmin > image_width or xmax > image_width:
          raise ValueError("object x coordinate larger than image width")

        if ymin > ymax:
          raise ValueError("ymin > ymax")
        if xmin > xmax:
          raise ValueError("xmin > xmax")

        obj_labels.append(label)
        # the augmentations performed by imgaug work with tuples of coordinates or numpy arrays of size (N, ),
        # so here I chose tuples which are simpler data structures
        obj_coords.append((xmin, ymin, xmax, ymax))

    return obj_labels, obj_coords

  def _maybe_download_image_dataset(self):
    """Downloads the dataset if it does not exist on disk"""
    # check for the existence of the dataset in the root directory chosen by the user
    if not os.path.exists(self.root) or not os.path.isdir(self.root):
      # check if there are no zips with the same prefix and suffix in the /tmp
      zipname = self._look_in_tmp_or_download_file(
        "problem_2",
        ".zip",
        "https://no_share_private_link.com/my_proprietary_dataset.zip",
      )

      # uncompress only JPGs and JSONs
      with zipfile.ZipFile(zipname, "r") as zip_file:
        if not os.path.isdir(self.root):
          os.mkdir(self.root)

        fileNames = zip_file.namelist()
        for fileName in fileNames:
          if fileName.endswith(("json", self.img_extension)):
            content = zip_file.open(fileName).read()
            image_file = open(os.path.join(self.root, os.path.basename(fileName)), "wb")
            image_file.write(content)
            image_file.close()

  def __init__(
    self,
    root: str,
    transforms: Union[Augmenter, T.Compose] = None,
    count_classes_from_zero: bool = False,
  ):
    """The dataset used in the problem

    :param root: the directory where the images and annotations are located (or will be extracted to, in case they aren't there)
    :type root: str
    :param transforms: an augmenter to apply to the data during loading. It can either be an :py.class:`Augmenter` or a :py.class:`Compose`, defaults to None
    :type transforms: Union[Augmenter, T.Compose], optional
    :param count_classes_from_zero: whether object classes should be numbered starting from 0, otherwise will start from 1
    :type count_classes_from_zero: bool, defaults to False
    :raises ValueError: if the number of image files and annotation files is different
    :raises ValueError: if the names of the image and annotation files are different
    """
    super().__init__(root, transforms, "jpg")
    # load images and annotations, sort them to ensure both collections match
    self.imgs = list(sorted(glob(os.path.join(root, "*." + self.img_extension))))
    annotation_file_paths = list(sorted(glob(os.path.join(root, "*.json"))))

    super().init_dataset(root, annotation_file_paths, count_classes_from_zero)
