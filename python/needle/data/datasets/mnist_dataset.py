from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as image_gzip_in:
            image_magic_num = int.from_bytes(image_gzip_in.read(4), byteorder="big")
            self.image_num = int.from_bytes(image_gzip_in.read(4), byteorder="big")
            image_row = int.from_bytes(image_gzip_in.read(4), byteorder="big")
            image_col = int.from_bytes(image_gzip_in.read(4), byteorder="big")
            image_buffer = image_gzip_in.read()

        with gzip.open(label_filename, "rb") as lable_gzip_in:
            lable_magic_num = int.from_bytes(lable_gzip_in.read(4), byteorder="big")
            lable_item = int.from_bytes(lable_gzip_in.read(4), byteorder="big")
            lable_buffer = lable_gzip_in.read()

        assert image_magic_num == 0x00000803
        assert lable_magic_num == 0x00000801
        assert self.image_num == lable_item

        self.images = np.array(list(image_buffer), dtype=np.uint8)
        self.labels = np.array(list(lable_buffer), dtype=np.uint8)

        self.images = self.images.astype(np.float32) / 255.0
        self.images = self.images.reshape(self.image_num, image_row, image_col, 1)

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        return self.apply_transforms(img), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.image_num
        ### END YOUR SOLUTION
