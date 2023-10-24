from torch.utils import data
import numpy as np
import cv2
import h5py
import os

__author__ = "Manuel Traub"

class MaskBBoxNumpy:
    def __init__(self, size):
        height, width = size
        x_range = np.linspace(0, width, width)
        y_range = np.linspace(0, height, height)

        x_coords, y_coords = np.meshgrid(x_range, y_range)

        self.x_coords = x_coords[None, None, :, :]
        self.y_coords = y_coords[None, None, :, :]

    def compute(self, mask):
        mask = (mask > 0.75).astype(np.float32)

        x_masked = self.x_coords * mask
        y_masked = self.y_coords * mask

        x_min = np.min(np.where(x_masked > 0, x_masked, np.inf), axis=(2, 3))
        y_min = np.min(np.where(y_masked > 0, y_masked, np.inf), axis=(2, 3))
        x_max = np.max(np.where(x_masked > 0, x_masked, -np.inf), axis=(2, 3))
        y_max = np.max(np.where(y_masked > 0, y_masked, -np.inf), axis=(2, 3))

        bbox = np.stack([x_min, y_min, x_max, y_max], axis=2).squeeze(0)
        return bbox

def compress_image(image, format='.jpg'):

    # Encode image to the specified format using OpenCV
    if format == '.jpg':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    is_success, buffer = cv2.imencode(format, image)
    if is_success:
        return np.array(buffer)
    else:
        raise Exception("Failed to compress image")

class HDF5Dataset:
    """
    HDF5Dataset: A class for managing datasets in the HDF5 format, specifically tailored for computer vision tasks.

    Attributes:
        mask_bboxes (MaskBBoxNumpy): An object for handling bounding box calculations for masks.
        hdf5_file (h5py.File): The HDF5 file handle.

    Methods:
        close(): Flushes any buffered data to disk and closes the HDF5 file.
        __getitem__(index: str): Provides direct access to any dataset within the HDF5 file.
        append_data(index: str, item: np.ndarray): Appends data to a specified dataset.
        append_image(index: str, item: np.ndarray): Appends a single image to a specified dataset.
        append_sequence(rgb_images: np.ndarray, depth_images: np.ndarray, foreground_masks: np.ndarray, instance_masks: List[np.ndarray]):
            Appends a sequence of images and their corresponding masks.

    Datasets:
        rgb_images: Stores RGB images. dtype: Variable-length sequence of uint8.
        depth_images: Stores depth images. dtype: Variable-length sequence of uint8.
        foreground_mask: Stores binary foreground masks. dtype: uint8.
        instance_masks: Stores instance segmentation masks. dtype: uint8.
        sequence_indices: Stores indices indicating the start and end of image sequences. dtype: np.compat.long.
        image_instance_indices: Stores indices indicating the start and end of instances within images. dtype: np.compat.long.
        instance_masks_images: Stores indices linking instance masks to RGB images. dtype: np.compat.long.
        instance_mask_bboxes: Stores bounding box coordinates for instance masks. dtype: uint8.

    Example:
        >>> hdf5_dataset = HDF5Dataset(root_path="/data", dataset_name="my_dataset", type="train", size=(256, 256))
        >>> hdf5_dataset.append_sequence(rgb_images, depth_images, foreground_masks, instance_masks)
        >>> hdf5_dataset.close()
    """

    def __init__(self, root_path: str, dataset_name: str, type: str, size: tuple):

        self.mask_bboxes = MaskBBoxNumpy(size)
        
        instance_counter = 1
        hdf5_file_path = os.path.join(root_path, f'{dataset_name}-{type}-{size[0]}x{size[1]}-v1.hdf5')
        while os.path.exists(hdf5_file_path):
            instance_counter += 1
            hdf5_file_path = os.path.join(root_path, f'{dataset_name}-{type}-{size[0]}x{size[1]}-v{instance_counter}.hdf5')

        data_path = os.path.join(root_path, dataset_name, type)

        # setup the hdf5 file
        hdf5_file = h5py.File(hdf5_file_path, "w")

        # Create datasets for rgb_images, depth_images, and instance_masks
        hdf5_file.create_dataset(
            "rgb_images",   
            (0, ),
            maxshape=(None, ),
            dtype=h5py.vlen_dtype(np.dtype('uint8')),
        )
        hdf5_file.create_dataset(
            "depth_images", 
            (0, ),
            maxshape=(None, ),
            dtype=h5py.vlen_dtype(np.dtype('uint8')),
        )
        hdf5_file.create_dataset(
            "foreground_mask",
            (0, 1, size[0], size[1]),
            maxshape=(None, 1, size[0], size[1]),
            dtype=np.uint8,
            compression='gzip',
            compression_opts=5,
            chunks=(1, 1, size[0], size[1])
        )
        hdf5_file.create_dataset(
            "instance_masks", 
            (0, 1, size[0], size[1]), 
            maxshape=(None, 1, size[0], size[1]), 
            dtype=np.uint8, 
            compression='gzip',
            compression_opts=5,
            chunks=(1, 1, size[0], size[1])
        )
        hdf5_file.create_dataset(
            "sequence_indices",
            (0, 2), # start index, number of images
            maxshape=(None, 2),
            dtype=np.compat.long,
            compression='gzip',
            compression_opts=5,
        )
        hdf5_file.create_dataset(
            "image_instance_indices",
            (0, 2), # start index, number of instances
            maxshape=(None, 2),
            dtype=np.compat.long,
            compression='gzip',
            compression_opts=5,
        )
        hdf5_file.create_dataset(
            "instance_masks_images", 
            (0, 1), 
            maxshape=(None, 1),
            compression='gzip',
            compression_opts=5,
            dtype=np.compat.long,
        )
        hdf5_file.create_dataset(
            "instance_mask_bboxes", 
            (0, 4), 
            maxshape=(None, 4), 
            compression='gzip',
            compression_opts=5,
            dtype=np.uint8, 
        )

        # Create a metadata group and set the attributes
        metadata_grp = hdf5_file.create_group("metadata")
        metadata_grp.attrs["dataset_name"] = dataset_name
        metadata_grp.attrs["type"] = type

        self.hdf5_file = hdf5_file

    def close(self):
        self.hdf5_file.flush()
        self.hdf5_file.close()

    def __getitem__(self, index):
        return self.hdf5_file[index]

    def append_data(self, index, item):
        self[index].resize((self[index].shape[0] + item.shape[0], *item.shape[1:]))
        self[index][-item.shape[0]:] = item

    def append_image(self, index, item):
        self[index].resize((self[index].shape[0] + 1,))
        self[index][-1] = item

    def append_sequence(self, rgb_images, depth_images, foreground_masks, instance_masks):
        """
        Appends a sequence of RGB images, depth images, foreground masks, and instance masks to the HDF5 dataset.

        This method is responsible for both adding new image and mask data to the HDF5 datasets and updating the 
        associated metadata. It takes special care to maintain the integrity of sequence and instance information, 
        updating indices and bounding boxes as needed.


        Parameters:
            rgb_images (np.ndarray): An array of RGB images to be appended. Shape should be (N, H, W, 3).
            depth_images (np.ndarray, optional): An array of depth images to be appended. Shape should be (N, H, W).
            foreground_masks (np.ndarray, optional): An array of binary foreground masks to be appended. Shape should be (N, H, W).
            instance_masks (List[np.ndarray], optional): A list of arrays containing instance masks for each image. Each array shape should be (M_i, H, W).

        Raises:
            AssertionError: If the number of rgb_images, depth_images, foreground_masks, and instance_masks are not aligned.

        Side Effects:
            - Resizes and appends data to 'rgb_images', 'depth_images', 'foreground_mask', 'instance_masks' datasets.
            - Updates 'sequence_indices' and 'image_instance_indices' to reflect the new sequence and instance information.
            - Computes and stores bounding box information for instance masks using `MaskBBoxNumpy`.

        Example:
            >>> self.append_sequence(rgb_images=np.random.rand(10, 256, 256, 3),
                                     depth_images=np.random.rand(10, 256, 256),
                                     foreground_masks=np.random.randint(2, size=(10, 256, 256)),
                                     instance_masks=[np.random.randint(2, size=(m, 256, 256)) for m in range(1, 11)])

        Note:
            - The depth images and masks should be normalized before passing. They will be scaled and converted to uint8 internally.
        """
        assert depth_images is None or len(rgb_images) == len(depth_images)
        assert foreground_masks is None or len(rgb_images) == len(foreground_masks)
        assert instance_masks is None or len(rgb_images) == len(instance_masks)

        if foreground_masks is not None:
            self.append_data('foreground_mask', np.expand_dims((foreground_masks * 255).astype(np.uint8), axis=1))

        self.append_data('sequence_indices', np.array([[self['rgb_images'].shape[0], rgb_images.shape[0]]]))

        for i in range(len(rgb_images)):
            
            if instance_masks is not None:
                self.append_data('instance_mask_bboxes', self.mask_bboxes.compute(instance_masks[i]))
                self.append_data('instance_masks_images', np.ones((instance_masks[i].shape[0],1)) * self['rgb_images'].shape[0])
                self.append_data('image_instance_indices', np.array([[self['instance_masks'].shape[0], instance_masks[i].shape[0]]]))
                self.append_data('instance_masks', np.expand_dims(instance_masks[i], axis=1)*255)

            self.append_image('rgb_images', compress_image(rgb_images[i]*255, '.jpg'))

            if depth_images is not None:
                self.append_image('depth_images', compress_image(depth_images[i]*255, '.png'))
        
        


if __name__ == "__main__":

    from tqdm import tqdm

    dataset = HDF5Dataset(
        root_path = './', 
        dataset_name = 'example-dataset', 
        type = 'test',
        size = (256, 256),
    )

    def generate_moving_circles_sequence(resolution, sequence_length):
        width, height = resolution

        # Initialize circle properties
        num_circles = np.random.randint(1, 6)
        initial_positions = np.random.randint(0, min(width, height), size=(num_circles, 2))
        colors = np.random.rand(num_circles, 3)
        radii = np.random.randint(5, 50, size=num_circles)

        # Generate random slopes and y-intercepts for lines
        slopes = np.random.uniform(-1, 1, size=num_circles)
        intercepts = np.random.randint(0, height, size=num_circles)

        # Initialize sequence of frames, depth images, foreground masks, and instance masks
        frames = np.zeros((sequence_length, height, width, 3), dtype=np.float32)
        depth_images = np.zeros((sequence_length, height, width), dtype=np.float32)
        fg_masks = np.zeros((sequence_length, height, width), dtype=np.float32)
        instance_masks = np.zeros((sequence_length, num_circles, height, width), dtype=np.float32)

        # Generate each frame in the sequence
        for t in range(sequence_length):
            fg_masks[t] = np.zeros((height, width), dtype=np.uint8)
            depth_images[t] = np.zeros((height, width), dtype=np.float32)

            # Calculate new positions along the line: y = mx + c
            dx = np.random.randint(1, 5, size=num_circles)  # Random step in x direction
            initial_positions[:, 0] += dx
            initial_positions[:, 1] = slopes * initial_positions[:, 0] + intercepts

            # Loop boundaries
            initial_positions[:, 0] %= width
            initial_positions[:, 1] %= height

            # Draw circles
            for i in range(num_circles):
                cv2.circle(frames[t], tuple(initial_positions[i]), radii[i], colors[i].tolist(), -1)

                # Update depth image
                cv2.circle(depth_images[t], tuple(initial_positions[i]), radii[i], (float(i+1) / num_circles), -1)

                # Update foreground mask
                cv2.circle(fg_masks[t], tuple(initial_positions[i]), radii[i], 1, -1)

                # Update instance mask
                instance_masks[t, i] = cv2.circle(np.zeros((height, width), dtype=np.uint8), tuple(initial_positions[i]), radii[i], 1, -1)

        # return instance_masks as a list of masks
        instance_masks = [instance_masks[i] for i in range(len(instance_masks))]

        return frames, depth_images, fg_masks, instance_masks

    for i in tqdm(range(100)):
        dataset.append_sequence(*generate_moving_circles_sequence((256, 256), np.random.randint(15, 25)))

    dataset.close()
