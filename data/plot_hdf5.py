import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from turbojpeg import decompress
from PIL import Image
import io

def decode_jpeg(image): 
    return decompress(image)

def decode_png(image):
    return Image.open(io.BytesIO(image))

def plot_dataset(h5_file, sequence_index, image_index = None):
    plt.clf()

    if '/sequence_indices' not in h5_file:
        image_index = sequence_index

    if image_index is None:
        sequence_start, sequence_lenght = h5_file['/sequence_indices'][sequence_index]
        sequence_end = sequence_start + sequence_lenght 
        image_index = random.randint(sequence_start, sequence_end)
    print(f'Plotting sequence {sequence_index} image {image_index}')
        
    mask_index_start = 0
    mask_index_end = 0
    if 'image_instance_indices' in h5_file and len(h5_file['/image_instance_indices']) > 0:
        mask_index_start, mask_index_length = h5_file['/image_instance_indices'][image_index]
        mask_index_end = mask_index_start + mask_index_length
    
    mask = None
    if '/instance_masks' in h5_file and len(h5_file['/instance_masks']) > 0:
        mask = h5_file['/instance_masks'][mask_index_start:mask_index_end].sum(axis=0)[0]
    
    foreground_mask = None
    if '/foreground_mask' in h5_file and len(h5_file['/foreground_mask']) > 0:
        foreground_mask = h5_file['/foreground_mask'][image_index][0]
    
    plt.subplot(2, 3, 1)
    if h5_file['/rgb_images'][image_index].dtype == np.uint8:
        print('Decoding JPEG')
        plt.imshow(decode_jpeg(h5_file['/rgb_images'][image_index]))
    else:
        plt.imshow(h5_file['/rgb_images'][image_index].transpose((1, 2, 0))[:,:,::-1])
    if '/instance_mask_bboxes' in h5_file and len(h5_file['/instance_mask_bboxes']) > 0:
        for bbox in h5_file['/instance_mask_bboxes'][mask_index_start:mask_index_end]:
            plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor='r', facecolor='none'))
    plt.title('RGB Image')
    
    if '/depth_images' in h5_file and len(h5_file['/depth_images']) > 0:
        plt.subplot(2, 3, 2)
        if h5_file['/depth_images'][image_index].dtype == np.uint8:
            plt.imshow(decode_png(h5_file['/depth_images'][image_index]))
        else:
            plt.imshow(h5_file['/depth_images'][image_index][0], cmap='gray')
        plt.title('Depth Image')

    if '/raw_depth' in h5_file and len(h5_file['/raw_depth']) > 0:
        plt.subplot(2, 3, 3)
        plt.imshow(h5_file['/raw_depth'][image_index][0], cmap='gray')
        plt.title('Raw Depth')

    if '/instance_depth_unoccluded' in h5_file and len(h5_file['/instance_depth_unoccluded']) > 0:
        plt.subplot(2, 3, 3)
        if h5_file['/instance_depth_unoccluded'][image_index].dtype == np.uint8:
            plt.imshow(decode_png(h5_file['/instance_depth_unoccluded'][image_index]))
        else:
            plt.imshow(h5_file['/instance_depth_unoccluded'][image_index][0], cmap='gray')
        plt.title('Raw Depth')
    
    if '/forward_flow' in h5_file and len(h5_file['/forward_flow']) > 0:
        plt.subplot(2, 3, 4)
        plt.imshow(np.linalg.norm(h5_file['/forward_flow'][image_index], axis=0), cmap='gray')
        plt.title('Forward Flow')

    if '/instance_rgb_unoccluded' in h5_file and len(h5_file['/instance_rgb_unoccluded']) > 0:
        plt.subplot(2, 3, 4)
        if h5_file['/instance_rgb_unoccluded'][image_index].dtype == np.uint8:
            plt.imshow(decode_png(h5_file['/instance_rgb_unoccluded'][image_index]))
        else:
            plt.imshow(h5_file['/instance_rgb_unoccluded'][image_index][0], cmap='gray')
        plt.title('Raw RGB')

    if '/backward_flow' in h5_file and len(h5_file['/backward_flow']) > 0:
        plt.subplot(2, 3, 5)
        plt.imshow(np.linalg.norm(h5_file['/backward_flow'][image_index], axis=0), cmap='gray')
        plt.title('Backward Flow')

    if '/instance_masks_unoccluded' in h5_file and len(h5_file['/instance_masks_unoccluded']) > 0:
        plt.subplot(2, 3, 5)
        plt.imshow(h5_file['/instance_masks_unoccluded'][image_index][0] / 255, cmap='gray')
        plt.title('Raw Mask')

    plt.subplot(2, 3, 6)
    if random.random() < 0.5:
        if foreground_mask is not None:
            plt.imshow(foreground_mask, cmap='gray')
        if mask is not None:
            for bbox in h5_file['/instance_mask_bboxes'][mask_index_start:mask_index_end]:
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor='r', facecolor='none'))
        plt.title('Foreground Mask')
    else:
        if mask is not None:
            plt.imshow(mask, cmap='gray')
            for bbox in h5_file['/instance_mask_bboxes'][mask_index_start:mask_index_end]:
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor='r', facecolor='none'))
        plt.title('Instance Mask and BBox')

    plt.tight_layout()
    plt.gcf().canvas.draw()

sequence_index = 0

def on_key(event, h5_file, num_sequences, num_images = None):
    global sequence_index
    img_index = None
    if event.key == 'n':
        #sequence_index = random.randint(0, num_sequences - 1)
        #img_index = random.randint(0, num_images - 1) if num_images is not None else None

        sequence_index = sequence_index + 1
        plot_dataset(h5_file, sequence_index, sequence_index)

    if event.key == 'r':
        plot_dataset(h5_file, sequence_index, sequence_index)

def main():
    parser = argparse.ArgumentParser(description="Plot random image data for a given HDF5 file.")
    parser.add_argument("filename", help="Path to the HDF5 file.")
    args = parser.parse_args()

    with h5py.File(args.filename, 'r') as h5_file:
        num_sequences = len(h5_file['/sequence_indices']) if '/sequence_indices' in h5_file else len(h5_file['/rgb_images'])
        image_index = num_images = None
        if num_sequences == 0:
            num_sequences = len(h5_file['/rgb_images'])
            num_images = len(h5_file['/rgb_images'])
            image_index = random.randint(0, num_sequences - 1)

        fig = plt.figure(figsize=(15, 10))  # Create the figure upfront
        fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, h5_file, num_sequences, num_images))  # Connect the event handler
        plot_dataset(h5_file, 0, 0)
        plt.show()  # Keep show() at the end to keep the figure open

if __name__ == "__main__":
    main()
