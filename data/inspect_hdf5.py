import h5py
import argparse

def list_datasets(h5_object, path=''):
    if isinstance(h5_object, h5py.Dataset):
        print(f"{path}: Shape: {h5_object.shape}, Datatype: {h5_object.dtype}")
    elif isinstance(h5_object, h5py.Group):
        for key in h5_object.keys():
            list_datasets(h5_object[key], f"{path}/{key}")

def main():
    parser = argparse.ArgumentParser(description="List shape and datatype of all datasets in an HDF5 file.")
    parser.add_argument("filename", help="Path to the HDF5 file.")
    args = parser.parse_args()

    with h5py.File(args.filename, 'r') as h5_file:
        print(f"Listing datasets in {args.filename}:")
        list_datasets(h5_file)

if __name__ == "__main__":
    main()

