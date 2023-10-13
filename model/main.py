import argparse
import sys
import torch as th
import numpy as np
from copy import copy
import os
import torch.distributed as dist


from utils.configuration import Configuration
from model.scripts import evaluation, training
import subprocess
import fcntl
import json
import time

def copy_data(dst, src):
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    subprocess.check_call(['rsync', '-avPh', src, dst])

def update_paths(data, scratch_path):
    lockfile = "/tmp/data_copy.lock"

    # Define datasets
    datasets = ['train', 'val', 'test']

    # Iterate over each dataset
    for dataset in datasets:
        for item in data['data'][dataset]:
            # Extract source path
            source_path = item['path']

            # Replace /mnt/lustre/butz/mtraub38 with scratch_path in the source path
            updated_path = source_path.replace('/mnt/lustre/butz/mtraub38', scratch_path)
            print(f"Updating: {source_path} -> {updated_path}")

            # Try to acquire the lock
            with open(lockfile, 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX) # Blocking lock

                # Perform the copy operation if data does not exist
                if not os.path.exists(updated_path):
                    copy_data(updated_path, source_path)
            
            # Update the path in JSON data
            item['path'] = updated_path

    return data


CFG_PATH = "cfg.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", default=CFG_PATH)
    parser.add_argument("-num-gpus", "--num-gpus", default=1, type=int)
    parser.add_argument("-n", "--n", default=-1, type=int)
    parser.add_argument("-load", "--load", default="", type=str)
    parser.add_argument("-scratch", "--scratch", type=str, default="")
    parser.add_argument("-load-bg", "--load-bg", default="", type=str)
    parser.add_argument("-load-objects", "--load-objects", default="", type=str)
    parser.add_argument("-load-mask", "--load-mask", default="", type=str)
    parser.add_argument("-load-depth", "--load-depth", default="", type=str)
    parser.add_argument("-load-rgb", "--load-rgb", default="", type=str)
    parser.add_argument("-load-embeddings", "--load-embeddings", default="", type=str)
    parser.add_argument("-load-stage1", "--load-stage1", default="", type=str)
    parser.add_argument("-load-proposal", "--load-proposal", default="", type=str)
    parser.add_argument("-dataset-file", "--dataset-file", default="", type=str)
    parser.add_argument("-port", "--port", default=29500, type=int)
    parser.add_argument("-device", "--device", default=0, type=int)
    parser.add_argument("-seed", "--seed", default=1234, type=int)
    parser.add_argument("-scale", "--scale", default=1, type=int)
    parser.add_argument("-testset", "--testset", action="store_true")
    parser.add_argument("-mask-rgbd", "--mask-rgbd", action="store_true")
    parser.add_argument("-input-mask-rgbd", "--input-mask-rgbd", action="store_true")
    parser.add_argument("-add-text", "--add-text", action="store_true")
    parser.add_argument("-preprocess-dataset", "--preprocess-dataset", action="store_true")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-train", "--train", action="store_true")
    mode_group.add_argument("-pretrain-bg", "--pretrain-bg", action="store_true")
    mode_group.add_argument("-pretrain-objects", "--pretrain-objects", action="store_true")
    mode_group.add_argument("-train-proposal", "--train-proposal", action="store_true")
    mode_group.add_argument("-eval", "--eval", action="store_true")
    mode_group.add_argument("-validate", "--validate", action="store_true")
    mode_group.add_argument("-save", "--save", action="store_true")
    mode_group.add_argument("-save-bg", "--save-bg", action="store_true")
    mode_group.add_argument("-save-objects", "--save-objects", action="store_true")
    mode_group.add_argument("-save-masks", "--save-masks", action="store_true")
    mode_group.add_argument("-save-depth", "--save-depth", action="store_true")
    mode_group.add_argument("-save-rgb", "--save-rgb", action="store_true")
    mode_group.add_argument("-save-proposal", "--save-proposal", action="store_true")
    mode_group.add_argument("-export", "--export", action="store_true")
    mode_group.add_argument("-extract-movi-e", "--extract-movi-e", action="store_true")
    parser.add_argument("-objects", "--objects", action="store_true")
    parser.add_argument("-single-gpu", "--single-gpu", action="store_true")
    parser.add_argument("-nice", "--nice", action="store_true")
    parser.add_argument("-export-latent", "--export-latent", action="store_true")
    parser.add_argument("-individual", "--individual", action="store_true")
    parser.add_argument("-shared-memory-size", "--shared-memory-size", default=-1, type=int)
    parser.add_argument("-float32-matmul-precision", "--float32-matmul-precision", default="highest", type=str)

    args = parser.parse_args(sys.argv[1:])

    th.set_float32_matmul_precision(args.float32_matmul_precision)

    if args.shared_memory_size > 0:
        SharedMemoryBytes.get_shared_memory(args.shared_memory_size)

    if not args.objects and not args.nice and not args.individual:
        args.objects = True

    cfg = Configuration(args.cfg)
    if args.scratch != "":
        cfg = update_paths(cfg, args.scratch)

    cfg.single_gpu = args.single_gpu
    
    cfg.seed = args.seed
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    if args.validate:
        cfg.validate = True
    else:
        cfg.validate = False

    if args.device >= 0:
        cfg.device = args.device
        cfg.model_path = f"{cfg.model_path}.device{cfg.device}"

    if args.n >= 0:
        cfg.model_path = f"{cfg.model_path}.run{args.n}"

    num_gpus = th.cuda.device_count()
    
    if cfg.device >= num_gpus:
        cfg.device = num_gpus - 1

    if args.num_gpus > 0:
        num_gpus = args.num_gpus

    if args.save or args.save_bg or args.save_objects or args.single_gpu:
        os.environ['RANK'] = "0"
        os.environ['WORLD_SIZE'] = str(num_gpus)
        os.environ['MASTER_ADDR'] = 'localhost' 
        os.environ['MASTER_PORT'] = str(args.port + args.device)
        dist.init_process_group(backend='nccl', init_method='env://')

    trainset = None
    valset   = None
    testset  = None

    if args.train or args.validate:
        training.train_loci(cfg, args.load if args.load != "" else None, args.load_objects if args.load_objects != "" else None, args.load_bg if args.load_bg != "" else None, args.load_proposal if args.load_proposal != "" else None)
    elif args.pretrain_objects:
        training.train_objects(cfg, args.load if args.load != "" else None, args.load_mask if args.load_mask != "" else None, args.load_depth if args.load_depth != "" else None, args.load_rgb if args.load_rgb != "" else None, args.load_embeddings if args.load_embeddings != "" else None, args.load_stage1 if args.load_stage1 != "" else None)
    elif args.train_proposal:
        training.train_proposal(cfg, args.load if args.load != "" else None, args.load_objects if args.load_objects != "" else None)
    elif args.pretrain_bg:
        training.train_background(cfg, args.load if args.load != "" else None)
    elif args.eval:
        evaluation.evaluate(cfg, num_gpus, testset if args.testset else valset, args.load)
    elif args.save:
        evaluation.save(cfg, testset if args.testset else trainset, args.load, args.load_proposal, [cfg.model.input_size[0] * args.scale, cfg.model.input_size[1] * args.scale], args.objects, args.nice, args.individual, args.add_text)
    elif args.save_bg:
        evaluation.save_bg(cfg, testset if args.testset else trainset, args.load, [cfg.model.input_size[0] * args.scale, cfg.model.input_size[1] * args.scale], args.add_text, args.individual)
    elif args.save_objects:
        evaluation.save_objects(cfg, testset if args.testset else trainset, args.load, [cfg.model.input_size[0] * args.scale, cfg.model.input_size[1] * args.scale], args.add_text, args.individual, args.mask_rgbd, args.export_latent, args.input_mask_rgbd)
    elif args.save_masks:
        evaluation.save_masks(cfg, testset if args.testset else trainset, args.load, [cfg.model.input_size[0] * args.scale, cfg.model.input_size[1] * args.scale], args.add_text, args.individual, args.mask_rgbd, args.export_latent)
    elif args.save_depth:
        evaluation.save_depth(cfg, testset if args.testset else trainset, args.load, [cfg.model.input_size[0] * args.scale, cfg.model.input_size[1] * args.scale], args.add_text, args.individual, args.mask_rgbd, args.export_latent)
    elif args.save_rgb:
        evaluation.save_rgb(cfg, testset if args.testset else trainset, args.load, [cfg.model.input_size[0] * args.scale, cfg.model.input_size[1] * args.scale], args.add_text, args.individual, args.mask_rgbd, args.export_latent)
    elif args.save_proposal:
        evaluation.save_proposal(cfg, args.load, [cfg.model.input_size[0] * args.scale, cfg.model.input_size[1] * args.scale], args.add_text, args.individual, args.mask_rgbd, args.export_latent)
    elif args.export:
        evaluation.export_dataset(cfg, trainset, testset, args.load, f"{args.load}.latent-states")
    elif args.extract_movi_e:
        extract_movi_e.run(cfg, trainset, valset, testset, args.load)
