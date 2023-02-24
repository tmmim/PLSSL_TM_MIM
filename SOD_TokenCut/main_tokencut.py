"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import os
import argparse
import random
import pickle
from glob import glob

import torch
import datetime
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import NoGtDataset, ImageDataset, Dataset, bbox_iou
from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt, visualize_mask_np 
from object_discovery import ncut 
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    parser.add_argument(
        "--load_local",
        type=str,
        default=None,
        help="If want to load local pre-train model.",
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="PLSelfSup",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k", 'PLSelfSup', 'PLSelfSup_full'],
        help="Dataset name.",
    )
    
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no_binary_graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
    else:
        if args.dataset == 'PLSelfSup_full':
            image_paths = sorted(glob('/mnt/ssd2/lxy/database/PLSelfSup/train/*'))
            dataset = NoGtDataset(image_paths, args.dataset)
        else:
            dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, device, args.load_local)

    # -------------------------------------------------------------------------------------------------------
    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with TokenCut 
        exp_name = f"TokenCut-{args.arch}"
        if "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

    print(f"Running TokenCut on the dataset {dataset.name} (exp: {exp_name})")

    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)
    OUTPUT_DIR = f"{args.output_dir}/{exp_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.save_feat_dir is not None : 
        os.mkdir(args.save_feat_dir)

    results_path = os.path.join(OUTPUT_DIR, "tower_info.pkl")

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    
    start_time = time.time() 
    ncut_times = []
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        init_image_size = img.shape # (3, H, W)

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        ) # (3, H_pad, W_pad)
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        # Size for transformers
        h_featmap = img.shape[-2] // args.patch_size
        w_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit"  in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])

                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    pred = dino_seg(attentions, (h_featmap, w_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    # Modality selection
                    if args.which_features == "k":
                        #feats = k[:, 1:, :]
                        feats = k
                    elif args.which_features == "q":
                        #feats = q[:, 1:, :]
                        feats = q
                    elif args.which_features == "v":
                        #feats = v[:, 1:, :]
                        feats = v
                        
                    if args.save_feat_dir is not None : 
                        np.save(os.path.join(args.save_feat_dir, im_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')), feats.cpu().numpy())
                        continue

            else:
                raise ValueError("Unknown model.")

        # ------------ Apply TokenCut ------------------------------------------- 
        if not args.dinoseg:
            pred, objects, mask, rles, ct_point, attn_map= ncut(feats, 
                                                                [h_featmap, w_featmap], 
                                                                scales, 
                                                                init_image_size, 
                                                                args.tau, args.eps, 
                                                                im_name=im_name, 
                                                                no_binary_graph=args.no_binary_graph)


            if args.visualize == "pred" and args.no_evaluation :
                image = dataset.load_image(im_name, size_im)
                visualize_predictions(image, pred, OUTPUT_DIR, im_name)
            if args.visualize == "attn" and args.no_evaluation:
                visualize_eigvec(attn_map, OUTPUT_DIR, im_name, [h_featmap, w_featmap], scales)
            if args.visualize == "all" and args.no_evaluation:
                image = dataset.load_image(im_name, size_im)
                visualize_eigvec(attn_map, OUTPUT_DIR, im_name, [h_featmap, w_featmap], scales)
                visualize_mask_np(image, mask, pred, ct_point, OUTPUT_DIR, im_name, scales)

        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        res = {'box': pred, 'rles': rles, 'ct_point': ct_point, 
               'raw_img_hw_size':dataset.raw_img_size, 
               'resize_img_hw_size':(init_image_size[1],init_image_size[2])}
        preds_dict[im_name] = res

        cnt += 1
        # Evaluation
        if not args.no_evaluation:
            # Compare prediction to GT boxes
            ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
            
            if torch.any(ious >= 0.5):
                corloc[im_id] = 1

            # image = dataset.load_image(im_name)
            # visualize_eigvec(attn_map, vis_folder, im_name, [h_featmap, w_featmap], scales)
            # visualize_mask_np(image, mask, pred, ct_point, vis_folder, im_name, scales)

            if cnt % 50 == 0:
                pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

        # save results every 200 imgs
        if cnt % 20 == 0:
            if args.save_predictions:
                with open(results_path, "wb") as f:
                    pickle.dump(preds_dict, f)

    end_time = time.time()
    print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')
    # Save predicted bounding boxes
    if args.save_predictions:
        with open(results_path, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % results_path)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")