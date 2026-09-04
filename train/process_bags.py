
import os
import pickle
import argparse
import tqdm
import yaml
import rosbag
from pathlib import Path

# utils
from vint_train.process_data.process_data_utils import (
    filter_backwards,
    get_images_and_odom,
    nav_to_xy_yaw,
    process_locobot_img,
    process_sacson_img,
    process_scand_img,
    process_tartan_img,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "vint_train", "process_data", "process_bags_config.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets", "tartan_drive")

IMAGE_PROCESSORS = {
    "process_tartan_img": process_tartan_img,
    "process_scand_img": process_scand_img,
    "process_locobot_img": process_locobot_img,
    "process_sacson_img": process_sacson_img,
}

ODOM_PROCESSORS = {
    "nav_to_xy_yaw": nav_to_xy_yaw,
}


def get_processor(registry, name: str, kind: str):
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown {kind} processor '{name}'. Available: {available}") from exc


def main(args: argparse.Namespace):

    # load the config file
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.dataset_name not in config:
        available = ", ".join(sorted(config))
        raise ValueError(f"Unknown dataset '{args.dataset_name}'. Available: {available}")
    dataset_config = config[args.dataset_name]

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # iterate recurisively through all the folders and get the path of files with .bag extension in the args.input_dir
    bag_files = []
    for root, _dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
    bag_files.sort()
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    # processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            b = rosbag.Bag(bag_path)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # name is that folders separated by _ and then the last part of the path
        bag_path_obj = Path(bag_path)
        traj_name = f"{bag_path_obj.parent.name}_{bag_path_obj.stem}"

        try:
            # load the bag data
            bag_img_data, bag_traj_data = get_images_and_odom(
                b,
                dataset_config["imtopics"],
                dataset_config["odomtopics"],
                get_processor(IMAGE_PROCESSORS, dataset_config["img_process_func"], "image"),
                get_processor(ODOM_PROCESSORS, dataset_config["odom_process_func"], "odometry"),
                rate=args.sample_rate,
                ang_offset=dataset_config["ang_offset"],
            )
        finally:
            b.close()

  
        if bag_img_data is None or bag_traj_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            continue
        # remove backwards movement
        cut_trajs = filter_backwards(bag_img_data, bag_traj_data)

        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            # make a folder for the traj
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)
            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)
            # save the image data to disk
            for i, img in enumerate(img_data_i):
                img.save(os.path.join(traj_folder_i, f"{i}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="tartan_drive",
    )
    parser.add_argument(
        "--config-path",
        "-c",
        default=DEFAULT_CONFIG_PATH,
        type=str,
        help="path to process_bags_config.yaml",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        type=str,
        help="path for processed dataset",
    )
    # number of trajs to process
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    # all caps for the dataset name
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
