import argparse
import tqdm
import os
from PIL import Image
from scripts.simple_vla import SimpleVLA

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np


def visualize_episode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="libero_spatial_no_noops", help="name of the dataset to visualize")
    args = parser.parse_args()

    # create TF dataset
    dataset_name = args.dataset_name
    print(f"Visualizing data from dataset: {dataset_name}")

    ### 1. 数据集构建方式一，这种方式要求数据集为公开数据集或者已经注册在tensorflow默认的数据集缓冲保存目录中
    # ds = tfds.load("dummy_data_ur5", split='train')

    ### 2. 数据集构建方式二，这种方式支持数据集为本地数据集, 只需要传入相应的参数即可
    ds = tfds.builder(
        dataset_name,
        data_dir="/liujinxin/code/rlds_dataset_builder/tensorflow_datasets/datasets--openvla--modified_libero_rlds",
    )
    ds = ds.as_dataset(split="train")

    ### 3. 创建一个指定大小的buffer，tensorflow会从这个buffer中读取数据, 该buffer会在每次采样一条数据后随机打乱
    ds = ds.shuffle(100)

    # visualize episodes
    for i, episode in enumerate(ds.take(3)):
        images = []
        hand_images_left = []
        hand_images_right = []
        for step in episode["steps"]:
            images.append(step["observation"]["image"].numpy())
            hand_images_left.append(step["observation"]["hand_image_left"].numpy())
            hand_images_right.append(step["observation"]["hand_image_right"].numpy())


def test_inference():
    model_path = "/liujinxin/code/CogACT_mjf/logs/libero--image_aug/checkpoints/step-005000-epoch-13-loss=0.0408.pt"
    # model_path = "/liujinxin/code/CogACT/models/CogACT-Base/checkpoints/CogACT-Base.pt"

    ds = tfds.builder(
        "libero_spatial_no_noops",
        data_dir="/liujinxin/code/rlds_dataset_builder/tensorflow_datasets/datasets--openvla--modified_libero_rlds",
    )

    ds = ds.as_dataset(split="train")
    ds = ds.shuffle(100)
    # data = ds.take(1)
    # step['observation'].keys()
    # dict_keys(['image', 'joint_state', 'state', 'wrist_image'])

    for i, episode in enumerate(ds.take(3)):
        for step in episode["steps"]:

            image = step["observation"]["image"].numpy()
            language = step["language_instruction"].numpy().decode()

            is_terminal = step["is_terminal"].numpy()
            joint_state = step["observation"]["joint_state"].numpy()
            state = step["observation"]["state"].numpy()
            wrist_image = step["observation"]["wrist_image"].numpy()
            action = step["action"]
            break
        break

    vla = SimpleVLA(model_path)
    print("VLA model loaded successfully.")

    img = Image.fromarray(image)
    actions = vla.generate_action(language, img)

    print(f"Actions shape: {actions.shape}")


if __name__ == "__main__":
    # visualize_episode()
    test_inference()
