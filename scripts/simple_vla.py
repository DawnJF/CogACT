from PIL import Image
from vla import load_vla
import torch
import os
import numpy as np
from functools import partial
from pathlib import Path

VISION_IMAGE_SIZE = 224


class SimpleVLA:
    def __init__(self, model_path):
        self.unnorm_key = "libero_spatial_no_noops"
        model_path = os.path.expanduser(model_path)

        # Load the model
        self.vla = load_vla(
            model_id_or_path=model_path,
            load_for_training=False,
            action_model_type="DiT-B",
            future_action_window_size=15,
        )
        self.vla.to("cuda:0").eval()

    def compose_input(self, img_scene, img_hand_left, img_hand_right, instruction, debug=True):
        img_scene = Image.fromarray(img_scene)
        img_hand_left = Image.fromarray(img_hand_left)
        img_hand_right = Image.fromarray(img_hand_right)
        image_all = {
            "scene": img_scene,
            "left": img_hand_left,
            "right": img_hand_right,
        }

        if debug:
            # images for final input
            img_scene.save(Path("./imgs_debug") / "eval_scene.png")
            img_hand_left.save(Path("./imgs_debug") / "eval_img_hand_left.png")
            img_hand_right.save(Path("./imgs_debug") / "eval_img_hand_right.png")
        return image_all

    def generate_action(self, instruction, image_all):
        with torch.inference_mode():
            actions, _ = self.vla.predict_action(
                image_all,
                instruction,
                unnorm_key=self.unnorm_key,
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
            )

        # np.ndarray and its shape is (16, 7)
        return actions


if __name__ == "__main__":
    model_path = "/liujinxin/code/CogACT_mjf/logs/libero--image_aug/checkpoints/step-005000-epoch-13-loss=0.0408.pt"
    vla = SimpleVLA(model_path)
    print("VLA model loaded successfully.")
