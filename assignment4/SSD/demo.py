import time
import pathlib
import torch
from PIL import Image
from vizer.draw import draw_boxes
from ssd.config.defaults import cfg
from ssd.data.datasets import COCODataset, VOCDataset, MNISTDetection
import argparse
import numpy as np
from ssd import torch_utils
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir: pathlib.Path, output_dir: pathlib.Path, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    elif dataset_type == "mnist":
        class_names = MNISTDetection.class_names
    else:
        raise NotImplementedError('Not implemented now.')

    model = SSDDetector(cfg)
    model = torch_utils.to_cuda(model)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    print(image_paths)
    output_dir.mkdir(exist_ok=True, parents=True)

    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    drawn_images = []
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = image_path.name

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(torch_utils.to_cuda(images))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).cpu().numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = "|".join([
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ])
        image_name = image_path.name

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(output_dir / image_name)
        drawn_images.append(drawn_image)
    return drawn_images


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo/voc', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=pathlib.Path(args.images_dir),
             output_dir=pathlib.Path(args.images_dir, "result"),
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
