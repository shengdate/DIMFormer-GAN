from pathlib import Path
import argparse
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


# =============================================================================
# 1. Project paths
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
BUILD_TEST_DIR = SCRIPT_PATH.parent
SRC_DIR = BUILD_TEST_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Make the project root importable when this script is executed directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_test import Mask2_GAN  # noqa: E402


CHECKPOINT_PATH = SRC_DIR / "best_model" / "best_model.pth"
TEST_IMAGE_DIR = PROJECT_ROOT / "test"
OUTPUT_DIR = PROJECT_ROOT / "results"

# Default image used when no --image argument is supplied.
DEFAULT_TEST_IMAGE = "1.png"


# =============================================================================
# 2. Model configuration
# =============================================================================

IN_CHANNELS = 3
NUM_QUERIES = 6
HIDDEN_DIM = 128
MASK_DIM = 128
NUM_CLASSES = 1
NUM_DECODER_LAYERS = 8
IMAGE_SIZE = (512, 512)


# =============================================================================
# 3. Inference configuration
# =============================================================================

SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

# Keep this consistent with the class ordering used during training.
# The original inference code uses class index 1 as foreground.
FOREGROUND_CLASS_INDEX = 1

# Optional instance-mask refinement:
# "none"     : do not refine any instance;
# "all"      : refine all retained instances;
# "selected" : refine only REFINE_INDICES.
REFINE_MODE = "none"
REFINE_INDICES = [0, 1, 2]

# Restore masks from network size to the original image size.
RESTORE_TO_ORIGINAL_SIZE = True


# =============================================================================
# 4. Visualization configuration
# =============================================================================

OVERLAY_ALPHA = 0.55
CONTOUR_THICKNESS = 1

INSTANCE_COLORS = [
    (204, 121, 167),
    (0, 191, 196),
    (240, 228, 66),
    (86, 180, 233),
    (230, 159, 0),
    (0, 158, 115),
    (213, 94, 0),
    (123, 104, 238),
]


# =============================================================================
# 5. Command-line arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run DIMFormer-GAN instance segmentation on one example image "
            "from the repository test directory."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_TEST_IMAGE,
        help=(
            "Image file name in the project test directory. "
            "Examples: --image 1.png, --image 2.png, ..., --image 6.png. "
            "Default: 2.png"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=(
            "Directory used to save prediction results. "
            "Default: <project_root>/results"
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib visualization window after inference.",
    )

    return parser.parse_args()


def validate_example_name(image_name: str) -> str:
    if Path(image_name).name != image_name:
        raise ValueError(
            "Please provide only an image file name from the test directory, "
            "for example: --image 3.png"
        )

    return image_name


# =============================================================================
# 6. Mask post-processing
# =============================================================================

def refine_instance_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2-D instance mask, received shape {mask.shape}."
        )

    binary_mask = np.ascontiguousarray(
        (mask > 0).astype(np.uint8)
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    # No foreground component.
    if num_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    foreground_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(foreground_areas)) + 1

    return (labels == largest_label).astype(np.uint8)


def resize_instance_masks(
    instance_masks: np.ndarray,
    output_hw: tuple[int, int],
) -> np.ndarray:
    if instance_masks.ndim != 3:
        raise ValueError(
            "instance_masks must have shape [N, H, W]. "
            f"Received {instance_masks.shape}."
        )

    num_instances = instance_masks.shape[0]
    output_h, output_w = output_hw

    if num_instances == 0:
        return np.zeros(
            (0, output_h, output_w),
            dtype=np.uint8,
        )

    resized_masks = np.zeros(
        (num_instances, output_h, output_w),
        dtype=np.uint8,
    )

    for instance_index in range(num_instances):
        resized_masks[instance_index] = cv2.resize(
            instance_masks[instance_index],
            (output_w, output_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return resized_masks


# =============================================================================
# 7. Visualization
# =============================================================================

def create_instance_overlay(
    image_rgb: np.ndarray,
    instance_masks: np.ndarray,
    alpha: float = 0.55,
    contour_thickness: int = 1,
) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"image_rgb must be [H, W, 3], received {image_rgb.shape}."
        )

    if instance_masks.ndim != 3:
        raise ValueError(
            f"instance_masks must be [N, H, W], received {instance_masks.shape}."
        )

    image_h, image_w = image_rgb.shape[:2]

    if instance_masks.shape[1:] != (image_h, image_w):
        raise ValueError(
            "The original image and predicted masks have different sizes. "
            f"Image={(image_h, image_w)}, "
            f"masks={instance_masks.shape[1:]}"
        )

    overlay = image_rgb.astype(np.float32).copy()

    # Fill each instance with a different color.
    for instance_index in range(instance_masks.shape[0]):
        mask = instance_masks[instance_index] > 0

        if not np.any(mask):
            continue

        color = np.asarray(
            INSTANCE_COLORS[
                instance_index % len(INSTANCE_COLORS)
            ],
            dtype=np.float32,
        )

        overlay[mask] = (
            (1.0 - alpha) * overlay[mask]
            + alpha * color
        )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    # Draw white instance boundaries.
    overlay_bgr = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR,
    )

    for instance_index in range(instance_masks.shape[0]):
        binary_mask = (
            instance_masks[instance_index] > 0
        ).astype(np.uint8)

        if not np.any(binary_mask):
            continue

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay_bgr,
            contours,
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=contour_thickness,
            lineType=cv2.LINE_AA,
        )

    return cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB,
    )


# =============================================================================
# 8. Main inference procedure
# =============================================================================

def main():
    args = parse_args()

    image_name = validate_example_name(
        args.image
    )

    image_path = TEST_IMAGE_DIR / image_name

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_stem = Path(
        image_name
    ).stem

    overlay_save_path = (
        output_dir
        / f"{image_stem}_instance_overlay.png"
    )

    # -------------------------------------------------------------------------
    # Check repository files
    # -------------------------------------------------------------------------

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Checkpoint file was not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Expected repository layout:\n"
            "  src/best_model/best_model.pth"
        )

    if not TEST_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            "Test-image directory was not found:\n"
            f"{TEST_IMAGE_DIR}\n\n"
            "Expected repository layout:\n"
            "  test/"
        )

    if not image_path.is_file():
        available_images = sorted(
            path.name
            for path in TEST_IMAGE_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                    ".tif",
                    ".tiff",
                }
            )
        )

        if available_images:
            available_text = "\n".join(
                f"  - {name}"
                for name in available_images
            )
        else:
            available_text = "  (no supported image files found)"

        raise FileNotFoundError(
            f"Test image was not found:\n"
            f"{image_path}\n\n"
            "Available images in test/:\n"
            f"{available_text}"
        )

    # -------------------------------------------------------------------------
    # Print runtime information
    # -------------------------------------------------------------------------

    print("=" * 90)
    print("DIMFormer-GAN single-image inference")
    print("=" * 90)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Checkpoint         : {CHECKPOINT_PATH}")
    print(f"Test directory     : {TEST_IMAGE_DIR}")
    print(f"Selected image     : {image_path}")
    print(f"Output directory   : {output_dir}")
    print(f"Overlay result     : {overlay_save_path}")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Build generator
    # -------------------------------------------------------------------------

    generator = Mask2_GAN.Mask2FormerGANGenerator(
        in_channels=IN_CHANNELS,
        num_queries=NUM_QUERIES,
        hidden_dim=HIDDEN_DIM,
        mask_dim=MASK_DIM,
        num_classes=NUM_CLASSES,
        num_decoder_layers=NUM_DECODER_LAYERS,
    ).to(device)

    # -------------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------------

    checkpoint = torch.load(
        str(CHECKPOINT_PATH),
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "G_state_dict" in checkpoint
    ):
        state_dict = checkpoint["G_state_dict"]
    else:
        state_dict = checkpoint

    generator.load_state_dict(
        state_dict
    )

    generator.eval()

    print(
        "Generator checkpoint loaded successfully."
    )

    # -------------------------------------------------------------------------
    # Read image
    # -------------------------------------------------------------------------

    with Image.open(
        image_path
    ) as image:
        original_pil = image.convert("RGB")

    original_width, original_height = (
        original_pil.size
    )

    original_rgb = np.asarray(
        original_pil
    ).copy()

    # -------------------------------------------------------------------------
    # Resize and normalize input
    # -------------------------------------------------------------------------

    network_input_pil = original_pil.resize(
        IMAGE_SIZE,
        resample=Image.Resampling.BILINEAR,
    )

    image_tensor = TF.to_tensor(
        network_input_pil
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image_tensor.dtype,
    ).view(
        3,
        1,
        1,
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image_tensor.dtype,
    ).view(
        3,
        1,
        1,
    )

    image_tensor = (
        (image_tensor - mean) / std
    ).unsqueeze(0).to(device)

    # -------------------------------------------------------------------------
    # Model inference
    # -------------------------------------------------------------------------

    with torch.inference_mode():
        outputs = generator(
            image_tensor
        )

    if not isinstance(
        outputs,
        dict,
    ):
        raise TypeError(
            "Model output must be a dictionary."
        )

    if "pred_logits" not in outputs:
        raise KeyError(
            'Model output does not contain "pred_logits".'
        )

    if "pred_masks" not in outputs:
        raise KeyError(
            'Model output does not contain "pred_masks".'
        )

    pred_logits = outputs[
        "pred_logits"
    ]

    pred_masks = outputs[
        "pred_masks"
    ]

    if (
        pred_logits.shape[-1]
        <= FOREGROUND_CLASS_INDEX
    ):
        raise ValueError(
            "FOREGROUND_CLASS_INDEX is incompatible "
            "with pred_logits. "
            f"pred_logits.shape={tuple(pred_logits.shape)}, "
            f"FOREGROUND_CLASS_INDEX="
            f"{FOREGROUND_CLASS_INDEX}"
        )

    # -------------------------------------------------------------------------
    # Query confidence and mask probabilities
    # -------------------------------------------------------------------------

    foreground_scores = torch.softmax(
        pred_logits,
        dim=-1,
    )[..., FOREGROUND_CLASS_INDEX]

    mask_probabilities = torch.sigmoid(
        pred_masks
    )

    keep = (
        foreground_scores[0]
        > SCORE_THRESHOLD
    )

    kept_query_indices = (
        torch.nonzero(
            keep,
            as_tuple=False,
        )
        .squeeze(1)
        .detach()
        .cpu()
        .tolist()
    )

    print(
        "All foreground scores:",
        foreground_scores[0]
        .detach()
        .cpu()
        .numpy(),
    )

    print(
        "Kept query indices:",
        kept_query_indices,
    )

    # -------------------------------------------------------------------------
    # Convert retained predictions to binary masks
    # -------------------------------------------------------------------------

    selected_masks = (
        mask_probabilities[0][keep]
        > MASK_THRESHOLD
    ).to(torch.uint8)

    instance_result = (
        selected_masks
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    num_instances = (
        instance_result.shape[0]
    )

    print(
        f"Detected instances: {num_instances}"
    )

    # -------------------------------------------------------------------------
    # Optional instance refinement
    # -------------------------------------------------------------------------

    if REFINE_MODE == "none":
        indices_to_refine = []

    elif REFINE_MODE == "all":
        indices_to_refine = list(
            range(num_instances)
        )

    elif REFINE_MODE == "selected":
        indices_to_refine = sorted(
            set(REFINE_INDICES)
        )

        invalid_indices = [
            index
            for index in indices_to_refine
            if (
                index < 0
                or index >= num_instances
            )
        ]

        if invalid_indices:
            raise IndexError(
                "Invalid REFINE_INDICES: "
                f"{invalid_indices}"
            )

    else:
        raise ValueError(
            "Unsupported REFINE_MODE: "
            f"{REFINE_MODE}"
        )

    for instance_index in indices_to_refine:
        instance_result[
            instance_index
        ] = refine_instance_mask(
            instance_result[
                instance_index
            ]
        )

    # -------------------------------------------------------------------------
    # Restore masks to original image size
    # -------------------------------------------------------------------------

    if RESTORE_TO_ORIGINAL_SIZE:
        instance_result = resize_instance_masks(
            instance_result,
            output_hw=(
                original_height,
                original_width,
            ),
        )

        visualization_rgb = (
            original_rgb
        )

    else:
        visualization_rgb = (
            np.asarray(
                network_input_pil
            ).copy()
        )

    print(
        "Final instance matrix shape "
        "[N, H, W]:",
        instance_result.shape,
    )

    # -------------------------------------------------------------------------
    # Create and save visualization
    # -------------------------------------------------------------------------

    overlay_rgb = create_instance_overlay(
        image_rgb=visualization_rgb,
        instance_masks=instance_result,
        alpha=OVERLAY_ALPHA,
        contour_thickness=CONTOUR_THICKNESS,
    )

    Image.fromarray(
        overlay_rgb
    ).save(
        overlay_save_path
    )

    print(
        "Overlay result saved:",
        overlay_save_path,
    )

    # -------------------------------------------------------------------------
    # Display result
    # -------------------------------------------------------------------------

    if not args.no_show:
        plt.figure(
            figsize=(10, 10)
        )

        plt.imshow(
            overlay_rgb
        )

        plt.axis(
            "off"
        )

        plt.tight_layout(
            pad=0
        )

        plt.show()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("=" * 90)
    print(
        "Inference completed successfully."
    )
    print(
        f"Input image        : {image_name}"
    )
    print(
        f"Detected instances : {num_instances}"
    )
    print(
        f"Overlay image      : "
        f"{overlay_save_path}"
    )
    print(
        "Matrix shape [N, H, W]:",
        instance_result.shape,
    )
    print("=" * 90)


if __name__ == "__main__":
    main()