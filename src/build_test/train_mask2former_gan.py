import os
import csv
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import src.build_test.Mask2_GAN as Mask2_GAN

try:
    from src.build_test.Dataset_build_voc import VOC_Dataset
except Exception as e:
    raise ImportError(
        "Failed to import VOC_Dataset.\n"
        "Please confirm that the dataset file path is: src/build_test/Dataset_build_voc.py.\n"
        f"Original error: {repr(e)}"
    )


# ============================================================
# 1. CONFIG: Modify all commonly used parameters here
# ============================================================
@dataclass
class Config:
    # ---------------- Dataset / Input ----------------
    DATA_ROOT: str = r"C:\Users\fssqt\Desktop\DIMFormer-GAN\VOC"
    IMAGE_SIZE: int = 512
    BATCH_SIZE: int = 10
    NUM_WORKERS: int = 0
    USE_NPY: bool = True
    RGB: bool = True
    NORMALIZE: bool = True
    DROP_LAST: bool = False

    # ---------------- Dataset Split ----------------
    # Training / Validation / Test = 70% / 15% / 15%
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    SPLIT_SEED: int = 42
    SPLIT_FILENAME: str = "dataset_split_70_15_15_seed42.json"

    # ---------------- Model Architecture ----------------
    NUM_CLASSES: int = 1
    NUM_QUERIES: int = 6
    HIDDEN_DIM: int = 128
    MASK_DIM: int = 128
    NUM_DECODER_LAYERS: int = 8
    D_BASE_CHANNELS: int = 64

    # ---------------- Hungarian Matching ----------------
    COST_CLASS: float = 2.0
    COST_MASK: float = 5.0
    COST_DICE: float = 5.0
    MATCH_SIZE: int = 256

    # ---------------- Instance Loss ----------------
    EOS_COEF: float = 0.1
    LAMBDA_MASK: float = 5.0
    LAMBDA_DICE: float = 5.0
    AUX_WEIGHT: float = 1.0

    # ---------------- Generator Total Loss Weights ----------------
    LAMBDA_SEM: float = 1.0
    LAMBDA_INS: float = 1.0
    LAMBDA_CONS: float = 0.5
    LAMBDA_ADV: float = 0.01

    # ---------------- Standard Training Parameters ----------------
    EPOCHS: int = 100
    GAN_START_EPOCH: int = 5      # Disable GAN for the first 5 epochs; start training D and the adversarial term from epoch 6
    LR_G: float = 1e-4
    LR_D: float = 5e-5
    BETA1: float = 0.5
    BETA2: float = 0.999
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    SEED: int = 42
    USE_CPU: bool = False
    CUDNN_BENCHMARK: bool = True
    PIN_MEMORY: bool = True


    MAX_ITERS_PER_EPOCH: int = -1
    LOG_INTERVAL: int = 1

    # ---------------- Save / Resume ----------------
    SAVE_DIR: str = r"./checkpoints/mask2_gan_train_100epochs"
    CHECKPOINT_DIRNAME: str = "epoch_checkpoints"
    SAVE_INTERVAL: int = 1
    SAVE_EVERY_EPOCH: bool = True

    LATEST_MODEL_NAME: str = "latest_mask2_gan.pth"
    BEST_MODEL_NAME: str = "best_train_loss_G_mask2_gan.pth"
    FINAL_MODEL_NAME: str = "final_mask2_gan_epoch_100.pth"
    LOSS_CSV_NAME: str = "epoch_loss_history.csv"
    TIME_JSON_NAME: str = "training_time.json"
    CONFIG_JSON_NAME: str = "train_config.json"
    LOSS_FIGURE_NAME: str = "epoch_loss_curves.png"


    RESUME_PATH: str = ""


    RESET_LOSS_CSV: bool = True


CFG = Config()


# ============================================================
# 2. Utility Functions
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(model: torch.nn.Module, flag: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(flag)


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def move_targets_to_device(
    targets: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    moved_targets: List[Dict[str, torch.Tensor]] = []
    for target in targets:
        moved_targets.append({
            "masks": target["masks"].to(device=device, dtype=torch.float32),
            "labels": target["labels"].to(device=device, dtype=torch.long),
            "sem_mask": target["sem_mask"].to(device=device, dtype=torch.float32),
        })
    return moved_targets


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch["images"] = batch["images"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    batch["sem_gt"] = batch["sem_gt"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    batch["instance_union"] = batch["instance_union"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    batch["targets"] = move_targets_to_device(batch["targets"], device)
    return batch


# ============================================================
# 3. collate_fn: Organize per-image instance targets into a batch
# ============================================================
def build_batch_from_targets(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("The current batch is empty and training data cannot be constructed.")

    images = torch.stack([sample["image"].float() for sample in samples], dim=0)
    names = [sample["name"] for sample in samples]

    targets = [Mask2_GAN.build_mask(sample["mask"]) for sample in samples]
    sem_gt = torch.stack([target["sem_mask"].float() for target in targets], dim=0)

    batch_size = len(targets)
    _, _, height, width = sem_gt.shape
    max_instances = max(target["masks"].shape[0] for target in targets)

    batched_masks = torch.zeros(
        (batch_size, max_instances, height, width),
        dtype=torch.float32,
    )
    batched_labels = torch.zeros(
        (batch_size, max_instances),
        dtype=torch.long,
    )
    valid_inst = torch.zeros(
        (batch_size, max_instances),
        dtype=torch.bool,
    )

    for batch_index, target in enumerate(targets):
        num_instances = target["masks"].shape[0]
        if num_instances > 0:
            batched_masks[batch_index, :num_instances] = target["masks"].float()
            batched_labels[batch_index, :num_instances] = target["labels"].long()
            valid_inst[batch_index, :num_instances] = True

    if max_instances > 0:
        instance_union = batched_masks.max(dim=1, keepdim=True)[0]
    else:
        instance_union = torch.zeros(
            (batch_size, 1, height, width),
            dtype=torch.float32,
        )

    return {
        "images": images,
        "targets": targets,
        "sem_gt": sem_gt,
        "batched_masks": batched_masks,
        "batched_labels": batched_labels,
        "valid_inst": valid_inst,
        "instance_union": instance_union,
        "names": names,
    }


# ============================================================
# 4. CSV / JSON Logging
# ============================================================
LOSS_FIELDNAMES = [
    "epoch",
    "global_step",
    "num_iterations",
    "loss_sem",
    "loss_ins",
    "loss_G",
    "loss_D",
    "loss_adv_G",
    "epoch_seconds",
    "total_seconds",
]


def init_loss_csv(csv_path: str, reset: bool) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if reset or not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=LOSS_FIELDNAMES)
            writer.writeheader()


def append_epoch_loss(csv_path: str, row: Dict[str, Any]) -> None:
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=LOSS_FIELDNAMES)
        writer.writerow({key: row.get(key, "") for key in LOSS_FIELDNAMES})


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_or_create_dataset_split(
    dataset_size: int,
    cfg: Config,
) -> Tuple[List[int], List[int], List[int], str]:
    """
    Randomly split the dataset into 70% / 15% / 15% and save the indices to a fixed JSON file.

    As long as DATA_ROOT, SPLIT_FILENAME, and the dataset order remain unchanged, all subsequent comparison experiments
    and ablation experiments will directly reuse the same indices, ensuring completely consistent dataset splits.
    """
    if dataset_size < 3:
        raise ValueError(
            f"The dataset contains only {dataset_size} samples and cannot be split into training/validation/test subsets."
        )

    ratio_sum = cfg.TRAIN_RATIO + cfg.VAL_RATIO + cfg.TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            "The sum of TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must equal 1.0; "
            f"the current sum is {ratio_sum:.8f}."
        )

    split_path = os.path.join(cfg.DATA_ROOT, cfg.SPLIT_FILENAME)

    if os.path.isfile(split_path):
        with open(split_path, mode="r", encoding="utf-8") as file:
            split_info = json.load(file)

        if int(split_info.get("dataset_size", -1)) != dataset_size:
            raise ValueError(
                "The existing dataset split file does not match the current dataset size.\n"
                f"Split file: {split_path}\n"
                f"dataset_size in the file={split_info.get('dataset_size')}, "
                f"current dataset_size={dataset_size}.\n"
                "Please confirm that no samples were added or removed; if a new split is required, back up and delete the existing split file first."
            )

        expected_seed = int(split_info.get("split_seed", cfg.SPLIT_SEED))
        if expected_seed != cfg.SPLIT_SEED:
            raise ValueError(
                f"The existing split uses SPLIT_SEED={expected_seed}, "
                f"but the current configuration uses {cfg.SPLIT_SEED}. To keep experimental splits consistent, use the same setting."
            )

        stored_ratios = split_info.get("ratios", {})
        expected_ratios = {
            "train": cfg.TRAIN_RATIO,
            "val": cfg.VAL_RATIO,
            "test": cfg.TEST_RATIO,
        }
        for key, expected_value in expected_ratios.items():
            stored_value = float(stored_ratios.get(key, -1.0))
            if abs(stored_value - expected_value) > 1e-8:
                raise ValueError(
                    f"The {key} ratio in the existing split file is {stored_value}, "
                    f"while the current configuration is {expected_value}. To keep experimental splits consistent, use the same configuration."
                )

        train_indices = [int(i) for i in split_info["train_indices"]]
        val_indices = [int(i) for i in split_info["val_indices"]]
        test_indices = [int(i) for i in split_info["test_indices"]]

        merged = train_indices + val_indices + test_indices
        if (
            len(merged) != dataset_size
            or len(set(merged)) != dataset_size
            or set(merged) != set(range(dataset_size))
        ):
            raise ValueError(
                f"The indices in the existing dataset split file are invalid or contain duplicates/missing entries: {split_path}"
            )

        print(f"Reusing fixed dataset split: {split_path}")
        return train_indices, val_indices, test_indices, split_path

    # First run: Shuffle indices with a fixed random seed and create a 70/15/15 split.
    indices = np.arange(dataset_size, dtype=np.int64)
    rng = np.random.default_rng(cfg.SPLIT_SEED)
    rng.shuffle(indices)

    train_size = int(dataset_size * cfg.TRAIN_RATIO)
    val_size = int(dataset_size * cfg.VAL_RATIO)
    test_size = dataset_size - train_size - val_size

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            "At least one subset is empty after the 70%/15%/15% split."
            f"dataset_size={dataset_size}, train={train_size}, "
            f"val={val_size}, test={test_size}"
        )

    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size + val_size].tolist()
    test_indices = indices[train_size + val_size:].tolist()

    split_info = {
        "dataset_size": dataset_size,
        "split_seed": cfg.SPLIT_SEED,
        "ratios": {
            "train": cfg.TRAIN_RATIO,
            "val": cfg.VAL_RATIO,
            "test": cfg.TEST_RATIO,
        },
        "sizes": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }
    save_json(split_path, split_info)
    print(f"Created and saved the fixed dataset split for the first time: {split_path}")

    return train_indices, val_indices, test_indices, split_path


# ============================================================
# 5. Build Models, Loss Functions, and Optimizers
# ============================================================
def build_models_losses_optimizers(
    cfg: Config,
    device: torch.device,
) -> Tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    in_channels = 3 if cfg.RGB else 1

    generator = Mask2_GAN.Mask2FormerGANGenerator(
        in_channels=in_channels,
        num_queries=cfg.NUM_QUERIES,
        hidden_dim=cfg.HIDDEN_DIM,
        mask_dim=cfg.MASK_DIM,
        num_classes=cfg.NUM_CLASSES,
        num_decoder_layers=cfg.NUM_DECODER_LAYERS,
    ).to(device)

    discriminator = Mask2_GAN.Discriminator(
        image_channels=in_channels,
        mask_channels=2,
        base_channels=cfg.D_BASE_CHANNELS,
    ).to(device)

    matcher = Mask2_GAN.HungarianMatcher(
        cost_class=cfg.COST_CLASS,
        cost_mask=cfg.COST_MASK,
        cost_dice=cfg.COST_DICE,
        match_size=(cfg.MATCH_SIZE, cfg.MATCH_SIZE),
    )

    instance_criterion = Mask2_GAN.InstanceCriterion(
        matcher=matcher,
        num_classes=cfg.NUM_CLASSES,
        eos_coef=cfg.EOS_COEF,
        lambda_mask=cfg.LAMBDA_MASK,
        lambda_dice=cfg.LAMBDA_DICE,
        aux_weight=cfg.AUX_WEIGHT,
    )

    generator_loss_fn = Mask2_GAN.GeneratorTotalLoss(
        instance_criterion=instance_criterion,
        lambda_sem=cfg.LAMBDA_SEM,
        lambda_ins=cfg.LAMBDA_INS,
        lambda_cons=cfg.LAMBDA_CONS,
        lambda_adv=cfg.LAMBDA_ADV,
    ).to(device)

    # Detach gradients from the semantic branch when training D; the generator forward pass is also wrapped in no_grad
    semantic_constraint_for_D = Mask2_GAN.semationmask(
        detach_semantic=True,
    ).to(device)

    optimizer_G = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.LR_G,
        betas=(cfg.BETA1, cfg.BETA2),
        weight_decay=cfg.WEIGHT_DECAY,
    )

    optimizer_D = torch.optim.AdamW(
        discriminator.parameters(),
        lr=cfg.LR_D,
        betas=(cfg.BETA1, cfg.BETA2),
        weight_decay=cfg.WEIGHT_DECAY,
    )

    return (
        generator,
        discriminator,
        generator_loss_fn,
        semantic_constraint_for_D,
        optimizer_G,
        optimizer_D,
    )


# ============================================================
# 6. Checkpoint Saving and Resuming
# ============================================================
def save_checkpoint(
    save_path: str,
    epoch: int,
    global_step: int,
    best_train_loss_G: float,
    epoch_metrics: Dict[str, float],
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    cfg: Config,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "best_train_loss_G": best_train_loss_G,
        "epoch_metrics": epoch_metrics,
        "G_state_dict": generator.state_dict(),
        "D_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "config": asdict(cfg),
    }, save_path)


def load_checkpoint(
    checkpoint_path: str,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int, float]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint["G_state_dict"])
    discriminator.load_state_dict(checkpoint["D_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

    # Epoch numbers in checkpoints are 1-based; resume training from the next epoch
    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_train_loss_G = float(checkpoint.get("best_train_loss_G", float("inf")))

    print(f"Checkpoint restored: {checkpoint_path}")
    print(f"Training will resume from epoch {start_epoch + 1}.")

    return start_epoch, global_step, best_train_loss_G


# ============================================================
# 7. GAN Training for a Single Epoch
# ============================================================
def train_one_epoch(
    epoch_index: int,
    dataloader: DataLoader,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_loss_fn: torch.nn.Module,
    semantic_constraint_for_D: torch.nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    global_step: int,
) -> Tuple[int, Dict[str, float]]:
    generator.train()
    discriminator.train()

    totals = {
        "loss_sem": 0.0,
        "loss_ins": 0.0,
        "loss_G": 0.0,
        "loss_D": 0.0,
        "loss_adv_G": 0.0,
    }
    num_iterations = 0

    # epoch_index starts from 0; GAN_START_EPOCH=5 means GAN is enabled from the displayed epoch 6
    use_adv = epoch_index >= cfg.GAN_START_EPOCH and cfg.LAMBDA_ADV > 0

    for iteration, batch in enumerate(dataloader):
        if cfg.MAX_ITERS_PER_EPOCH > 0 and iteration >= cfg.MAX_ITERS_PER_EPOCH:
            break

        batch = move_batch_to_device(batch, device)
        images = batch["images"]
        targets = batch["targets"]
        sem_gt = batch["sem_gt"]
        real_union = batch["instance_union"]

        # ----------------------------------------------------
        # A. Update Discriminator D
        # ----------------------------------------------------
        if use_adv:
            set_requires_grad(discriminator, True)
            optimizer_D.zero_grad(set_to_none=True)

            # Do not build a computation graph for G when training D
            with torch.no_grad():
                fake_outputs_for_D = generator(images)
                fake_pack = semantic_constraint_for_D(fake_outputs_for_D)
                fake_semantic = fake_pack["semantic_prob"]
                fake_union = fake_pack["constrained_union"]

            d_loss_dict = Mask2_GAN.discriminator_hinge_loss(
                D=discriminator,
                image=images,
                fake_semantic=fake_semantic,
                fake_instance_union=fake_union,
                real_semantic=sem_gt,
                real_instance_union=real_union,
            )

            loss_D = d_loss_dict["loss_D"]
            loss_D.backward()

            if cfg.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    cfg.GRAD_CLIP,
                )

            optimizer_D.step()
        else:
            loss_D = torch.zeros((), device=device)
            d_loss_dict = {"loss_D": loss_D}

        # ----------------------------------------------------
        # B. Update Generator G
        # ----------------------------------------------------
        # Freeze D parameters while still allowing gradients to flow through D back to G outputs
        set_requires_grad(discriminator, False)
        optimizer_G.zero_grad(set_to_none=True)

        outputs = generator(images)
        g_loss_dict = generator_loss_fn(
            outputs=outputs,
            targets=targets,
            sem_gt=sem_gt,
            image=images,
            D=discriminator,
            use_adv=use_adv,
        )

        loss_G = g_loss_dict["loss_G"]
        loss_G.backward()

        if cfg.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                cfg.GRAD_CLIP,
            )

        optimizer_G.step()
        set_requires_grad(discriminator, True)

        # ----------------------------------------------------
        # C. Accumulate losses for this epoch; write only one CSV row after the epoch ends
        # ----------------------------------------------------
        current = {
            "loss_sem": float(g_loss_dict["loss_sem"].detach().cpu()),
            "loss_ins": float(g_loss_dict["loss_ins_total"].detach().cpu()),
            "loss_G": float(g_loss_dict["loss_G"].detach().cpu()),
            "loss_D": float(d_loss_dict["loss_D"].detach().cpu()),
            "loss_adv_G": float(g_loss_dict["loss_adv_G"].detach().cpu()),
        }

        for key in totals:
            totals[key] += current[key]

        num_iterations += 1
        global_step += 1

        if (iteration + 1) % cfg.LOG_INTERVAL == 0:
            print(
                f"[Epoch {epoch_index + 1:03d}/{cfg.EPOCHS:03d}] "
                f"[Iter {iteration + 1:04d}/{len(dataloader):04d}] "
                f"step={global_step:07d} | "
                f"sem={current['loss_sem']:.6f} | "
                f"ins={current['loss_ins']:.6f} | "
                f"G={current['loss_G']:.6f} | "
                f"D={current['loss_D']:.6f} | "
                f"adv_G={current['loss_adv_G']:.6f} | "
                f"GAN={'ON' if use_adv else 'OFF'}"
            )

    if num_iterations == 0:
        raise RuntimeError(
            "No iterations were executed in the current epoch."
            "Please check whether the dataset is empty or MAX_ITERS_PER_EPOCH is configured incorrectly."
        )

    averages = {
        key: value / num_iterations
        for key, value in totals.items()
    }
    averages["num_iterations"] = float(num_iterations)

    return global_step, averages


# ============================================================
# 8. Plot Loss Curves
# ============================================================
def draw_loss_curves(csv_path: str, save_path: str) -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"pandas or matplotlib is not installed; skipping loss curve plotting: {repr(e)}")
        return

    if not os.path.exists(csv_path):
        print(f"Loss CSV does not exist; skipping plotting: {csv_path}")
        return

    dataframe = pd.read_csv(csv_path)
    required_columns = [
        "epoch",
        "loss_sem",
        "loss_ins",
        "loss_G",
        "loss_D",
        "loss_adv_G",
    ]
    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]
    if missing_columns:
        print(f"Loss CSV is missing columns {missing_columns}; skipping plotting.")
        return

    plt.figure(figsize=(12, 7))
    plt.plot(dataframe["epoch"], dataframe["loss_sem"], label="Semantic loss")
    plt.plot(dataframe["epoch"], dataframe["loss_ins"], label="Instance loss")
    plt.plot(dataframe["epoch"], dataframe["loss_G"], label="Generator loss")
    plt.plot(dataframe["epoch"], dataframe["loss_D"], label="Discriminator loss")
    plt.plot(dataframe["epoch"], dataframe["loss_adv_G"], label="Generator adversarial loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Mask2Former-GAN Training Losses")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss curve saved: {save_path}")


# ============================================================
# 9. Main Function: Fixed 100-Epoch Standard Training
# ============================================================
def main() -> None:
    cfg = CFG
    set_seed(cfg.SEED)

    if cfg.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cfg.USE_CPU else "cpu"
    )

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(cfg.SAVE_DIR, cfg.CHECKPOINT_DIRNAME)
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_csv_path = os.path.join(cfg.SAVE_DIR, cfg.LOSS_CSV_NAME)
    latest_model_path = os.path.join(cfg.SAVE_DIR, cfg.LATEST_MODEL_NAME)
    best_model_path = os.path.join(cfg.SAVE_DIR, cfg.BEST_MODEL_NAME)
    final_model_path = os.path.join(cfg.SAVE_DIR, cfg.FINAL_MODEL_NAME)
    config_json_path = os.path.join(cfg.SAVE_DIR, cfg.CONFIG_JSON_NAME)
    time_json_path = os.path.join(cfg.SAVE_DIR, cfg.TIME_JSON_NAME)
    loss_figure_path = os.path.join(cfg.SAVE_DIR, cfg.LOSS_FIGURE_NAME)

    # Preserve the existing CSV when resuming; when training from scratch, reset it according to the configuration
    reset_csv = cfg.RESET_LOSS_CSV and not bool(cfg.RESUME_PATH.strip())
    init_loss_csv(loss_csv_path, reset=reset_csv)
    save_json(config_json_path, asdict(cfg))

    print("=" * 90)
    print("Mask2Former-GAN: Fixed 70%/15%/15% dataset split, with the training set trained for a fixed 100 epochs")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"DATA_ROOT: {cfg.DATA_ROOT}")
    print(f"SAVE_DIR: {cfg.SAVE_DIR}")
    print(f"Dataset image size: {cfg.IMAGE_SIZE} x {cfg.IMAGE_SIZE}")
    print(f"Batch size: {cfg.BATCH_SIZE}")
    print(f"Epochs: {cfg.EPOCHS}")
    print(f"GAN starts after the first {cfg.GAN_START_EPOCH} epoch(s)")
    print(f"Epoch-level loss CSV: {loss_csv_path}")
    print("=" * 90)

    # --------------------------------------------------------
    # A. Load the full dataset and create a fixed random 70% / 15% / 15% split
    #    All comparison and ablation experiments reuse the same split JSON under DATA_ROOT
    # --------------------------------------------------------
    dataset = VOC_Dataset(
        root=cfg.DATA_ROOT,
        image_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        use_npy=cfg.USE_NPY,
        rgb=cfg.RGB,
        normalize=cfg.NORMALIZE,
    )

    if len(dataset) == 0:
        raise ValueError(f"The dataset is empty; please check DATA_ROOT: {cfg.DATA_ROOT}")

    train_indices, val_indices, test_indices, split_path = load_or_create_dataset_split(
        dataset_size=len(dataset),
        cfg=cfg,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Shuffle the training set every epoch; keep validation and test sets in fixed order and never use them for gradient updates.
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(cfg.SEED)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        generator=train_loader_generator,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.PIN_MEMORY and device.type == "cuda"),
        drop_last=cfg.DROP_LAST,
        collate_fn=build_batch_from_targets,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.PIN_MEMORY and device.type == "cuda"),
        drop_last=False,
        collate_fn=build_batch_from_targets,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.PIN_MEMORY and device.type == "cuda"),
        drop_last=False,
        collate_fn=build_batch_from_targets,
    )

    print(f"Dataset total size: {len(dataset)}")
    print(
        f"Split sizes -> train: {len(train_dataset)} ({cfg.TRAIN_RATIO:.0%}), "
        f"val: {len(val_dataset)} ({cfg.VAL_RATIO:.0%}), "
        f"test: {len(test_dataset)} ({cfg.TEST_RATIO:.0%})"
    )
    print(f"Fixed split file: {split_path}")
    print(f"Train iterations per epoch: {len(train_dataloader)}")
    print(f"Validation iterations: {len(val_dataloader)}")
    print(f"Test iterations: {len(test_dataloader)}")

    # --------------------------------------------------------
    # B. Build Models, Losses, and Optimizers
    # --------------------------------------------------------
    (
        generator,
        discriminator,
        generator_loss_fn,
        semantic_constraint_for_D,
        optimizer_G,
        optimizer_D,
    ) = build_models_losses_optimizers(cfg, device)

    start_epoch = 0
    global_step = 0
    best_train_loss_G = float("inf")

    if cfg.RESUME_PATH.strip():
        start_epoch, global_step, best_train_loss_G = load_checkpoint(
            checkpoint_path=cfg.RESUME_PATH,
            generator=generator,
            discriminator=discriminator,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            device=device,
        )

    if start_epoch >= cfg.EPOCHS:
        raise ValueError(
            f"The checkpoint has already been trained through epoch {start_epoch}, "
            f"but the current EPOCHS={cfg.EPOCHS}, so there are no remaining epochs to train."
        )

    sync_cuda(device)
    training_start_time = time.perf_counter()

    # --------------------------------------------------------
    # C. Train for a Fixed Number of Epochs
    # --------------------------------------------------------
    for epoch_index in range(start_epoch, cfg.EPOCHS):
        epoch_start_time = time.perf_counter()

        global_step, epoch_avg = train_one_epoch(
            epoch_index=epoch_index,
            dataloader=train_dataloader,
            generator=generator,
            discriminator=discriminator,
            generator_loss_fn=generator_loss_fn,
            semantic_constraint_for_D=semantic_constraint_for_D,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            device=device,
            cfg=cfg,
            global_step=global_step,
        )

        sync_cuda(device)
        epoch_seconds = time.perf_counter() - epoch_start_time
        total_seconds = time.perf_counter() - training_start_time

        epoch_number = epoch_index + 1
        epoch_row = {
            "epoch": epoch_number,
            "global_step": global_step,
            "num_iterations": int(epoch_avg["num_iterations"]),
            "loss_sem": epoch_avg["loss_sem"],
            "loss_ins": epoch_avg["loss_ins"],
            "loss_G": epoch_avg["loss_G"],
            "loss_D": epoch_avg["loss_D"],
            "loss_adv_G": epoch_avg["loss_adv_G"],
            "epoch_seconds": epoch_seconds,
            "total_seconds": total_seconds,
        }
        append_epoch_loss(loss_csv_path, epoch_row)

        print("-" * 90)
        print(
            f"[Epoch {epoch_number:03d}/{cfg.EPOCHS:03d} completed] "
            f"sem={epoch_avg['loss_sem']:.6f} | "
            f"ins={epoch_avg['loss_ins']:.6f} | "
            f"G={epoch_avg['loss_G']:.6f} | "
            f"D={epoch_avg['loss_D']:.6f} | "
            f"adv_G={epoch_avg['loss_adv_G']:.6f} | "
            f"time={epoch_seconds:.2f}s"
        )
        print(f"Average loss for this epoch written to: {loss_csv_path}")

        # Update the best value before writing the checkpoint to ensure the latest best value is saved
        is_best = epoch_avg["loss_G"] < best_train_loss_G
        if is_best:
            best_train_loss_G = epoch_avg["loss_G"]

        checkpoint_metrics = {
            "loss_sem": epoch_avg["loss_sem"],
            "loss_ins": epoch_avg["loss_ins"],
            "loss_G": epoch_avg["loss_G"],
            "loss_D": epoch_avg["loss_D"],
            "loss_adv_G": epoch_avg["loss_adv_G"],
        }

        # latest: Overwrite every epoch to support resuming after interruption
        save_checkpoint(
            save_path=latest_model_path,
            epoch=epoch_number,
            global_step=global_step,
            best_train_loss_G=best_train_loss_G,
            epoch_metrics=checkpoint_metrics,
            generator=generator,
            discriminator=discriminator,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            cfg=cfg,
        )

        # Save a separate checkpoint for each epoch; by default, save every epoch
        if (
            cfg.SAVE_EVERY_EPOCH
            and cfg.SAVE_INTERVAL > 0
            and epoch_number % cfg.SAVE_INTERVAL == 0
        ):
            epoch_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"mask2_gan_epoch_{epoch_number:03d}.pth",
            )
            save_checkpoint(
                save_path=epoch_checkpoint_path,
                epoch=epoch_number,
                global_step=global_step,
                best_train_loss_G=best_train_loss_G,
                epoch_metrics=checkpoint_metrics,
                generator=generator,
                discriminator=discriminator,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                cfg=cfg,
            )
            print(f"Checkpoint for this epoch saved: {epoch_checkpoint_path}")

        # Preserve the original saving logic: use only the training subset epoch-average loss_G as the "best training" criterion; validation/test subsets do not participate in gradient updates
        if is_best:
            save_checkpoint(
                save_path=best_model_path,
                epoch=epoch_number,
                global_step=global_step,
                best_train_loss_G=best_train_loss_G,
                epoch_metrics=checkpoint_metrics,
                generator=generator,
                discriminator=discriminator,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                cfg=cfg,
            )
            print(
                f"Training-set loss_G reached a new best value: {best_train_loss_G:.6f}, "
                f"model saved to: {best_model_path}"
            )

        # Update the curves once per epoch so existing results are preserved even if training is interrupted
        draw_loss_curves(loss_csv_path, loss_figure_path)
        print("-" * 90)

    # --------------------------------------------------------
    # D. Save the Final Model and Training Time
    # --------------------------------------------------------
    sync_cuda(device)
    total_training_seconds = time.perf_counter() - training_start_time

    final_metrics = {
        "loss_sem": epoch_avg["loss_sem"],
        "loss_ins": epoch_avg["loss_ins"],
        "loss_G": epoch_avg["loss_G"],
        "loss_D": epoch_avg["loss_D"],
        "loss_adv_G": epoch_avg["loss_adv_G"],
    }

    save_checkpoint(
        save_path=final_model_path,
        epoch=cfg.EPOCHS,
        global_step=global_step,
        best_train_loss_G=best_train_loss_G,
        epoch_metrics=final_metrics,
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        cfg=cfg,
    )

    time_info = {
        "epochs_configured": cfg.EPOCHS,
        "start_epoch": start_epoch + 1,
        "last_epoch": cfg.EPOCHS,
        "global_step": global_step,
        "dataset_size": len(dataset),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "test_dataset_size": len(test_dataset),
        "split_ratios": {
            "train": cfg.TRAIN_RATIO,
            "val": cfg.VAL_RATIO,
            "test": cfg.TEST_RATIO,
        },
        "split_seed": cfg.SPLIT_SEED,
        "split_file": split_path,
        "batch_size": cfg.BATCH_SIZE,
        "total_training_seconds_this_run": total_training_seconds,
        "total_training_hours_this_run": total_training_seconds / 3600.0,
        "best_train_loss_G": best_train_loss_G,
        "loss_csv": loss_csv_path,
        "latest_checkpoint": latest_model_path,
        "best_checkpoint": best_model_path,
        "final_checkpoint": final_model_path,
        "loss_figure": loss_figure_path,
    }
    save_json(time_json_path, time_info)
    draw_loss_curves(loss_csv_path, loss_figure_path)

    print("=" * 90)
    print("Completed 100 training epochs on the training set; validation and test sets remained separate and were not used for gradient updates.")
    print(f"Total runtime for this run: {total_training_seconds / 3600.0:.4f} hours")
    print(f"Total runtime for this run: {total_training_seconds:.2f} seconds")
    print(f"Five average loss values per epoch: {loss_csv_path}")
    print(f"Loss curve: {loss_figure_path}")
    print(f"latest model: {latest_model_path}")
    print(f"Best training-set loss_G model: {best_model_path}")
    print(f"Final model at epoch 100: {final_model_path}")
    print(f"Training time record: {time_json_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()

