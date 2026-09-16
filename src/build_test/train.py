"""
Mask2Former-GAN training script with a robust fixed dataset split.
"""
import os
import csv
import json
import time
import random
import shutil
from pathlib import Path
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
        "Please make sure the dataset module is located at: "
        "src/build_test/Dataset_build_voc.py.\n"
        f"Original error: {repr(e)}"
    )


def find_data_root(folder_name: str = "VOC") -> str:
    search_starts = [
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
    ]

    checked = set()

    for start_dir in search_starts:
        for base_dir in [start_dir, *start_dir.parents]:
            candidate = (base_dir / folder_name).resolve()

            candidate_key = str(candidate).lower()
            if candidate_key in checked:
                continue
            checked.add(candidate_key)

            if candidate.is_dir():
                print(f"Dataset directory found: {candidate}")
                return str(candidate)

    checked_text = "\n".join(f"  - {path}" for path in sorted(checked))
    raise FileNotFoundError(
        f"Could not find the dataset directory '{folder_name}'.\n"
        "Please place the dataset folder inside the project directory.\n"
        "Checked locations include:\n"
        f"{checked_text}"
    )

# ============================================================
# 1. CONFIG: commonly used parameters
# ============================================================
@dataclass
class Config:
    # ---------------- Dataset / input ----------------
    DATA_ROOT: str = find_data_root()
    IMAGE_SIZE: int = 512
    BATCH_SIZE: int = 10
    NUM_WORKERS: int = 0
    USE_NPY: bool = True
    RGB: bool = True
    NORMALIZE: bool = True
    DROP_LAST: bool = False

    # ---------------- Dataset split ----------------
    # Train / validation / test = 70% / 15% / 15%.
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    SPLIT_SEED: int = 42
    SPLIT_FILENAME: str = "dataset_split_70_15_15_seed42.json"

    # If the existing split JSON does not match the current dataset,
    # automatically back it up and generate a new split.
    AUTO_REBUILD_SPLIT_ON_MISMATCH: bool = True

    # Set this to True only when you intentionally want to recreate the split
    # even if the existing split file is still valid.
    FORCE_RECREATE_SPLIT: bool = False

    # Preserve the previous split file before rebuilding it.
    BACKUP_OLD_SPLIT: bool = True
    SPLIT_BACKUP_DIRNAME: str = "split_backups"

    # ---------------- Model architecture ----------------
    NUM_CLASSES: int = 1
    NUM_QUERIES: int = 6
    HIDDEN_DIM: int = 128
    MASK_DIM: int = 128
    NUM_DECODER_LAYERS: int = 8
    D_BASE_CHANNELS: int = 64

    # ---------------- Hungarian matching ----------------
    COST_CLASS: float = 2.0
    COST_MASK: float = 5.0
    COST_DICE: float = 5.0
    MATCH_SIZE: int = 256

    # ---------------- Instance loss ----------------
    EOS_COEF: float = 0.1
    LAMBDA_MASK: float = 5.0
    LAMBDA_DICE: float = 5.0
    AUX_WEIGHT: float = 1.0

    # ---------------- Generator loss weights ----------------
    LAMBDA_SEM: float = 1.0
    LAMBDA_INS: float = 1.0
    LAMBDA_CONS: float = 0.5
    LAMBDA_ADV: float = 0.01

    # ---------------- Training parameters ----------------
    EPOCHS: int = 100
    GAN_START_EPOCH: int = 5      # Disable GAN for the first 5 epochs; enable D/adversarial loss from epoch 6
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

    # ---------------- Saving / resuming ----------------
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
# 2. Utility functions
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
# 3. collate_fn: organize per-image instance targets into a batch
# ============================================================
def build_batch_from_targets(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("The current batch is empty and cannot be assembled.")

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
# 4. CSV / JSON logging
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


def _validate_split_ratios(cfg: Config) -> None:
    """Validate that the configured split ratios form a valid partition."""
    ratio_sum = cfg.TRAIN_RATIO + cfg.VAL_RATIO + cfg.TEST_RATIO

    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            "TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must sum to 1.0. "
            f"Current sum: {ratio_sum:.8f}."
        )

    if min(cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.TEST_RATIO) <= 0:
        raise ValueError(
            "TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must all be greater than 0."
        )


def _build_new_split(
    dataset_size: int,
    cfg: Config,
    split_path: str,
) -> Tuple[List[int], List[int], List[int], str]:
    """
    Create a deterministic train/validation/test split and save it to JSON.

    With dataset_size=200 and ratios 0.70/0.15/0.15, this produces:
        train = 140
        val   = 30
        test  = 30
    """
    if dataset_size < 3:
        raise ValueError(
            f"The dataset contains only {dataset_size} sample(s), which is not "
            "enough to create train/validation/test subsets."
        )

    _validate_split_ratios(cfg)

    indices = np.arange(dataset_size, dtype=np.int64)
    rng = np.random.default_rng(cfg.SPLIT_SEED)
    rng.shuffle(indices)

    train_size = int(dataset_size * cfg.TRAIN_RATIO)
    val_size = int(dataset_size * cfg.VAL_RATIO)
    test_size = dataset_size - train_size - val_size

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            "At least one subset would be empty after splitting. "
            f"dataset_size={dataset_size}, train={train_size}, "
            f"val={val_size}, test={test_size}."
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

    print(f"Created a new fixed dataset split: {split_path}")
    print(
        "Split sizes -> "
        f"train={len(train_indices)}, "
        f"val={len(val_indices)}, "
        f"test={len(test_indices)}"
    )

    return train_indices, val_indices, test_indices, split_path


def _get_split_mismatch_reason(
    split_info: Dict[str, Any],
    dataset_size: int,
    cfg: Config,
) -> str:
    """
    Return an empty string if the stored split is compatible with the current
    dataset/configuration. Otherwise, return a human-readable mismatch reason.
    """
    stored_dataset_size = int(split_info.get("dataset_size", -1))
    if stored_dataset_size != dataset_size:
        return (
            "dataset size mismatch: "
            f"stored dataset_size={stored_dataset_size}, "
            f"current dataset_size={dataset_size}"
        )

    stored_seed = int(split_info.get("split_seed", -1))
    if stored_seed != cfg.SPLIT_SEED:
        return (
            "split seed mismatch: "
            f"stored SPLIT_SEED={stored_seed}, "
            f"current SPLIT_SEED={cfg.SPLIT_SEED}"
        )

    stored_ratios = split_info.get("ratios", {})
    expected_ratios = {
        "train": cfg.TRAIN_RATIO,
        "val": cfg.VAL_RATIO,
        "test": cfg.TEST_RATIO,
    }

    for key, expected_value in expected_ratios.items():
        try:
            stored_value = float(stored_ratios.get(key, -1.0))
        except (TypeError, ValueError):
            return f"invalid stored ratio for '{key}'"

        if abs(stored_value - expected_value) > 1e-8:
            return (
                f"{key} ratio mismatch: "
                f"stored={stored_value}, current={expected_value}"
            )

    required_keys = ["train_indices", "val_indices", "test_indices"]
    for key in required_keys:
        if key not in split_info:
            return f"missing key '{key}' in the split JSON"

    try:
        train_indices = [int(i) for i in split_info["train_indices"]]
        val_indices = [int(i) for i in split_info["val_indices"]]
        test_indices = [int(i) for i in split_info["test_indices"]]
    except (TypeError, ValueError):
        return "the stored split indices are not valid integers"

    merged = train_indices + val_indices + test_indices

    if len(merged) != dataset_size:
        return (
            "the total number of stored indices does not match dataset_size: "
            f"indices={len(merged)}, dataset_size={dataset_size}"
        )

    if len(set(merged)) != dataset_size:
        return "the stored split contains duplicated indices"

    if set(merged) != set(range(dataset_size)):
        return "the stored split contains missing or out-of-range indices"

    return ""


def _backup_split_file(
    split_path: str,
    cfg: Config,
) -> str:
    """Copy the old split JSON to a timestamped backup file."""
    if not cfg.BACKUP_OLD_SPLIT or not os.path.isfile(split_path):
        return ""

    backup_dir = os.path.join(cfg.DATA_ROOT, cfg.SPLIT_BACKUP_DIRNAME)
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(split_path)
    stem, ext = os.path.splitext(filename)

    backup_path = os.path.join(
        backup_dir,
        f"{stem}_backup_{timestamp}{ext}",
    )

    # Avoid overwriting a backup if the function is called twice
    # within the same second.
    suffix = 1
    while os.path.exists(backup_path):
        backup_path = os.path.join(
            backup_dir,
            f"{stem}_backup_{timestamp}_{suffix:02d}{ext}",
        )
        suffix += 1

    shutil.copy2(split_path, backup_path)
    print(f"Backed up the previous split file to: {backup_path}")
    return backup_path


def load_or_create_dataset_split(
    dataset_size: int,
    cfg: Config,
) -> Tuple[List[int], List[int], List[int], str]:
    """
    Load a fixed dataset split or rebuild it when the stored metadata no longer
    matches the current dataset.

    Why this is needed:
        A split JSON stores sample indices for a specific dataset size.
        If the dataset later changes from, for example, 10 samples to 200
        samples, those old indices are no longer a valid partition.

    Default behavior in this version:
        1. Reuse the existing split when it is valid.
        2. If it is stale or invalid, back it up.
        3. Automatically create a new deterministic split using SPLIT_SEED.
        4. Reuse the newly generated JSON in later runs.

    Set AUTO_REBUILD_SPLIT_ON_MISMATCH=False if you prefer strict behavior
    that raises an error instead of rebuilding automatically.
    """
    if dataset_size < 3:
        raise ValueError(
            f"The dataset contains only {dataset_size} sample(s), which is not "
            "enough to create train/validation/test subsets."
        )

    _validate_split_ratios(cfg)

    split_path = os.path.join(cfg.DATA_ROOT, cfg.SPLIT_FILENAME)

    if cfg.FORCE_RECREATE_SPLIT:
        if os.path.isfile(split_path):
            print(
                "FORCE_RECREATE_SPLIT=True. "
                "The current split file will be replaced."
            )
            _backup_split_file(split_path, cfg)

        return _build_new_split(
            dataset_size=dataset_size,
            cfg=cfg,
            split_path=split_path,
        )

    if not os.path.isfile(split_path):
        return _build_new_split(
            dataset_size=dataset_size,
            cfg=cfg,
            split_path=split_path,
        )

    try:
        with open(split_path, mode="r", encoding="utf-8") as file:
            split_info = json.load(file)
        mismatch_reason = _get_split_mismatch_reason(
            split_info=split_info,
            dataset_size=dataset_size,
            cfg=cfg,
        )
    except Exception as exc:
        split_info = {}
        mismatch_reason = (
            "failed to read or validate the existing split JSON: "
            f"{repr(exc)}"
        )

    if not mismatch_reason:
        train_indices = [int(i) for i in split_info["train_indices"]]
        val_indices = [int(i) for i in split_info["val_indices"]]
        test_indices = [int(i) for i in split_info["test_indices"]]

        print(f"Reusing the fixed dataset split: {split_path}")
        return train_indices, val_indices, test_indices, split_path

    message = (
        "The existing dataset split is incompatible with the current dataset.\n"
        f"Split file: {split_path}\n"
        f"Reason: {mismatch_reason}\n"
    )

    if not cfg.AUTO_REBUILD_SPLIT_ON_MISMATCH:
        raise ValueError(
            message
            + "To rebuild automatically, set "
              "AUTO_REBUILD_SPLIT_ON_MISMATCH=True, or manually back up and "
              "delete the old split JSON."
        )

    print("=" * 90)
    print("WARNING: rebuilding the dataset split.")
    print(message.rstrip())
    _backup_split_file(split_path, cfg)

    train_indices, val_indices, test_indices, split_path = _build_new_split(
        dataset_size=dataset_size,
        cfg=cfg,
        split_path=split_path,
    )

    print(
        "The stale split has been replaced with a new deterministic split. "
        "Future runs will reuse this file."
    )
    print("=" * 90)

    return train_indices, val_indices, test_indices, split_path


# ============================================================
# 5. Build models, loss functions, and optimizers
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

    # Detach semantic gradients for D; the generator forward pass for D is also wrapped in no_grad
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
# 6. Checkpoint saving and loading
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

    # The checkpoint stores epochs using 1-based numbering; resume from the next epoch
    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_train_loss_G = float(checkpoint.get("best_train_loss_G", float("inf")))

    print(f"Checkpoint restored: {checkpoint_path}")
    print(f"Training will resume from epoch {start_epoch + 1}.")

    return start_epoch, global_step, best_train_loss_G


# ============================================================
# 7. Train one GAN epoch
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

    # epoch_index is 0-based; GAN_START_EPOCH=5 enables GAN at displayed epoch 6
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
        # A. Update discriminator D
        # ----------------------------------------------------
        if use_adv:
            set_requires_grad(discriminator, True)
            optimizer_D.zero_grad(set_to_none=True)

            # Do not build a generator computation graph while training D
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
        # B. Update generator G
        # ----------------------------------------------------
        # Freeze D parameters while still allowing gradients to flow through D to G outputs
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
        # C. Accumulate losses; write one CSV row after the epoch finishes
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
            "No iteration was executed in the current epoch. "
            "Check whether the dataset is empty or MAX_ITERS_PER_EPOCH is configured incorrectly."
        )

    averages = {
        key: value / num_iterations
        for key, value in totals.items()
    }
    averages["num_iterations"] = float(num_iterations)

    return global_step, averages


# ============================================================
# 8. Draw loss curves
# ============================================================
def draw_loss_curves(csv_path: str, save_path: str) -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"pandas or matplotlib is unavailable; skipping loss plotting: {repr(e)}")
        return

    if not os.path.exists(csv_path):
        print(f"Loss CSV does not exist; skipping plot: {csv_path}")
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
        print(f"Loss CSV is missing columns {missing_columns}; skipping plot.")
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
    print(f"Loss curves saved to: {save_path}")


# ============================================================
# 9. Main function: fixed 100-epoch training
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

    # Keep the existing CSV when resuming; optionally reset it for a fresh run
    reset_csv = cfg.RESET_LOSS_CSV and not bool(cfg.RESUME_PATH.strip())
    init_loss_csv(loss_csv_path, reset=reset_csv)
    save_json(config_json_path, asdict(cfg))

    print("=" * 90)
    print("Mask2Former-GAN: fixed 70%/15%/15% split, 100 training epochs")
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
    # A. Load the full dataset and create/reuse a fixed 70%/15%/15% split
    #    All comparison and ablation experiments should reuse the same split JSON under DATA_ROOT
    # --------------------------------------------------------
    dataset = VOC_Dataset(
        root=cfg.DATA_ROOT,
        image_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        use_npy=cfg.USE_NPY,
        rgb=cfg.RGB,
        normalize=cfg.NORMALIZE,
    )

    if len(dataset) == 0:
        raise ValueError(f"The dataset is empty. Please check DATA_ROOT: {cfg.DATA_ROOT}")

    train_indices, val_indices, test_indices, split_path = load_or_create_dataset_split(
        dataset_size=len(dataset),
        cfg=cfg,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Shuffle the training subset each epoch; validation/test remain ordered and never receive gradient updates
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
    # B. Build models, losses, and optimizers
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
            f"The checkpoint has already reached epoch {start_epoch}, "
            f"but EPOCHS={cfg.EPOCHS}; there are no remaining epochs to train."
        )

    sync_cuda(device)
    training_start_time = time.perf_counter()

    # --------------------------------------------------------
    # C. Fixed-number epoch training
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
        print(f"Epoch-average losses written to: {loss_csv_path}")

        # Update the best value before saving checkpoints so the newest best value is stored
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

        # latest: overwrite every epoch to support interruption recovery
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

        # Save an independent epoch checkpoint at the configured interval
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
            print(f"Epoch checkpoint saved to: {epoch_checkpoint_path}")

        # Keep the original selection logic: the best checkpoint is based only on mean training loss_G
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
                f"New best training loss_G: {best_train_loss_G:.6f}. "
                f"Model saved to: {best_model_path}"
            )

        # Update curves every epoch so partial results remain available after interruption
        draw_loss_curves(loss_csv_path, loss_figure_path)
        print("-" * 90)

    # --------------------------------------------------------
    # D. Save the final model and timing information
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
    print("Completed 100 training epochs; validation and test subsets remained independent.")
    print(f"Total runtime: {total_training_seconds / 3600.0:.4f} hours")
    print(f"Total runtime: {total_training_seconds:.2f} seconds")
    print(f"Epoch-average loss CSV: {loss_csv_path}")
    print(f"Loss curves: {loss_figure_path}")
    print(f"Latest checkpoint: {latest_model_path}")
    print(f"Best training-loss_G checkpoint: {best_model_path}")
    print(f"Final epoch checkpoint: {final_model_path}")
    print(f"Training-time record: {time_json_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()