import glob
import os


# Apply WanDB
import wandb
from datasets import PolypDataset
from hyperparams import (
    batch_size,
    output_dir,
    project,
    seed,
    test_size,
    trainsize,
    val_size,
    n_epochs,
    wandb_host,
    wandb_key
)
from seeder import set_seed_everything
from transforms import semi_transform, train_transform, val_transform

# Import model
from models import OnlineSegmentationModel
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import generate_image_scores, pickle_dump, pickle_load, prioritizer

# Reconfig your API Key here
os.environ["WANDB_API_KEY"] = wandb_key
os.environ["WANDB_BASE_URL"] = wandb_host

# Create output dir
os.makedirs(output_dir, exist_ok=True)
# Login wandb
wandb.login()

set_seed_everything(seed)

f_images = glob.glob("data/TrainDataset/images/*")
f_masks = glob.glob("data/TrainDataset/masks/*")

X_train, X_val, y_train, y_val = train_test_split(
    f_images, f_masks, test_size=val_size, random_state=seed
)
# Use 40% for labeled
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X_train, y_train, test_size=test_size, random_state=seed
)

# Setup dataset
val_dataset = PolypDataset(
    X_val, y_val, trainsize=trainsize, transform=val_transform
)

labeled_dataset = PolypDataset(
    X_labeled, y_labeled, trainsize=trainsize, transform=train_transform
)

unlabeled_dataset = PolypDataset(
    X_unlabeled, y_unlabeled, trainsize=trainsize, transform=semi_transform
)

print(f"Valid size: {len(val_dataset)}")

n_cpu = os.cpu_count()

valid_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_cpu,
    pin_memory=True,
)

labeled_dataloader = DataLoader(
    labeled_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
    pin_memory=True,
)


def train_model(model, experiment_name, max_epochs):
    wandb_logger = WandbLogger(project=project, name=experiment_name, log_model=False)
    trainer = pl.Trainer(gpus=1, logger=wandb_logger, max_epochs=max_epochs)
    trainer.fit(model, val_dataloaders=valid_dataloader)
    wandb.finish()


# Training teacher model
print("TRAINING TEACHER MODEL...")
model = OnlineSegmentationModel(
    "FPN",
    "densenet169",
    in_channels=3,
    out_classes=1,
    use_momentum=True,
    momentum=0.95,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=None,
    checkpoint_path="{}/fcn_densenet169_teacher.pth".format(output_dir),
    mm_checkpoint_path="{}/fcn_densenet169_momentum_teacher.pth".format(output_dir),
)
experiment_name = "Train teacher model - labeled ratio = {} %".format(
    100 - int(test_size * 100)
)

train_model(model, experiment_name, n_epochs)


# Generate new dataloader with smaller batchsize
labeled_dataloader = DataLoader(
    labeled_dataset, batch_size=batch_size // 2, shuffle=True, num_workers=n_cpu, pin_memory=True
)
unlabeled_dataloader = DataLoader(
    unlabeled_dataset, batch_size=batch_size // 2, shuffle=True, num_workers=n_cpu, pin_memory=True
)
# Training student model with offline learning with origin teacher
trained_teacher = torch.load("{}/fcn_densenet169_teacher.pth".format(output_dir))
print("TRAINING STUDENT MODEL OFFLINE FROM ORIGIN TEACHER...")
model = OnlineSegmentationModel(
    "FPN",
    "densenet169",
    in_channels=3,
    out_classes=1,
    momentum=0.99,
    use_momentum=True,
    use_soft_label=True,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=unlabeled_dataloader,
    checkpoint_path="{}/fcn_densenet169_student_offline_origin_teacher.pth".format(
        output_dir
    ),
    mm_checkpoint_path="{}/fcn_densenet169_momentum_student_offline_origin_teacher.pth".format(
        output_dir
    ),
    teacher=trained_teacher,
    is_semi=True,
)
experiment_name = (
    "Train student offline with origin teacher - labeled ratio = {} %".format(
        100 - int(test_size * 100)
    )
)

train_model(model, experiment_name, n_epochs)

# Training student model with online learning with origin teacher
print("TRAINING STUDENT MODEL ONLINE FROM ORIGIN TEACHER...")
model = OnlineSegmentationModel(
    "FPN",
    "densenet169",
    in_channels=3,
    out_classes=1,
    momentum=0.99,
    use_momentum=True,
    use_soft_label=True,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=unlabeled_dataloader,
    checkpoint_path="{}/fcn_densenet169_student_online_origin_teacher.pth".format(
        output_dir
    ),
    mm_checkpoint_path="{}/fcn_densenet169_momentum_student_online_origin_teacher.pth".format(
        output_dir
    ),
    mm_teacher_checkpoint_path="{}/fcn_densenet169_momentum_teacher_online_origin_teacher.pth".format(
        output_dir
    ),
    teacher=trained_teacher,
    is_semi=True,
    is_online=True,
)
experiment_name = (
    "Train student online with origin teacher - labeled ratio = {} %".format(
        100 - int(test_size * 100)
    )
)

train_model(model, experiment_name, n_epochs)

# # Training student model with offline learning with momentum teacher
trained_teacher = torch.load(
    "{}/fcn_densenet169_momentum_teacher.pth".format(output_dir)
)

print("TRAINING STUDENT MODEL OFFLINE FROM MOMENTUM TEACHER...")
model = OnlineSegmentationModel(
    "FPN",
    "densenet169",
    in_channels=3,
    out_classes=1,
    momentum=0.99,
    use_momentum=True,
    use_soft_label=True,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=unlabeled_dataloader,
    checkpoint_path="{}/fcn_densenet169_student_offline_momentum_teacher.pth".format(
        output_dir
    ),
    mm_checkpoint_path="{}/fcn_densenet169_momentum_student_offline_momentum_teacher.pth".format(
        output_dir
    ),
    teacher=trained_teacher,
    is_semi=True,
)
experiment_name = (
    "Train student offline with momentum teacher - labeled ratio = {} %".format(
        100 - int(test_size * 100)
    )
)

train_model(model, experiment_name, n_epochs)

# Training student model with online learning with momentum teacher
print("TRAINING STUDENT MODEL ONLINE FROM MOMENTUM TEACHER...")
model = OnlineSegmentationModel(
    "FPN",
    "densenet169",
    in_channels=3,
    out_classes=1,
    momentum=0.99,
    use_momentum=True,
    use_soft_label=True,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=unlabeled_dataloader,
    checkpoint_path="{}/fcn_densenet169_student_online_momentum_teacher.pth".format(
        output_dir
    ),
    mm_checkpoint_path="{}/fcn_densenet169_momentum_student_online_momentum_teacher.pth".format(
        output_dir
    ),
    mm_teacher_checkpoint_path="{}/fcn_densenet169_momentum_teacher_online_momentum_teacher.pth".format(
        output_dir
    ),
    teacher=trained_teacher,
    is_semi=True,
    is_online=True,
)
experiment_name = (
    "Train student online with momentum teacher - labeled ratio = {} %".format(
        100 - int(test_size * 100)
    )
)

train_model(model, experiment_name, n_epochs)
