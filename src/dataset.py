"""Dataset construction for the reassessment image-only task."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    zip_name: str
    story_id: int
    position_label: int
    split: str


class StoryImageDataset(Dataset):
    def __init__(self, zip_path: Path, samples: list[Sample], transform: Callable | None = None) -> None:
        self.zip_path = Path(zip_path)
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with zipfile.ZipFile(self.zip_path) as archive:
            with archive.open(sample.zip_name) as file:
                image = Image.open(file).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.position_label - 1


def build_samples(zip_path: Path, train_stories: int = 14, val_stories: int = 4) -> tuple[list[Sample], dict]:
    with zipfile.ZipFile(zip_path) as archive:
        image_names = [
            item.filename
            for item in archive.infolist()
            if item.filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    usable_count = (len(image_names) // 5) * 5
    usable_names = image_names[:usable_count]
    excluded_names = image_names[usable_count:]
    total_stories = usable_count // 5

    if train_stories + val_stories > total_stories:
        raise ValueError("Requested split uses more stories than are available.")

    samples: list[Sample] = []
    for idx, name in enumerate(usable_names):
        story_id = idx // 5
        position_label = (idx % 5) + 1
        split = "train" if story_id < train_stories else "validation"
        samples.append(Sample(name, story_id, position_label, split))

    summary = {
        "total_images_in_zip": len(image_names),
        "usable_images": usable_count,
        "excluded_images": excluded_names,
        "total_complete_stories": total_stories,
        "train_stories": train_stories,
        "validation_stories": val_stories,
        "train_images": sum(s.split == "train" for s in samples),
        "validation_images": sum(s.split == "validation" for s in samples),
        "train_class_distribution": {
            str(label): sum(s.split == "train" and s.position_label == label for s in samples)
            for label in range(1, 6)
        },
        "validation_class_distribution": {
            str(label): sum(s.split == "validation" and s.position_label == label for s in samples)
            for label in range(1, 6)
        },
    }
    return samples, summary


def save_dataset_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
