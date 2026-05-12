"""Train baseline and five CNN experiment variations."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dataset import StoryImageDataset, build_samples, save_dataset_summary
from model import StoryPositionCNN


ROOT = Path(__file__).resolve().parents[1]
PACKAGED_ZIP = ROOT / "data" / "dnn dataset.zip"
DEFAULT_ZIP = PACKAGED_ZIP if PACKAGED_ZIP.exists() else Path.home() / "Downloads" / "dnn dataset.zip"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += labels.size(0)

    return total_loss / total_items, total_correct / total_items


def collect_predictions(model, loader, device="cpu"):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    return y_true, y_pred


def make_loaders(zip_path: Path, augment: bool, batch_size: int):
    samples, summary = build_samples(zip_path)
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "validation"]

    train_steps = [transforms.Resize((64, 64))]
    if augment:
        train_steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ])
    train_steps.extend([transforms.ToTensor()])

    eval_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataset = StoryImageDataset(zip_path, train_samples, transforms.Compose(train_steps))
    val_dataset = StoryImageDataset(zip_path, val_samples, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, summary


def experiment_configs():
    return [
        {"name": "Baseline CNN", "modification": "3-block CNN", "kwargs": {}, "augment": False},
        {"name": "Dropout", "modification": "Adds dropout 0.50", "kwargs": {"dropout": 0.5}, "augment": False},
        {"name": "Batch Normalization", "modification": "Adds batch normalization", "kwargs": {"batch_norm": True}, "augment": False},
        {"name": "More Filters", "modification": "Uses 32-64-128 filters", "kwargs": {"filters": (32, 64, 128)}, "augment": False},
        {"name": "Kernel Size 5", "modification": "Changes kernel size from 3 to 5", "kwargs": {"kernel_size": 5}, "augment": False},
        {"name": "Data Augmentation", "modification": "Adds flip and rotation", "kwargs": {}, "augment": True},
    ]


def train_experiment(config, zip_path: Path, epochs: int, batch_size: int, device: str):
    train_loader, val_loader, summary = make_loaders(zip_path, config["augment"], batch_size)
    model = StoryPositionCNN(**config["kwargs"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        history.append({
            "experiment": config["name"],
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        })
        print(
            f"{config['name']} epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

    y_true, y_pred = collect_predictions(model, val_loader, device)
    return history, summary, {"experiment": config["name"], "y_true": y_true, "y_pred": y_pred}


def plot_loss_curves(all_history: list[dict], configs: list[dict], plot_dir: Path) -> None:
    plt.figure(figsize=(11, 7))
    for config in configs:
        rows = [row for row in all_history if row["experiment"] == config["name"]]
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row["train_loss"] for row in rows], linestyle="--", label=f"{config['name']} train")
        plt.plot(epochs, [row["validation_loss"] for row in rows], label=f"{config['name']} val")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.title("Training and Validation Loss Curves")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curves.png", dpi=160)
    plt.close()


def plot_accuracy_curves(all_history: list[dict], configs: list[dict], plot_dir: Path) -> None:
    plt.figure(figsize=(11, 7))
    for config in configs:
        rows = [row for row in all_history if row["experiment"] == config["name"]]
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row["train_accuracy"] for row in rows], linestyle="--", label=f"{config['name']} train")
        plt.plot(epochs, [row["validation_accuracy"] for row in rows], label=f"{config['name']} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Training and Validation Accuracy Curves")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_dir / "accuracy_curves.png", dpi=160)
    plt.close()


def plot_final_accuracy_bar(final_rows: list[dict], plot_dir: Path) -> None:
    names = [row["Experiment"] for row in final_rows]
    train_acc = [float(row["Train Accuracy"]) for row in final_rows]
    val_acc = [float(row["Validation Accuracy"]) for row in final_rows]
    x_positions = list(range(len(names)))
    width = 0.38

    plt.figure(figsize=(11, 6))
    plt.bar([x - width / 2 for x in x_positions], train_acc, width=width, label="Train accuracy")
    plt.bar([x + width / 2 for x in x_positions], val_acc, width=width, label="Validation accuracy")
    plt.xticks(x_positions, names, rotation=25, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Final Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "final_accuracy_comparison.png", dpi=160)
    plt.close()


def plot_dataset_distribution(summary: dict, plot_dir: Path) -> None:
    labels = ["1", "2", "3", "4", "5"]
    train_counts = [summary["train_class_distribution"][label] for label in labels]
    val_counts = [summary["validation_class_distribution"][label] for label in labels]
    x_positions = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(8, 5))
    plt.bar([x - width / 2 for x in x_positions], train_counts, width=width, label="Train")
    plt.bar([x + width / 2 for x in x_positions], val_counts, width=width, label="Validation")
    plt.xticks(x_positions, [f"Position {label}" for label in labels])
    plt.ylabel("Number of images")
    plt.title("Class Distribution by Story Position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "class_distribution.png", dpi=160)
    plt.close()


def plot_confusion_matrices(predictions: list[dict], final_rows: list[dict], plot_dir: Path) -> None:
    best_experiment = max(final_rows, key=lambda row: float(row["Validation Accuracy"]))["Experiment"]
    selected = [
        item for item in predictions
        if item["experiment"] in {"Baseline CNN", best_experiment}
    ]

    for item in selected:
        matrix = confusion_matrix(item["y_true"], item["y_pred"], labels=[0, 1, 2, 3, 4])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["1", "2", "3", "4", "5"],
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Validation Confusion Matrix: {item['experiment']}")
        plt.tight_layout()
        safe_name = item["experiment"].lower().replace(" ", "_")
        plt.savefig(plot_dir / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close()


def write_outputs(
    all_history: list[dict],
    configs: list[dict],
    result_dir: Path,
    summary: dict,
    predictions: list[dict],
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_summary(summary, result_dir / "dataset_summary.json")

    metrics_path = result_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_history[0].keys()))
        writer.writeheader()
        writer.writerows(all_history)

    final_rows = []
    for config in configs:
        rows = [row for row in all_history if row["experiment"] == config["name"]]
        final = rows[-1]
        final_rows.append({
            "Experiment": config["name"],
            "Modification": config["modification"],
            "Train Loss": f"{final['train_loss']:.4f}",
            "Validation Loss": f"{final['validation_loss']:.4f}",
            "Train Accuracy": f"{final['train_accuracy']:.3f}",
            "Validation Accuracy": f"{final['validation_accuracy']:.3f}",
        })

    header = "| Experiment | Modification | Train Loss | Validation Loss | Train Accuracy | Validation Accuracy |"
    divider = "|---|---|---:|---:|---:|---:|"
    lines = [header, divider]
    for row in final_rows:
        lines.append(
            f"| {row['Experiment']} | {row['Modification']} | {row['Train Loss']} | "
            f"{row['Validation Loss']} | {row['Train Accuracy']} | {row['Validation Accuracy']} |"
        )
    (result_dir / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_loss_curves(all_history, configs, plot_dir)
    plot_accuracy_curves(all_history, configs, plot_dir)
    plot_final_accuracy_bar(final_rows, plot_dir)
    plot_dataset_distribution(summary, plot_dir)
    plot_confusion_matrices(predictions, final_rows, plot_dir)


def update_readme(readme_path: Path, table_path: Path) -> None:
    readme = readme_path.read_text(encoding="utf-8")
    table = table_path.read_text(encoding="utf-8")
    start = "<!-- RESULTS_TABLE_START -->"
    end = "<!-- RESULTS_TABLE_END -->"
    replacement = (
        f"{start}\n{table}\n"
        "![Loss curves](results/plots/loss_curves.png)\n\n"
        "![Accuracy curves](results/plots/accuracy_curves.png)\n\n"
        "![Final accuracy comparison](results/plots/final_accuracy_comparison.png)\n\n"
        "![Class distribution](results/plots/class_distribution.png)\n\n"
        "![Baseline confusion matrix](results/plots/confusion_matrix_baseline_cnn.png)\n"
        f"{end}"
    )
    before = readme.split(start)[0]
    after = readme.split(end)[1]
    readme_path.write_text(before + replacement + after, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = experiment_configs()
    all_history = []
    predictions = []
    final_summary = None

    for config in configs:
        history, summary, prediction = train_experiment(config, args.zip_path, args.epochs, args.batch_size, device)
        all_history.extend(history)
        predictions.append(prediction)
        final_summary = summary

    write_outputs(all_history, configs, ROOT / "results", final_summary, predictions)
    update_readme(ROOT / "README.md", ROOT / "results" / "results_table.md")
    print("\nSaved results to", ROOT / "results")


if __name__ == "__main__":
    main()


