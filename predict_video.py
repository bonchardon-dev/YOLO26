from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image as PILImage
from datasets import IterableDataset
from ultralytics import YOLO

from consts import YOLO_MODEL
from data.dataset import DataSetConstruction  


class YoloModelPredict:
    """
    Stream bird/non-bird images from Hugging Face, prepare a YOLO
    classification dataset, fine-tune the model, and run predictions.

    Labels:
        0 = bird
        1 = non_bird
    """

    def __init__(self, config_path: str = "data/data.yaml"):
        self.config_path = Path(config_path)

        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # YOLO_MODEL should be a checkpoint path/name, e.g.:
        # "yolo26n-cls.pt"
        self.model = YOLO(YOLO_MODEL)

    @staticmethod
    def _get_image(value) -> PILImage.Image:
        """Decode an image from a Hugging Face dataset sample."""
        if isinstance(value, PILImage.Image):
            return value

        if isinstance(value, dict):
            if value.get("bytes"):
                return PILImage.open(BytesIO(value["bytes"]))

            if value.get("path"):
                return PILImage.open(value["path"])

        if isinstance(value, bytes):
            return PILImage.open(BytesIO(value))

        raise TypeError(f"Unsupported image type: {type(value)}")

    def stream_dataset(self) -> Path:
        """
        Stream samples from Hugging Face and save them incrementally
        into the folders expected by Ultralytics classification.
        """
        root = Path(self.config["dataset_dir"])
        img_num = int(self.config["img_num_per_class"])
        train_ratio = float(self.config["train_ratio"])

        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        # Each class has its own train/validation counter.
        class_names = {0: "bird", 1: "non_bird"}
        train_limit = int(img_num * train_ratio)
        seen = {0: 0, 1: 0}

        dataset: IterableDataset = DataSetConstruction().dataset_balanced(
            img_num=img_num
        )

        root.mkdir(parents=True, exist_ok=True)

        for index, sample in enumerate(dataset):
            label = int(sample["label"])

            if label not in class_names:
                raise ValueError(f"Unexpected label: {label}")

            split = "train" if seen[label] < train_limit else "val"

            output_dir = root / split / class_names[label]
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{index:07d}.jpg"

            image = self._get_image(sample["image"])
            try:
                image.convert("RGB").save(
                    output_path,
                    format="JPEG",
                    quality=95,
                )
            finally:
                image.close()

            seen[label] += 1

            if (index + 1) % 500 == 0:
                print(f"Streamed and saved {index + 1} images")

        print("Dataset preparation complete:", root.resolve())
        print("Samples per class:", seen)

        return root

    def train_yolo_model(self) -> None:
        """Prepare streamed data and fine-tune the classifier."""
        dataset_dir = self.stream_dataset()

        results = self.model.train(
            data=str(dataset_dir),
            epochs=int(self.config["epochs"]),
            imgsz=int(self.config["imgsz"]),
            batch=int(self.config["batch"]),
            project="runs/bird_classifier",
            name="train",
            save=True,
        )

        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        print(f"Best weights saved at: {best_weights}")

        # Load the fine-tuned model for subsequent predictions.
        self.model = YOLO(str(best_weights))

    def streaming_pics(
        self,
        image_paths: list[str] | None = None,
    ) -> None:
        """Stream prediction results for a list of images."""
        image_paths = image_paths or ["croc.jpeg", "corc2.jpeg"]

        results = self.model.predict(
            source=image_paths,
            stream=True,
            imgsz=int(self.config["imgsz"]),
            save=True,
        )

        for result in results:
            if result.probs is not None:
                class_id = int(result.probs.top1)
                confidence = float(result.probs.top1conf)

                print(
                    f"{result.path}: "
                    f"{result.names[class_id]} "
                    f"({confidence:.2%})"
                )

    def process_video(self, input_video: str) -> None:
        """Classify video frames and save an annotated output video."""
        results = self.model.predict(
            source=input_video,
            stream=True,
            save=True,
            imgsz=int(self.config["imgsz"]),
        )

        for result in results:
            if result.probs is not None:
                class_id = int(result.probs.top1)
                print(
                    f"Frame: {result.path} | "
                    f"{result.names[class_id]}"
                )

        print("Video prediction finished. Check the runs directory.")


if __name__ == "__main__":
    predictor = YoloModelPredict()

    # Stream the Hugging Face dataset and fine-tune.
    predictor.train_yolo_model()

    # Classify images using the trained model.
    predictor.streaming_pics(["croc.jpeg", "corc2.jpeg"])

    # Classify an input video.
    # predictor.process_video("input.mp4")
