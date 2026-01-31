import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from multilingual_clip.pt_multilingual_clip import MultilingualCLIP
from transformers import AutoTokenizer


@dataclass
class KeyframeInput:
    path: Path
    video: str
    video_related_id: str


def load_keyframes():
    data_dir = Path("data", "keyframes")
    for video_grouped_dir in data_dir.iterdir():
        video = video_grouped_dir.name
        for image_path in video_grouped_dir.iterdir():
            image_name = image_path.name
            video_related_id = image_name.split(".")[0]
            print(f"embedding {image_path}")
            yield KeyframeInput(
                path=image_path, video=video, video_related_id=video_related_id
            )


def embedding_keyframes(preprocess, model, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained="laion400m_e32"
    )

    model.to(device)
    for keyframe_input in load_keyframes():
        image = Image.open(keyframe_input.path)
        image = preprocess(image).unsqueeze(0).to(device)  # type: ignore
        embeddings: torch.Tensor = model.encode_image(image)  # type: ignore
        embeddings_np = embeddings.numpy(force=True)
        npy_path = Path(
            "prepared-data",
            "embeddings",
            keyframe_input.video,
            f"{keyframe_input.video_related_id}.npy",
        )
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"saving {npy_path}")
        with open(
            npy_path,
            "wb",
        ) as f:
            np.save(f, embeddings_np, allow_pickle=False)


@dataclass
class ObjectInput:
    path: Path
    video: str
    video_related_id: str


def load_objects():
    data_dir = Path("data", "objects")
    for video_grouped_dir in data_dir.iterdir():
        video = video_grouped_dir.name
        for object_path in video_grouped_dir.iterdir():
            object_name = object_path.name
            video_related_id = object_name.split(".")[0]
            print(f"embedding {object_path}")
            yield ObjectInput(
                path=object_path, video=video, video_related_id=video_related_id
            )


@dataclass
class ObjectDetectionWithScore:
    class_entity: str
    score: float

    def pass_threshold(self, threshold: float):
        return self.score >= threshold


@dataclass
class DetectionFilteredObject:
    object_input: ObjectInput
    class_entities: list[str]
    scores: list[float]


def load_objects_to_embed():
    for object_input in load_objects():
        with open(object_input.path, "r") as f:
            object_data = json.load(f)
        object_det_scorings = object_data["detection_scores"]
        object_det_class_entites = object_data["detection_class_entities"]
        det_filtered_object = DetectionFilteredObject(
            object_input=object_input,
            class_entities=object_det_class_entites,
            scores=[float(x) for x in object_det_scorings],
        )

        yield det_filtered_object


def embedding_objects():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultilingualCLIP.from_pretrained(
        "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus"
    ).to(device)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-16Plus")
    for det_filtered_object in load_objects_to_embed():
        with torch.no_grad():
            embeddings: torch.Tensor = model.forward(
                det_filtered_object.class_entities, tokenizer
            ) * torch.tensor(det_filtered_object.scores).unsqueeze(-1)
            embeddings = embeddings.sum(dim=0)
        embeddings_np = embeddings.numpy(force=True)
        npy_path = Path(
            "prepared-data",
            "objects",
            det_filtered_object.object_input.video,
            f"{det_filtered_object.object_input.video_related_id}.npy",
        )
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"saving {npy_path}: {embeddings_np.shape}")
        with open(npy_path, "wb") as f:
            np.save(f, embeddings_np)


def main():
    embedding_objects()


if __name__ == "__main__":
    main()
