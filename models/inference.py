import base64
import io
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

INPUT_SIZE = 120
KERNEL_SIZE = 3
DROPOUT = 0.3

class PokemonCNN(nn.Module):
    def __init__(self, num_classes: int, kernel_size: int = KERNEL_SIZE, dropout: float = DROPOUT):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15 * 15 * 128, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def _decode_base64_image(b64_str: str, size: Tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB").resize(size)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def _ensure_nchw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
        if arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.ndim == 4 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (0, 3, 1, 2))
    return arr

def _infer_num_classes(state_dict: Dict[str, torch.Tensor]) -> int:
    for key in reversed(list(state_dict.keys())):
        if key.endswith(".weight") and state_dict[key].ndim == 2:
            return state_dict[key].shape[0]
    raise KeyError("No se pudo inferir num_classes")

def model_fn(model_dir: str) -> nn.Module:
    model_path = os.path.join(model_dir, "pokemon_cnn.pt")
    state = torch.load(model_path, map_location="cpu")
    num_classes = _infer_num_classes(state)
    model = PokemonCNN(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def input_fn(request_body: str, content_type: str) -> torch.Tensor:
    if content_type != "application/json":
        raise ValueError(f"content_type no soportado: {content_type}")

    payload = json.loads(request_body)

    if "image" in payload:
        data = _decode_base64_image(payload["image"])
    elif "images" in payload:
        data = np.stack([_decode_base64_image(b) for b in payload["images"]], axis=0)
    else:
        raise ValueError("El JSON debe contener 'image' o 'images'")

    data = _ensure_nchw(data)
    return torch.tensor(data, dtype=torch.float32)

def predict_fn(input_data: torch.Tensor, model: nn.Module) -> List[Dict[str, Any]]:
    with torch.no_grad():
        logits = model(input_data)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idx = torch.max(probs, dim=1)

    results = []
    for cls_id, prob in zip(top_idx.tolist(), top_probs.tolist()):
        results.append({"class_id": int(cls_id), "prob": float(prob)})
    return results

def output_fn(prediction: List[Dict[str, Any]], accept: str) -> str:
    if accept != "application/json":
        raise ValueError(f"accept no soportado: {accept}")
    return json.dumps({"predictions": prediction})
