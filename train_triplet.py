import argparse
import os
from glob import glob

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTModel

from prepare_triplet_data import create_dataset


class TripletDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.anchor_paths = sorted(glob(os.path.join(root_dir, "*_anchor_*.png")))
        self.triplets = self._create_triplets()

    def _create_triplets(self):
        triplets = []
        for anchor_path in self.anchor_paths:
            base_name = "_".join(anchor_path.split("_")[:-2])
            positive_path = anchor_path.replace("anchor", "positive")
            negative_paths = glob(
                os.path.join(
                    self.root_dir, f"*_negative_*.png"
                )
            )
            # Ensure negative is not from the same video
            anchor_video_name = os.path.basename(base_name).split('_')[0]
            valid_negative_paths = [p for p in negative_paths if not os.path.basename(p).startswith(anchor_video_name)]

            if os.path.exists(positive_path) and valid_negative_paths:
                negative_path = valid_negative_paths[0] # Simplified selection
                triplets.append((anchor_path, positive_path, negative_path))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        anchor_processed = self.processor(
            images=anchor_img, return_tensors="pt"
        ).pixel_values.squeeze(0)
        positive_processed = self.processor(
            images=positive_img, return_tensors="pt"
        ).pixel_values.squeeze(0)
        negative_processed = self.processor(
            images=negative_img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        return anchor_processed, positive_processed, negative_processed


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 준비
    create_dataset()

    # 모델 및 프로세서 로드
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)

    # 데이터셋 및 데이터로더
    dataset = TripletDataset(root_dir="triplet_dataset", processor=processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 손실 함수 및 옵티마이저
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 학습 루프
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            optimizer.zero_grad()

            anchor_output = model(pixel_values=anchor).last_hidden_state.mean(dim=1)
            positive_output = model(pixel_values=positive).last_hidden_state.mean(dim=1)
            negative_output = model(pixel_values=negative).last_hidden_state.mean(dim=1)

            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Loss: {running_loss / len(dataloader)}"
        )

    # 모델 저장
    torch.save(model.state_dict(), "finetuned_vit.pth")
    print("Fine-tuned model saved as finetuned_vit.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    args = parser.parse_args()
    train(args)