import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import CrohnsDataset
from torchvision import transforms
from preprocess import preprocess_data
from model import aap, feature_max, CombinedModel
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import logging
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('/root/autodl-fs/ASR/bert-base-uncased')
bert_model = BertModel.from_pretrained('/root/autodl-fs/ASR/bert-base-uncased').to(device)

def extract_text_features(text_info):
    inputs = tokenizer(text_info, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_features = outputs.last_hidden_state[:, 0, :]
    return cls_features

def plot_roc_curve(labels, outputs, phase, epoch):
    fpr, tpr, _ = roc_curve(labels, outputs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{phase} ROC Curve - Epoch {epoch+1}')
    plt.legend(loc="lower right")
    save_path = f'./log6_long_test/{phase}_roc_curve_epoch_{epoch+1}.png'
    plt.savefig(save_path)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='CD Classification')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--image_per_model', type=int, default=3)
    parser.add_argument('--split_seed', type=int, default=42)
    return parser.parse_args()

def main(args):
    if not os.path.exists('log'):
        os.makedirs('log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler('log/log.txt', 'w'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # =========================
    # Data augmentation (train only)
    # =========================
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # =========================
    # Split indices reproducibly
    # =========================
    tmp_dataset = CrohnsDataset(data_dir=args.data_dir, transform=test_transforms)
    n_total = len(tmp_dataset)

    train_size = int(0.8 * n_total)
    test_size = n_total - train_size
    print("train_size: ", train_size)
    print("test_size: ", test_size)

    g = torch.Generator()
    g.manual_seed(args.split_seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_indices = perm[:train_size]
    test_indices = perm[train_size:]

    train_dataset = Subset(CrohnsDataset(data_dir=args.data_dir, transform=train_transforms), train_indices)
    test_dataset  = Subset(CrohnsDataset(data_dir=args.data_dir, transform=test_transforms),  test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=False)

    model = CombinedModel(num_classes=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    alpha = 0.25
    gamma = 0.25

    log_directory = "./log"
    os.makedirs(log_directory, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        predict_file = f"{log_directory}/predict_{epoch}.xlsx"
        if os.path.exists(predict_file):
            os.remove(predict_file)

    # =========================
    # ✅ Train loop
    # =========================
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        running_loss = 0.0

        all_labels = []
        all_probs = []
        train_results = []

        for images, labels, person_name, text_info in train_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1)

            images = images.transpose(1, 2)

            text_features = extract_text_features(text_info).to(device)

            output_1, output_2, output_3, output_4 = model(images, text_features)  # logits


            loss_1 = criterion(output_1, labels)
            loss_2 = criterion(output_2, labels)
            loss_3 = criterion(output_3, labels)
            loss_4 = criterion(output_4, labels)

            loss = alpha * loss_4 + gamma * (loss_1 + loss_2 + loss_3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probs_4 = torch.softmax(output_4.detach(), dim=1)  # [N,2]

            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs_4.cpu().numpy())

            for name, prob in zip(person_name, probs_4.cpu().numpy()):
                train_results.append({"Patient Name": name, "Predicted Probabilities": list(prob)})

        pd.DataFrame(train_results).to_excel(
            f"./log/train_results_epoch_{epoch + 1}.xlsx", index=False
        )

        all_probs = torch.tensor(all_probs)  # shape [N,2]
        predicted = torch.argmax(all_probs, dim=1)

        accuracy = accuracy_score(all_labels, predicted.cpu().numpy())
        auc_score = roc_auc_score(all_labels, all_probs[:, 1].cpu().numpy())
        precision = precision_score(all_labels, predicted.cpu().numpy(), zero_division=0)
        recall = recall_score(all_labels, predicted.cpu().numpy(), zero_division=0)
        f1 = f1_score(all_labels, predicted.cpu().numpy(), zero_division=0)

        plot_roc_curve(all_labels, all_probs[:, 1].cpu().numpy(), "Train", epoch)

        logger.info(
            f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
            f'Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, '
            f'Recall: {recall:.4f}, F1 Score: {f1:.4f}'
        )

        # =========================
        # ✅ Eval loop
        # =========================
        model.eval()
        test_loss = 0.0

        all_test_labels = []
        all_test_probs = []
        test_results = []

        with torch.no_grad():
            for images, labels, person_name, text_info in test_loader:
                images = images.to(device)
                labels = labels.to(device).view(-1)

                images = images.transpose(1, 2)

                text_features = extract_text_features(text_info).to(device)

                output_1, output_2, output_3, output_4 = model(images, text_features)  # logits

                # ✅ loss 用 logits
                loss = criterion(output_4, labels)
                test_loss += loss.item()

                # ✅ 指标/记录用概率
                probs_4 = torch.softmax(output_4, dim=1)

                all_test_labels.extend(labels.cpu().numpy())
                all_test_probs.extend(probs_4.cpu().numpy())

                for name, prob in zip(person_name, probs_4.cpu().numpy()):
                    test_results.append({"Patient Name": name, "Predicted Probabilities": list(prob)})

        pd.DataFrame(test_results).to_excel(
            f"./log/test_results_epoch_{epoch + 1}.xlsx", index=False
        )

        all_test_probs = torch.tensor(all_test_probs)
        predicted = torch.argmax(all_test_probs, dim=1)

        test_accuracy = accuracy_score(all_test_labels, predicted.cpu().numpy())
        test_auc = roc_auc_score(all_test_labels, all_test_probs[:, 1].cpu().numpy())
        test_precision = precision_score(all_test_labels, predicted.cpu().numpy(), zero_division=0)
        test_recall = recall_score(all_test_labels, predicted.cpu().numpy(), zero_division=0)
        test_f1 = f1_score(all_test_labels, predicted.cpu().numpy(), zero_division=0)

        model_save_dir = ""
        os.makedirs(model_save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{model_save_dir}/model_epoch_{epoch + 1}.pth")

        plot_roc_curve(all_test_labels, all_test_probs[:, 1].cpu().numpy(), "Validation", epoch)

        logger.info(
            f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.4f}, '
            f'Test AUC: {test_auc:.4f}, Test Precision: {test_precision:.4f}, '
            f'Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}'
        )

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists('data_processed'):
        print("data_processed...")
        preprocess_data('data', 'data_processed')
    main(args)
