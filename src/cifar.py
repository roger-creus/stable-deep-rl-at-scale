import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

from utils.utils import (
    parse_mlp_depth, 
    parse_mlp_width, 
    get_optimizer, 
    get_grad_norms, 
    get_weight_norms,
    get_grad_cosine,
    get_dormant_neurons
)

import csv
import os
import wandb

def get_mlp(mlp_type, input_size, mlp_width, mlp_depth, use_ln=False, device="cuda"):
    if mlp_type == "default":
        from models.mlp import MLP as MLPClass
    elif mlp_type == "residual":
        from models.mlp import ResidualMLP as MLPClass
    elif mlp_type == "multiskip_residual":
        from models.mlp import MultiSkipResidualMLP as MLPClass
    else:
        raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")
    
    trunk_hidden_size = parse_mlp_width(mlp_width)
    trunk_num_layers = parse_mlp_depth(mlp_depth, mlp_type)
    
    mlp = MLPClass(
        input_size=int(input_size),
        hidden_size=trunk_hidden_size,
        output_size=512,
        num_layers=trunk_num_layers,
        use_ln=use_ln,
        activation_fn="relu",
        device=device,
        last_act=True
    )
    return mlp

class NonStationaryDataset(Dataset):
    def __init__(self, dataset, permutation=None):
        self.dataset = dataset
        self.permutation = permutation if permutation is not None else list(range(len(dataset.targets)))
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        label = self.dataset.targets[self.permutation[idx]]
        return image, label

    def reshuffle_labels(self):
        n = len(self.dataset.targets)
        self.permutation = np.random.permutation(n).tolist()

def train_cifar(run_name, mlp_type, optimizer_name, mlp_depth, mlp_width, dataset_name="cifar100", non_stationary=False, epochs=25, batch_size=128, use_ln=False, device="cuda"): 
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb.
    wandb.init(
        project=f"{dataset_name}_good",
        name=run_name,
        config={
            "mlp_type": mlp_type,
            "optimizer": optimizer_name,
            "mlp_depth": mlp_depth,
            "mlp_width": mlp_width,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.00025,
            "dataset": dataset_name,
            "non_stationary": non_stationary,
            "use_ln": use_ln
        }
    )
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    dataset_class = torchvision.datasets.CIFAR100 if dataset_name == "cifar100" else torchvision.datasets.CIFAR10
    num_classes = 100 if dataset_name == "cifar100" else 10
    num_datapoints = 10000
    
    trainset_base = dataset_class(root=f'./{dataset_name}_data', train=True, download=True, transform=transform)
    testset_base = dataset_class(root=f'./{dataset_name}_data', train=False, download=True, transform=transform)
    
    # Create indices for subsetting
    train_indices = list(range(num_datapoints))
    test_indices = list(range(num_datapoints))
    
    # Subset the targets/labels directly
    trainset_base.targets = [trainset_base.targets[i] for i in train_indices]
    testset_base.targets = [testset_base.targets[i] for i in test_indices]
    
    # Subset the data
    if hasattr(trainset_base, 'data'):
        trainset_base.data = trainset_base.data[train_indices]
        testset_base.data = testset_base.data[test_indices]
    else:
        trainset_base.imgs = [trainset_base.imgs[i] for i in train_indices]
        testset_base.imgs = [testset_base.imgs[i] for i in test_indices]
    
    if non_stationary:
        trainset = NonStationaryDataset(trainset_base)
        testset = NonStationaryDataset(testset_base)
    else:
        trainset = trainset_base
        testset = testset_base
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    mlp_input_size = 2048
    mlp = get_mlp(mlp_type, input_size=mlp_input_size, mlp_width=mlp_width, mlp_depth=mlp_depth, use_ln=use_ln, device=device)
    
                
    class CIFARClassifier(nn.Module):
        def __init__(self, mlp, num_classes, use_ln=False):
            super().__init__()
            
            layers = []
            layers.extend([
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ])
            if use_ln:
                layers.append(nn.GroupNorm(8, 64))
            layers.extend([
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2),
            ])

            layers.extend([
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ])
            if use_ln:
                layers.append(nn.GroupNorm(16, 128))
            layers.extend([
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2),
            ])

            layers.extend([
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ])
            if use_ln:
                layers.append(nn.GroupNorm(16, 128))
            layers.extend([
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2),
            ])

            self.cnn = nn.Sequential(*layers)
            
            self.network = nn.Sequential(
                self.cnn,
                nn.Flatten(),
            )
            
            self.trunk = mlp
            self.classifier = nn.Linear(512, num_classes)
        
        def forward(self, x):
            features = self.network(x)
            features = self.trunk(features)
            return self.classifier(features)
    
    model = CIFARClassifier(mlp, num_classes, use_ln=use_ln).to(device)
    
    if not hasattr(model, "prev_grad_dirs"):
        model.prev_grad_dirs = {}
    
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name)(model.parameters(), lr=0.00025)
    
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    
    change_points = [20, 40, 60, 80] if non_stationary else []
    
    for epoch in range(epochs):
        if non_stationary and epoch in change_points:
            trainset.reshuffle_labels()
            testset.reshuffle_labels()
            print(f"Reshuffling labels at epoch {epoch}")
            
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        aggregated_grad_norms = {}
        aggregated_grad_cosines = {}
        aggregated_weight_norms = {}
        aggregated_dormant_neurons = {}
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            batch_grad_norms = get_grad_norms(model, use_ln=use_ln)
            batch_grad_cosine = get_grad_cosine(model, use_ln=use_ln)
            batch_weight_norms = get_weight_norms(model, use_ln=use_ln)
            batch_dormant_neurons = get_dormant_neurons(model, images, use_ln=use_ln)
            
            for key, value in batch_grad_norms.items():
                aggregated_grad_norms.setdefault(key, []).append(value)
            for key, value in batch_grad_cosine.items():
                aggregated_grad_cosines.setdefault(key, []).append(value if value is not None else 0)
            for key, value in batch_weight_norms.items():
                aggregated_weight_norms.setdefault(key, []).append(value)
            for key, value in batch_dormant_neurons.items():
                aggregated_dormant_neurons.setdefault(key, []).append(value)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        train_loss = total_loss / len(trainloader)
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        test_loss = total_loss / len(testloader)
        test_acc = correct / total
        print(f"Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        
        avg_grad_norms = {k: sum(v_list)/len(v_list) for k, v_list in aggregated_grad_norms.items()}
        avg_grad_cosines = {k: sum(v_list)/len(v_list) for k, v_list in aggregated_grad_cosines.items()}
        avg_weight_norms = {k: sum(v_list)/len(v_list) for k, v_list in aggregated_weight_norms.items()}
        avg_dormant_neurons = {k: sum(v_list)/len(v_list) for k, v_list in aggregated_dormant_neurons.items()}
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        
        for key, value in avg_grad_norms.items():
            log_dict[f"grad_norms/{key}"] = value
        for key, value in avg_grad_cosines.items():
            log_dict[f"grad_cosines/{key}"] = value
        for key, value in avg_weight_norms.items():
            log_dict[f"weight_norms/{key}"] = value
        for key, value in avg_dormant_neurons.items():
            log_dict[f"dormant_neurons/{key}"] = value
            
        wandb.log(log_dict, step=epoch)
    
    last_n = min(3, len(train_loss_hist))
    avg_train_loss = sum(train_loss_hist[-last_n:]) / last_n
    avg_test_loss = sum(test_loss_hist[-last_n:]) / last_n
    avg_train_acc = sum(train_acc_hist[-last_n:]) / last_n
    avg_test_acc = sum(test_acc_hist[-last_n:]) / last_n

    csv_filename = f"{run_name}.csv"
    csv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
    
    with open(csv_filepath, "w", newline="") as csvfile:
         fieldnames = ["mlp_type", "mlp_depth", "optimizer", "dataset", "non_stationary", "train_loss", "test_loss", "train_acc", "test_acc"]
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
         writer.writeheader()
         writer.writerow({
              "mlp_type": mlp_type,
              "mlp_depth": mlp_depth,
              "optimizer": optimizer_name,
              "dataset": dataset_name,
              "non_stationary": non_stationary,
              "train_loss": avg_train_loss,
              "test_loss": avg_test_loss,
              "train_acc": avg_train_acc,
              "test_acc": avg_test_acc
         })
    print(f"Logged experiment results to {csv_filepath}")
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_type", type=str, required=True)
    parser.add_argument("--mlp_depth", type=str, required=True)
    parser.add_argument("--mlp_width", type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--non_stationary", action="store_true", help="Make the problem non-stationary by shuffling labels")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--use_ln", action="store_true", help="Use layer normalization in both CNN and MLP")
    args = parser.parse_args()
    
    if args.optimizer == "kron":
        args.lr /= 3.0
    
    run_name = f"DATA:{args.dataset}_MLP.TYPE:{args.mlp_type}_MLP.DEPTH:{args.mlp_depth}_MLP.WIDTH:{args.mlp_width}_OPTIM:{args.optimizer}_NS:{args.non_stationary}_LN:{args.use_ln}"
    train_cifar(
        run_name,
        args.mlp_type,
        args.optimizer,
        args.mlp_depth,
        args.mlp_width,
        dataset_name=args.dataset,
        non_stationary=args.non_stationary, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        use_ln=args.use_ln)
