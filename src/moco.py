import torch
import torchvision
import argparse
import wandb
import os
import copy

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule

from utils.utils import get_optimizer
from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description="MoCo Training and Linear Probing")
    parser.add_argument("--train_epochs", type=int, default=50, help="Number of epochs for MoCo training")
    parser.add_argument("--probe_epochs", type=int, default=50, help="Number of epochs for linear probing")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"], help="Backbone architecture")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--probe_batch_size", type=int, default=256, help="Batch size for linear probing")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"], help="Dataset to use")
    parser.add_argument("--track", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--lr", type=float, default=0.3, help="Learning rate for MoCo training")
    parser.add_argument("--probe_lr", type=float, default=0.05, help="Learning rate for linear probing")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop", "radam", "kron"], help="Optimizer to use")
    parser.add_argument("--memory_bank_size", type=int, default=4096, help="Size of the memory bank")
    parser.add_argument("--momentum_val", type=float, default=0.996, help="Base momentum value for EMA")
    return parser.parse_args()

class MoCo(torch.nn.Module):
    def __init__(self, backbone, feature_dim=512):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        self.projection_head = MoCoProjectionHead(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=128,
        )

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z
    
    def forward_momentum(self, x):
        features = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(features).detach()
        return z
    
    def get_features(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        return features


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=512, num_classes=100):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)


def main():
    args = parse_args()
    
    # Initialize wandb if requested
    if args.track:
        wandb.init(
            project=f"moco_{args.dataset}",
            config=vars(args),
            name=f"moco_{args.dataset}_{args.backbone}_{args.optimizer}"
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create backbone
    if args.backbone == "resnet18":
        resnet = torchvision.models.resnet18()
        feature_dim = 512
    elif args.backbone == "resnet34":
        resnet = torchvision.models.resnet34()
        feature_dim = 512
    elif args.backbone == "resnet50":
        resnet = torchvision.models.resnet50()
        feature_dim = 2048
    
    # Create backbone from resnet without the classification head
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    
    # Build the MoCo model
    model = MoCo(backbone, feature_dim=feature_dim)
    print(model)
    
    # Prepare transform that creates multiple random views for every image
    transform = MoCoV2Transform(input_size=32)
    
    # Create dataset
    if args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root='cifar10_data', train=True, download=True, transform=transform)
        num_classes = 10
    else:  # cifar100
        dataset = torchvision.datasets.CIFAR100(root='cifar100_data', train=True, download=True, transform=transform)
        num_classes = 100
    
    # Build a PyTorch dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Loss function
    criterion = NTXentLoss(memory_bank_size=(args.memory_bank_size, 128))
    
    # Optimizer
    optimizer = get_optimizer(args.optimizer)(model.parameters(), lr=args.lr)
    print(optimizer)
    
    # Train the model
    model = model.to(device)
    
    print(f"Starting MoCo training for {args.train_epochs} epochs...")
    for epoch in range(args.train_epochs):
        model.train()
        total_loss = 0.0
        momentum_val = cosine_schedule(epoch, args.train_epochs, args.momentum_val, 1)
        
        for batch in dataloader:
            x_query, x_key = batch[0]
            update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
            update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)
            
            x_query, x_key = x_query.to(device), x_key.to(device)
            query = model(x_query)
            key = model.forward_momentum(x_key)
            loss_val = criterion(query, key)
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            total_loss += loss_val.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.train_epochs}, Loss: {avg_loss:.5f}")
        
        if args.track:
            wandb.log({"train_loss": avg_loss, "epoch": epoch, "momentum": momentum_val})
    
    # Save the model
    # torch.save(model.state_dict(), f"moco_{args.backbone}_{args.dataset}.pt")
    
    # Linear probing evaluation
    print("Starting linear probing evaluation...")
    
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='cifar10_data', train=True, download=True, transform=transform_test)
        test_dataset = torchvision.datasets.CIFAR10(root='cifar10_data', train=False, download=True, transform=transform_test)
    else:  # cifar100
        train_dataset = torchvision.datasets.CIFAR100(root='cifar100_data', train=True, download=True, transform=transform_test)
        test_dataset = torchvision.datasets.CIFAR100(root='cifar100_data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.probe_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.probe_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Freeze the MoCo model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Create and train the linear classifier
    classifier = LinearClassifier(input_dim=feature_dim, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.probe_lr)
    
    best_test_acc = 0.0
    
    # Train the linear classifier
    for epoch in range(args.probe_epochs):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Extract features using the frozen backbone
            with torch.no_grad():
                features = model.get_features(inputs)
            
            # Train the linear classifier
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_accuracy = 100.0 * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on test set
        classifier.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Extract features
                features = model.get_features(inputs)
                
                # Forward pass through classifier
                outputs = classifier(features)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_accuracy = 100.0 * correct / total
        test_loss = test_loss / len(test_loader)
        
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
        
        # Log to wandb
        if args.track:
            wandb.log({
                "probe_train_loss": train_loss,
                "probe_train_acc": train_accuracy,
                "probe_test_loss": test_loss,
                "probe_test_acc": test_accuracy,
                "probe_epoch": epoch,
                "best_test_acc": best_test_acc
            })
        
        # Print results every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Linear Probe Epoch {epoch+1}/{args.probe_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
    
    print(f"Linear probing completed! Best test accuracy: {best_test_acc:.2f}%")
    
    if args.track:
        wandb.finish()

if __name__ == "__main__":
    main()