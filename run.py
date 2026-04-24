import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# ==================== Part 1: PrunableLinear Layer ====================

class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gates for each weight.
    Gates control whether weights are active (1) or pruned (0).
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Gate scores - one per weight, initialized to positive values
        # so gates start near 1 (active) after sigmoid
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features) * 2.0)
        
    def forward(self, x):
        """
        Forward pass with gated weights.
        Gates are computed via sigmoid(gate_scores), then multiplied element-wise
        with weights before the linear transformation.
        """
        # Transform gate_scores to [0, 1] range using sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gates to weights (element-wise multiplication)
        pruned_weights = self.weight * gates
        
        # Standard linear transformation using pruned weights
        # F.linear equivalent: x @ pruned_weights.T + bias
        output = torch.matmul(x, pruned_weights.t()) + self.bias
        
        return output
    
    def get_gates(self):
        """Return the current gate values (after sigmoid)."""
        return torch.sigmoid(self.gate_scores)


# ==================== Neural Network Definition ====================

class SelfPruningNet(nn.Module):
    """
    A simple CNN for CIFAR-10 using PrunableLinear layers.
    Architecture: Conv layers for feature extraction, then prunable FC layers.
    """
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        
        # Convolutional layers (not pruned for simplicity, but could be)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Prunable fully connected layers
        # After 2 poolings, 32x32 -> 8x8, so 64*8*8 = 4096 features
        self.fc1 = PrunableLinear(64 * 8 * 8, 512)
        self.fc2 = PrunableLinear(512, 128)
        self.fc3 = PrunableLinear(128, 10)  # 10 classes in CIFAR-10
        
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Prunable FC layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_all_gates(self):
        """Collect all gate values from all PrunableLinear layers."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates().flatten())
        return torch.cat(gates)


# ==================== Part 2: Sparsity Loss ====================

def compute_sparsity_loss(model):
    """
    Compute L1 regularization on all gates.
    This encourages gates to go to 0, pruning the network.
    """
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # L1 norm of gates (sum of absolute values)
            # Since gates are always positive (sigmoid output), this is just the sum
            gates = module.get_gates()
            sparsity_loss += gates.sum()
    
    return sparsity_loss


def compute_sparsity_level(model, threshold=1e-2):
    """
    Calculate percentage of weights that are pruned (gate < threshold).
    """
    total_gates = 0
    pruned_gates = 0
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates()
            total_gates += gates.numel()
            pruned_gates += (gates < threshold).sum().item()
    
    return (pruned_gates / total_gates) * 100 if total_gates > 0 else 0


# ==================== Part 3: Training and Evaluation ====================

def train_model(lambda_sparsity, num_epochs=50, device='cuda', verbose=True):
    """
    Train the self-pruning network with specified sparsity regularization.
    
    Args:
        lambda_sparsity: Weight of sparsity loss (higher = more pruning)
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
        verbose: Whether to print progress
    
    Returns:
        model, test_accuracy, sparsity_level, gate_values
    """
    
    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Model setup
    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_sp_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Classification loss
            cls_loss = criterion(outputs, labels)
            
            # Sparsity loss (L1 on gates)
            sp_loss = compute_sparsity_loss(model)
            
            # Total loss
            total_loss = cls_loss + lambda_sparsity * sp_loss
            
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_sp_loss += sp_loss.item()
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(trainloader)
            avg_cls = running_cls_loss / len(trainloader)
            avg_sp = running_sp_loss / len(trainloader)
            sparsity = compute_sparsity_level(model)
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} '
                  f'(Cls: {avg_cls:.4f}, Sp: {avg_sp:.4f}) '
                  f'Sparsity: {sparsity:.2f}%')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    sparsity_level = compute_sparsity_level(model)
    gate_values = model.get_all_gates().cpu().detach().numpy()
    
    if verbose:
        print(f'\n=== Final Results (λ={lambda_sparsity}) ===')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        print(f'Sparsity Level: {sparsity_level:.2f}%')
        print(f'Active Weights: {100-sparsity_level:.2f}%\n')
    
    return model, test_accuracy, sparsity_level, gate_values


# ==================== Main Experiment ====================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Test different lambda values
    lambda_values = [0.0001, 0.001, 0.01]
    results = []
    
    for lam in lambda_values:
        print(f'Training with λ = {lam}')
        print('=' * 50)
        model, acc, sparsity, gates = train_model(lam, num_epochs=50, device=device)
        results.append({
            'lambda': lam,
            'accuracy': acc,
            'sparsity': sparsity,
            'gates': gates,
            'model': model
        })
    
    # Print summary table
    print('\n' + '=' * 60)
    print('SUMMARY TABLE')
    print('=' * 60)
    print(f'{"Lambda":<12} {"Test Accuracy":<20} {"Sparsity Level (%)":<20}')
    print('-' * 60)
    for r in results:
        print(f'{r["lambda"]:<12} {r["accuracy"]:<20.2f} {r["sparsity"]:<20.2f}')
    print('=' * 60)
    
    # Plot gate distribution for best model (middle lambda usually best tradeoff)
    best_result = results[1]  # Medium lambda
    
    plt.figure(figsize=(10, 6))
    plt.hist(best_result['gates'], bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Gate Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Gate Values (λ={best_result["lambda"]})', fontsize=14)
    plt.axvline(x=0.01, color='red', linestyle='--', label='Pruning Threshold (0.01)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gate_distribution.png', dpi=300, bbox_inches='tight')
    print('\nGate distribution plot saved as gate_distribution.png')