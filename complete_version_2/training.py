from pathlib import Path
import torch

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Saves the model and optimizer state to a checkpoint file.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)


class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, num_epochs, output_dir):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.model.device), labels.to(self.model.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}')

            # Save checkpoint
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(self.model, self.optimizer, epoch, checkpoint_path)
