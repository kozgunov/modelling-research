from pathlib import Path
import torch
from pathlib import Path
import torch
import optuna
from optuna.trial import TrialState

def save_checkpoint(model, optimizer, epoch, filepath): # save the model and optimizer state to the checkpoint file.
    torch.save(
    {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)
    print("model saved successfully")


class Trainer:
    def __init__(self, model, train_loader, criterion,optimizer, num_epochs, output_dir):
        self.model = model.get_model()
        self.device = model.get_device()
        self.train_loader = train_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
        self.best_epoch = -1


    def train(self, optimizer):
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad() # zero the parameter gradients

                outputs = self.model(images) # forward pass
                loss = self.criterion(outputs, labels)

                loss.backward()  # backward pass
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}')

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = epoch
                checkpoint_path = self.output_dir / f'best_checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(self.model, optimizer, epoch, checkpoint_path)
            elif epoch - self.best_epoch > 2:  # save checkpoint if divergence detected
                print("Divergence detected. Saving model checkpoint.")
                checkpoint_path = self.output_dir / f'divergence_checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(self.model, optimizer, epoch, checkpoint_path)


    def objective(self, trial): # Optuna optimization
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train(optimizer)
        return self.best_loss

    def optimize(self, n_trials=10):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", study.best_params)
        print("Best loss:", study.best_value)
