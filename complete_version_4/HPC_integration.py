import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def parallel_model_training(rank, world_size, model, train_loader, criterion, optimizer, num_epochs):
    # parallelization of training
    setup(rank, world_size)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    cleanup()


def model_parallelism(model, num_gpus): # equal to simple_model_parallelism (from prev. versions)
    # flag that we have only 1 GPU
    if num_gpus < 2:
        return model
    # otherwise split the model across GPUs
    devices = list(range(num_gpus))
    model = torch.nn.DataParallel(model, device_ids=devices)
    return model


def pipeline_parallelism(model, num_gpus):
    # ------------------------------------- parallelize pipeline--------------------------------------------
    pass


# compose these functions
def mixed_precision_training(model, optimizer):
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()


def train_step(images, labels, criterion, scaler, autocast, model, optimizer):
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss


def data_parallelism(model):
    # data parallelism to train the model on multiple GPUs, if possible
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism.")
        model = torch.nn.DataParallel(model)
    else:
        print("Data parallelism not available. Using single GPU.")
    return model


def objective(self, trial):  # Optuna optimization grids
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    self.train(optimizer)
    return self.best_loss


def hyperparameter_tuning_hpc(study, num_trials, num_nodes, node_rank):
    # applying HPC for hyperparameter tuning using Optuna on multiple nodes.
    if node_rank == 0: # only rank 0 node will run the study (others don't make sense)
        study.optimize(objective, n_trials=num_trials) 
        print("Best hyperparameters:", study.best_params)
        print("Best loss:", study.best_value)
    else:
        # wait for achieving rank = 0...
        print(f"Node {node_rank} waiting for hyperparameter tuning to complete.")


def hpc_deployment(model, deployment_nodes):
    # deploy the model across multiple nodes using HPC.
    model_path = "deployed_model.pth"
    torch.save(model.state_dict(), model_path)

    for node in deployment_nodes:
        # --------------------code to distribute model to different deployment nodes ----------------------------------
        print(f"Deploying model to node {node}.")
        os.system(f"scp {model_path} user@{node}:/path/to/deploy")

    print("Model deployment across nodes completed.")
