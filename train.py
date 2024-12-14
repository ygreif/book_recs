import ray
import torch
from data import data
from models import concatmodel
from torch.utils.data import DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler

dset = data.BookDataset('./data/medium.csv')
train_dset, test_dset, val_dset = data.split_data(dset)
print("Num books", dset.num_books, "Num users", dset.num_users)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

USE_RAY = False

def epoch_loss(model, dataloader):
    loss = 0
    total_samples = 0
    for i, (features, target) in enumerate(dataloader):
        book, user, _ = features
        user, book, target = user.to(DEVICE), book.to(DEVICE), target.to(DEVICE)
        prediction = model(user, book)
        loss += (prediction.squeeze() - target.float()).pow(2).sum()
        total_samples += len(target)
    return loss / total_samples

def sanity_checks(model, dataloader):
    min_prediction = float('inf')
    max_prediction = float('-inf')
    min_target = float('inf')
    max_target = float('-inf')

    for i, (features, target) in enumerate(dataloader):
        book, user, _ = features
        user, book, target = user.to(DEVICE), book.to(DEVICE), target.to(DEVICE)
        prediction = model(user, book)
        min_prediction = min(min_prediction, prediction.min().item())
        max_prediction = max(max_prediction, prediction.max().item())
        min_target = min(min_target, target.min().item())
        max_target = max(max_target, target.max().item())


    print(f"Min prediction: {min_prediction} Max prediction: {max_prediction}")
    print(f"Min target: {min_target} Max target: {max_target}")
    print(f"Gradiant norm for user embeddings: {model.user_embeddings.weight.grad.norm()}")
    print(f"Gradiant norm for book embeddings: {model.book_embeddings.weight.grad.norm()}")


def train_by_params(hidden_layers=[64, 32], batch_size=2, embedding_size=64, lr=0.001, epochs=10):
    config = {
        'hidden_layers': hidden_layers,
        'batch_size': batch_size,
        'embedding_size': embedding_size,
        'lr': lr,
        'epochs': epochs
    }
    train_by_config(config)

def train_by_config(config):
    hidden_layers = config['hidden_layers']
    batch_size = config['batch_size']
    embedding_size = config['embedding_size']
    lr = config['lr']
    epochs = config['epochs']

    model = concatmodel.ConcatModel(dset.num_users, dset.num_books, embedding_size, hidden_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    loss_fn = torch.nn.MSELoss()
    train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=True)

    train_loss = epoch_loss(model, train_dataloader)
    test_loss = epoch_loss(model, test_dataloader)
    if USE_RAY:
        ray.train.report({'train_loss':train_loss.item(), 'test_loss':test_loss.item()})
    else:
        print(f"Initial Train Loss: {train_loss} Test Loss: {test_loss}")

    #ray.train.report(train_loss=train_loss.item(), test_loss=test_loss.item())
    for epoch in range(epochs):
        for i, (features, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            book, user, _ = features
            user, book, target = user.to(DEVICE), book.to(DEVICE), target.to(DEVICE)
            prediction = model(user, book)
            loss = loss_fn(prediction.squeeze(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss = epoch_loss(model, train_dataloader)
        test_loss = epoch_loss(model, test_dataloader)
        if USE_RAY:
            ray.train.report({'train_loss':train_loss.item(), 'test_loss':test_loss.item()})
        else:
            print(f"Epoch {epoch} Train Loss: {epoch_loss(model, train_dataloader)} Test Loss: {epoch_loss(model, test_dataloader)}")
            sanity_checks(model, train_dataloader)
    return model

def run_ray():
    config = {
        'hidden_layers': tune.choice([ [512, 256], [128], [128, 128] ]),
        'batch_size': tune.choice([1, 2, 4, 8, 16]),
        'embedding_size': tune.choice([128, 256, 512]),
        'lr': tune.loguniform(1e-4, 1e-1),
        'epochs': 10
    }

    scheduler = ASHAScheduler(
        metric="train_loss",
        mode="min",
        max_t=10,
        grace_period=3,
        reduction_factor=2)

    analysis = tune.run(
        train_by_config,
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=tune.CLIReporter(
            metric_columns=["train_loss", "test_loss"],
            parameter_columns=["hidden_layers", "batch_size", "embedding_size", "lr", "epochs"]),
        resources_per_trial={"cpu": 8, "gpu": 0 if torch.cuda.is_available() else 0}
    )

    print("Best config: ", analysis.get_best_config(metric="train_loss"))
    print("Best result: ", analysis.get_best_result(metric="train_loss"))
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    if USE_RAY:
        run_ray()
    else:
        train_by_params()
        import pdb;pdb.set_trace()
