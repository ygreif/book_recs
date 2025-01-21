import argparse

# import torch.cuda.amp as amp
import ray
from scipy import sparse
import torch
from data import data
from models import concatmodel
from torch.utils.data import DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import perf

#DEVICE = torch.device("cpu")

#USE_RAY = False

@perf.timing_decorator
def epoch_loss(model, dataloader):
    # Turn off backward pass
    with torch.no_grad():
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


def train_by_params(device, use_ray, hidden_layers=[10], batch_size=64, embedding_size=8, lr=0.001, epochs=5, weight_decay=0.01, dropout_prob=0.5, train_dset=False, test_dset=False, train_dset_id=False, test_dset_id=False):
    config = {
        'hidden_layers': hidden_layers,
        'batch_size': batch_size,
        'embedding_size': embedding_size,
        'lr': lr,
        'epochs': epochs,
        'weight_decay': weight_decay,
        'dropout_prob': dropout_prob,

        'train_dset': train_dset,
        'test_dset': test_dset,
        'train_dset_id': train_dset_id,
        'test_dset_id': test_dset_id,

        'device': device,
        'use_ray': use_ray,
        'sparse': sparse,

        'num_users': dset.num_users,
        'num_books': dset.num_books
    }
    train_by_config(config)

def train_by_config(config):
    hidden_layers = config['hidden_layers']
    batch_size = config['batch_size']
    embedding_size = config['embedding_size']
    lr = config['lr']
    epochs = config['epochs']
    weight_decay = config.get("weight_decay", 0.0)

    USE_RAY = config['use_ray']
    DEVICE = config['device']
    sparse = config['sparse']

    num_users = config['num_users']
    num_books = config['num_books']

    model = concatmodel.ConcatModel(num_users, num_books, embedding_size, hidden_layers, dropout_prob=config['dropout_prob'], sparse=sparse).to(DEVICE)
    if not sparse:
        optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)]
    else:
        optimizers = [
            torch.optim.SparseAdam(model.sparse_parameters(), lr=lr),
            torch.optim.Adam(model.dense_parameters(), lr=lr, weight_decay=weight_decay)
        ]
    if USE_RAY:
        train_dset = ray.get(config['train_dset_id'])
        test_dset = ray.get(config['test_dset_id'])
    else:
        train_dset = config['train_dset']
        test_dset = config['test_dset']

    loss_fn = torch.nn.MSELoss()
    train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    print("Calculating intiial loss")
    train_loss = epoch_loss(model, train_dataloader)
    test_loss = epoch_loss(model, test_dataloader)
    if USE_RAY:
        ray.train.report({'train_loss':train_loss.item(), 'test_loss':test_loss.item()})
    else:
        print(f"Initial Train Loss: {train_loss} Test Loss: {test_loss}")

    #ray.train.report(train_loss=train_loss.item(), test_loss=test_loss.item())
    for epoch in range(epochs):
        with perf.time_block("Epoch:"):
            for i, (features, target) in enumerate(train_dataloader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                book, user, _ = features
                user, book, target = user.to(DEVICE), book.to(DEVICE), target.to(DEVICE)
                prediction = model(user, book)
                loss = loss_fn(prediction.squeeze(), target.float())
                sparse_l2_penalty = model.sparse_l2_penalty(user, book, weight_decay)
                loss += weight_decay * sparse_l2_penalty
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

        train_loss = epoch_loss(model, train_dataloader)
        test_loss = epoch_loss(model, test_dataloader)
        if USE_RAY:
            ray.train.report({'train_loss':train_loss.item(), 'test_loss':test_loss.item()})
        else:
            print(f"Epoch {epoch} Train Loss: {epoch_loss(model, train_dataloader)} Test Loss: {epoch_loss(model, test_dataloader)}")
            #with torch.no_grad():
            #    sanity_checks(model, train_dataloader)
    return model

def generate_hidden_layers(embedding_size, structure):
    if structure == 'one':
        return [embedding_size * 2]
    elif structure == 'one_large':
        return [embedding_size * 4]
    elif structure == 'two':
        return [embedding_size * 2, embedding_size]
    elif structure == 'two_large':
        return [embedding_size * 4, embedding_size * 2]
    else:
        return [embedding_size * 3, embedding_size * 2, embedding_size]


def run_ray(device, train_dset_id, test_dset_id):
    config = {
        'embedding_size': tune.choice([32, 64, 128, 256]),
        'structure': tune.choice(['three', 'two', 'two_large', 'one', 'one_large']),
        'hidden_layers': tune.sample_from(lambda spec: generate_hidden_layers(spec.config.embedding_size, spec.config.structure)),
#        'hidden_layers': tune.choice([[512, 256], [128], [128, 128] ]),
        'batch_size': tune.choice([1, 2, 4, 8, 16]),

        'dropout_prob': tune.choice([0, 0.5]),
        'weight_decay': tune.loguniform(1e-4, 1e-1),
        #'embedding_size': tune.choice([32]),
        'lr': tune.loguniform(1e-4, 1e-1),
        'epochs': 10,
        'train_dset_id': train_dset_id,
        'test_dset_id': test_dset_id,

        'num_users': dset.num_users,
        'num_books': dset.num_books,

        'device': device,
        'use_ray': True,
        'sparse': True
    }

    scheduler = ASHAScheduler(
        metric="train_loss",
        mode="min",
        max_t=10,
        grace_period=3,
        reduction_factor=2)

    # random is the default algo

    analysis = tune.run(
        train_by_config,
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=tune.CLIReporter(
            metric_columns=["train_loss", "test_loss"],
            parameter_columns=["hidden_layers", "batch_size", "embedding_size", "lr", "epochs"]),
        resources_per_trial={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0}
    )

    print("Best config: ", analysis.get_best_config(metric="train_loss"))
    print("Best result: ", analysis.get_best_result(metric="train_loss"))
    import pdb;pdb.set_trace()

if __name__ == '__main__':

    import inspect
    closure_vars = inspect.getclosurevars(train_by_config)
    # this print shows an empty dictionary
    print(closure_vars.nonlocals)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ray", action='store_true')
    parser.add_argument("--infile", default='./data/Books_rating_encoded.csv')
    parser.add_argument("--sparse", default=True)

    args = parser.parse_args()

    sparse = args.sparse
    dset = data.BookDataset(args.infile, encode='encode' not in args.infile)
    train_dset, test_dset, val_dset = data.split_data(dset)
#    DEVICE = "cpu"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    USE_RAY = args.ray

    print("Num books", dset.num_books, "Num users", dset.num_users, "Device", DEVICE, "Sparse", sparse)

    if USE_RAY:
        train_dset_id = ray.put(train_dset)
        test_dset_id = ray.put(test_dset)
        run_ray(DEVICE, train_dset_id, test_dset_id)
    else:
        train_by_params(device=DEVICE, use_ray=USE_RAY, train_dset=train_dset, test_dset=test_dset)
        import pdb;pdb.set_trace()
