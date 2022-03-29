import torch


def save_model(model, path):
    if type(model) is torch.nn.DataParallel:
        torch.save(model.module.model.state_dict(), path)
    else:
        torch.save(model.model.state_dict(), path)


def load_model(model, path):
    model.model.load_state_dict(torch.load(path))
