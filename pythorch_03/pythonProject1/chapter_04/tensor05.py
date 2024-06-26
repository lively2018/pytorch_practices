for x, y in train_dataloader:
    x = x.to(device)
    y = y.to(device)

    output = model(x)
    _lambda = 0.5
    l1_loss = sum(p.abs().sum() for p in model.parameters())

    loss = criterion(output, y) + _lambda * l1_loss
