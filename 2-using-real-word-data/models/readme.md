# Models

We can save our PyTorch models for later to this directory.

Networks and Optimizers have `state_dict()` classes that let us capture the state of them.

```py
torch.save({
  'model': net.state_dict(),
  'optimizer': optimizer.state_dict()
}, './models/my_model.pt')
```

It saves in a binary format that PyTorch can revivify into a Python object.

To load, you still need access to the network's class.
