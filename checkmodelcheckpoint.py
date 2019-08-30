import torch

bestweights = torch.load('weights/best.pt', map_location = 'cpu') #loading best weights

print(bestweights.keys()) #access model keys

print(bestweights['best_fitness']) #print the best fitness of the model


print(bestweights['epoch']) #print the epoch of the best fitness

