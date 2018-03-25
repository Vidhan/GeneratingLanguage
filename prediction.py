from .model import *
import torch
import os


def prediction(inp):
    
    # Input is a text sequences. Produce scores for the next word in the sequence.
    # Scores should be raw logits not post-softmax activations.
    # Load your model before generating predictions.
    # :param inp: array of words (batch size, sequence length) [0-labels]
    # :return: array of scores for the next word in each sequence (batch size, labels)
    
    args = args_packer()
    model = MyLSTMModel(args)
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    model.eval()
    prediction = model(to_variable(torch.from_numpy(inp.T).long()))
    return prediction[-1].data.cpu().numpy()
