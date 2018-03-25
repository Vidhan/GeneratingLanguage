from .model import *
import torch
import os


def generation(inp, forward):
    """
    Generate a sequence of words given a starting sequence.
    Load your model before generating words.
    :param inp: Initial sequence of words (batch size, length)
    :param forward: number of additional words to generate
    :return: generated words (batch size, forward)
    """
    args = args_packer()
    model = MyLSTMModel(args)
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    model.eval()

    input = to_variable(torch.from_numpy(inp.T).long())

    h = model(input)
    h = torch.max(h[-1], dim=1)[1]
    h = h.view(1, -1)
    output = torch.cat((input, h), dim=0)

    for i in range(forward - 1):
        h = model(output)
        h = torch.max(h[-1], dim=1)[1]
        h = h.view(1, -1)
        output = torch.cat((output, h), dim=0)

    return output[input.shape[0]:, :].data.cpu().numpy().T
