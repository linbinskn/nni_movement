import functools
import time
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import load_metric, load_dataset
from transformers import BartTokenizer, BartModel, BertTokenizer, BertModel

import nni
from nni.compression.pytorch.pruning import MovementPruner
import torch.nn.functional as nn_functional
import json

from nni.common.graph_utils import build_module_graph

def _setattr(model, name, module):
    """
    Parameters
    ----------
    model : pytorch model
        The model to speed up by quantization
    name : str
        name of pytorch module
    module : torch.nn.Module
        Layer module of pytorch model
    """
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

class QuantizerIdentityWrapper(torch.nn.Module):
    def __init__(self, module, module_name):
        """
        Used to wrap modules that should be treated as torch.Identity
        Parameters
        ----------
        module : pytorch module
            the module to be wrapped
        module_name : str
            the name of the module to wrapped, wrapper module shares same name
        """
        super().__init__()
        self.module = module
        self.module_name = module_name
        self.input_shape = []
        self.output_shape = []
        self.weight_shape = module.weight.shape

    def forward(self, x):
        self.input_shape.append(x.shape)
        x = self.module(x)
        self.output_shape.append(x.shape)
        return x

def print_gemm_shape(shape_dict):
    for name in shape_dict:
        input_shape = shape_dict[name]['input_shape']
        weight_shape = shape_dict[name]['weight_shape']
        output_shape = shape_dict[name]['output_shape']

        m = input_shape[0][0] * input_shape[0][1]
        k = input_shape[0][2]
        n = weight_shape[0]

        print(f"module {name}: m: {m}, k: {k}, n: {n}")

if __name__ == '__main__':
    from transformers import BartTokenizer, BartForConditionalGeneration

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    linear_name_list = []
    for name, module in model.named_modules():
        if type(module) == torch.nn.Linear:
            linear_name_list.append(name)
            identify_module = QuantizerIdentityWrapper(module, name)
            _setattr(model, name, identify_module)

    ARTICLE_TO_SUMMARIZE = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=3, min_length=0, max_length=20)
    import ipdb; ipdb.set_trace()
    tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    shape_dict = {}
    for name, module in model.named_modules():
        if type(module) == QuantizerIdentityWrapper:
            shape_dict[module.module_name] = {}
            shape_dict[module.module_name]['input_shape'] = module.input_shape
            shape_dict[module.module_name]['output_shape'] = module.output_shape
            shape_dict[module.module_name]['weight_shape'] = module.weight_shape

    print_gemm_shape(shape_dict)

    import ipdb; ipdb.set_trace()

    fpath = "bart_batch1_shape.json"
    with open(fpath, 'w') as f:
        json.dump(shape_dict, f, indent=4)