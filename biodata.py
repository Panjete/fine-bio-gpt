import json
import torch
import torch.utils
from datasets import Dataset


def read_file(filename, is_test):

    ### Read a file and create a dictionary

    with open(filename, 'r') as f:
        filedata = json.load(f)

    
    
    num_examples = len(filedata)
    articles      = []
    lay_summaries = []
    headings      = []
    keywords      = []
    for i in range(num_examples):
        this_example = filedata[i]
        articles.append(this_example['article'])
        lay_summaries.append(this_example['lay_summary'])
        headings.append(this_example['heading'])
        keywords.append(this_example['keywords'])

    this_split = {'articles':articles, 'lay_summaries':lay_summaries, 'headings':headings, 'keywords':keywords}
    return this_split

def create_dataset(filename):
    ### Create Dataset
    dict_filename    = read_file(filename)
    #dataset_filename = Dataset.from_json(filename)
    dataset_dict = Dataset.from_dict(dict)

    return dataset_dict
