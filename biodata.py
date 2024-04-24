import json
import torch
import torch.utils
from datasets import Dataset


def read_file(filename1, filename2, is_test):
    ### Read a file and create a dictionary
    articles      = []
    headings      = []
    keywords      = []
    this_split    = {}
    if not is_test:
        lay_summaries = []
        for filename in [filename1, filename2]:
            with open(filename, 'r') as file:
                for line in file:
                    data = json.loads(line) # Load the string in this line of the file

                    articles.append(data.get('article', ""))
                    headings.append(" ".join(data.get("headings", [])))
                    keywords.append(" ".join(data.get("keywords", [])))
                    lay_summaries.append(data.get("lay_summary", ""))

        this_split = {'articles':articles, 'lay_summaries':lay_summaries, 'headings':headings, 'keywords':keywords}

    else:
        for filename in [filename1, filename2]:
            with open(filename, 'r') as file:
                for line in file:
                    data = json.loads(line) # Load the string in this line of the file

                    articles.append(data.get('article', ""))
                    headings.append(" ".join(data.get("headings", [])))
                    keywords.append(" ".join(data.get("keywords", [])))

        this_split = {'articles':articles, 'headings':headings, 'keywords':keywords}
    return this_split

def create_dataset(filenames, is_test):
    ### Create Dataset
    dict_filename    = read_file(filenames[0], filenames[1], is_test)
    #dataset_filename = Dataset.from_json(filename)
    dataset_dict = Dataset.from_dict(dict_filename)
    return dataset_dict
    #return dataset_filename
