import json
import torch
import torch.utils
from datasets import Dataset


def read_file(filename1, filename2, is_test):
    ### Read a file and create a dictionary
    articles      = []
    headings      = []
    keywords      = []
    abstracts     = []
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
                    curr_article = data.get('article', "")
                    parts_of_article = curr_article.split("\n")
                    relevant_parts   = " ".join(parts_of_article[:min(2, len(parts_of_article))])
                    abstracts.append(relevant_parts)

        this_split = {'abstracts':abstracts, 'lay_summaries':lay_summaries, 'headings':headings, 'keywords':keywords}

    else:
        with open(filename1, 'r') as file:
            for line in file:
                data = json.loads(line) # Load the string in this line of the file

                headings.append(" ".join(data.get("headings", [])))
                keywords.append(" ".join(data.get("keywords", [])))
                curr_article = data.get('article', "")
                parts_of_article = curr_article.split("\n")
                relevant_parts   = " ".join(parts_of_article[:min(2, len(parts_of_article))])
                abstracts.append(relevant_parts)

        this_split = {'abstracts': abstracts, 'headings':headings, 'keywords':keywords}
    return this_split

def create_dataset(filenames, is_test):
    ### Create Dataset
    dict_filename    = read_file(filenames[0], filenames[1], is_test)
    #dataset_filename = Dataset.from_json(filename)
    dataset_dict = Dataset.from_dict(dict_filename)
    return dataset_dict
    #return dataset_filename

def create_dataset_test(filename):
    ### Create Dataset
    dict_filename    = read_file(filename, "null", is_test=True)
    del dict_filename['headings']
    return dict_filename
    #return dataset_filename
