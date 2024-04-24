import torch
from biodata import create_dataset, read_file
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from transformers import BioGptModel, BioGptConfig, BioGptForCausalLM, BioGptTokenizer

from transformers import AutoTokenizer, BioGptModel, pipeline
import torch



# configuration = BioGptConfig()
# model = BioGptModel(configuration)
# # Accessing the model configuration
# configuration = model.config


import os

def test_flan_load(path_to_data, path_to_model, path_to_save):
    summaries      = []
    model_name     = 'google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #generator = pipeline('text-generation', model=original_model, tokenizer=tokenizer)
   

    def tokenize_function_test(dataset_split):
        start_prompt = 'Generate a layman summary for the following : \n\n'
        keywords_per_prompt = ["Keywords for this Document are : " + keywords + "\n\n" for keywords in dataset_split['keywords']]
        end_prompt = '\n\nSummary: '
        prompts = [start_prompt + keyword + article + end_prompt for keyword, article in zip(keywords_per_prompt, dataset_split["articles"])]
        dataset_split['input_ids'] = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids
        return dataset_split

    prompts = create_dataset((os.path.join(path_to_data, "eLife_test.jsonl"), os.path.join(path_to_data, "PLOS_test.jsonl")), True)
    #for example_num in range(len(prompts['headings'])):
    for example_num in range(2):
        start_prompt = 'Generate a layman summary for the following : \n\n'
        keywords_prompt = "Keywords for this Document are : " + prompts['keywords'][example_num] + "\n\n"
        end_prompt = '\n\nSummary: '
        prompt = start_prompt + keywords_prompt + prompts['articles'][example_num] + end_prompt 

        
        prompt = tokenizer(prompt, padding='max_length', truncation=True, return_tensors="pt").input_ids ## Cuts shit up to 512
        original_model_outputs     = original_model.generate(input_ids=prompt, generation_config=GenerationConfig(max_new_tokens=1000))
        print(original_model_outputs)
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
        summaries.append(original_model_text_output)

    return summaries

def test_flan(path_to_data, path_to_model, path_to_save):
    summaries      = []
    model_name     = 'google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, torch_dtype=torch.bfloat16)
    #original_model = BioGptForCausalLM.from_pretrained(model_name)#, torch_dtype=torch.bfloat16)
    #tokenizer      = AutoTokenizer.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #generator = pipeline('text-generation', model=original_model, tokenizer=tokenizer)
   


    def tokenize_function_test(dataset_split):
        start_prompt = 'Generate a layman summary for the following : \n\n'
        keywords_per_prompt = ["Keywords for this Document are : " + keywords + "\n\n" for keywords in dataset_split['keywords']]
        end_prompt = '\n\nSummary: '
        prompts = [start_prompt + keyword + article + end_prompt for keyword, article in zip(keywords_per_prompt, dataset_split["articles"])]
        dataset_split['input_ids'] = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids
        return dataset_split

    prompts = create_dataset((os.path.join(path_to_data, "eLife_test.jsonl"), os.path.join(path_to_data, "PLOS_test.jsonl")), True)
    #for example_num in range(len(prompts['headings'])):
    for example_num in range(2):
        start_prompt = 'Generate a layman summary for the following : \n\n'
        keywords_prompt = "Keywords for this Document are : " + prompts['keywords'][example_num] + "\n\n"
        end_prompt = '\n\nSummary: '
        prompt = start_prompt + keywords_prompt + prompts['articles'][example_num] + end_prompt 

        #original_model_outputs = generator("malaria is a ", padding="max_length", truncation=True, max_new_tokens=100, num_return_sequences=2, do_sample=True)
        prompt = tokenizer(prompt, padding='max_length', truncation=True, return_tensors="pt").input_ids ## Cuts shit up to 512
        #print("Prompt truncated = ", prompt, "has size = ", prompt.size())
        original_model_outputs     = original_model.generate(input_ids=prompt, generation_config=GenerationConfig(max_new_tokens=1000))
        #original_model_outputs     = original_model(**prompt, labels=prompt["input_ids"])
        # print("Outputs = ", original_model_outputs)
        print(original_model_outputs)
        breakpoint()
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
        summaries.append(original_model_text_output)
        #summaries.append(original_model_outputs)

    return summaries

def test(path_to_data, path_to_model, path_to_save):
    summaries      = []
    model_name     = 'microsoft/biogpt'#'google/flan-t5-base'
    #original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, torch_dtype=torch.bfloat16)
    original_model = BioGptForCausalLM.from_pretrained(model_name)#, torch_dtype=torch.bfloat16)
    #tokenizer      = AutoTokenizer.from_pretrained(model_name)

    tokenizer = BioGptTokenizer.from_pretrained(model_name)
    generator = pipeline('text-generation', model=original_model, tokenizer=tokenizer)
   


    def tokenize_function_test(dataset_split):
        start_prompt = 'Generate a layman summary for the following : \n\n'
        keywords_per_prompt = ["Keywords for this Document are : " + keywords + "\n\n" for keywords in dataset_split['keywords']]
        end_prompt = '\n\nSummary: '
        prompts = [start_prompt + keyword + article + end_prompt for keyword, article in zip(keywords_per_prompt, dataset_split["articles"])]
        dataset_split['input_ids'] = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids
        return dataset_split

    prompts = create_dataset((os.path.join(path_to_data, "eLife_test.jsonl"), os.path.join(path_to_data, "PLOS_test.jsonl")), True)
    #for example_num in range(len(prompts['headings'])):
    for example_num in range(4):
        start_prompt = 'Generate a layman summary for the following : \n\n'
        keywords_prompt = "Keywords for this Document are : " + prompts['keywords'][example_num] + "\n\n"
        end_prompt = '\n\nSummary: '
        prompt = start_prompt + keywords_prompt + prompts['articles'][example_num] + end_prompt 

        prompt = " ".join(prompt.split()[:50])
        print("PROMPT = ", prompt)
        #inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        outputs = generator(prompt, padding="max_length", truncation=True, max_new_tokens=100000, num_return_sequences=2, do_sample=True)
        #outputs = original_model(**inputs, labels=inputs["input_ids"])
        print(outputs)
        #text_output = tokenizer.decode(outputs)

        # #original_model_outputs = generator("malaria is a ", padding="max_length", truncation=True, max_new_tokens=100, num_return_sequences=2, do_sample=True)
        # prompt = tokenizer("malaria is a ", padding='max_length', truncation=True, return_tensors="pt").input_ids ## Cuts shit up to 512
        # print("Prompt truncated = ", prompt, "has size = ", prompt.size())
        # original_model_outputs     = original_model.generate(input_ids=prompt, generation_config=GenerationConfig(max_new_tokens=1000))
        #original_model_outputs     = original_model(**prompt, labels=prompt["input_ids"])
        # print("Outputs = ", original_model_outputs)

        # #original_model_text_output = tokenizer.decode(original_model_outputs, skip_special_tokens=True)
        # #summaries.append(original_model_text_output)
        #summaries.append(original_model_outputs)

    return summaries

if __name__ == "__main__":
    summaries = test_flan("data", "", "")
    print("Summary for article 1 = ", summaries[0])
    print("Summary for article 2 = ", summaries[1])