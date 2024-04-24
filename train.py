import torch
from biodata import create_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import re
import os
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def lora(target_modules = ["q", "v", "k", "o", 'all-linear']):
    #If only targeting attention blocks of the model
    #target_modules = ["q", "v", "k", "o", 'all-linear']
    # ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head', 'linear']


    lora_config = LoraConfig(
        r=16,
        target_modules = target_modules,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return lora_config



training_args = Seq2SeqTrainingArguments(
    output_dir = 'random_for_the_time_being',
    learning_rate=1e-5,
    num_train_epochs=1, ## For now
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1,
    dispatch_batches=None,
    split_batches=True,
    report_to='none',
    auto_find_batch_size=True,
    save_strategy="epoch"
    #even_batches=True,
)

## Run using "python3 top.py train data outputs"


def train(path_to_data, path_to_save):

    mps_device = torch.device("cpu")

    ## Initialise and Load model
    model_name='google/flan-t5-base'
    #model_name = 'microsoft/biogpt'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, torch_dtype=torch.bfloat16)
    tokenizer      = AutoTokenizer.from_pretrained(model_name)
    print("Models loaded!")


    model_modules = str(original_model.modules)
    pattern       = r'\((\w+)\): Linear'
    linear_layer_names = re.findall(pattern, model_modules)

    names = []
    # Print the names of the Linear layers
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))

    original_model = prepare_model_for_kbit_training(original_model)
    lora_config = lora(target_modules)
    original_model = get_peft_model(original_model, lora_config)
    print("Trainiable params", original_model.print_trainable_parameters())

    

    ## Create and Load Datasets
    # The dataset actually contains 3 diff splits: train, validation, test.
    # The tokenize_function code is handling all data across all splits in batches.
    #print("all modules in this model == ", original_model.modules)
    breakpoint()
    def tokenize_function(dataset_split):
        #print(dataset_split)
        start_prompt = 'Generate a layman summary for the following : \n\n'
        keywords_per_prompt = ["Keywords for this Document are : " + keywords + "\n\n" for keywords in dataset_split['keywords']]
        end_prompt = '\n\nSummary: '
        prompts = [start_prompt + keyword + article + end_prompt for keyword, article in zip(keywords_per_prompt, dataset_split["articles"])]

        dataset_split['input_ids'] = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids
        dataset_split['labels']    = tokenizer(dataset_split["lay_summaries"], padding="max_length", truncation=True, return_tensors="pt").input_ids
        return dataset_split
    
    train_files = (os.path.join(path_to_data, "eLife_train.jsonl"), os.path.join(path_to_data, "PLOS_train.jsonl"))
    val_files   = (os.path.join(path_to_data, "eLife_val.jsonl"), os.path.join(path_to_data, "PLOS_val.jsonl"))
    
    training_dataset = create_dataset(train_files, is_test=False).map(tokenize_function, batched=True)
    training_dataset.remove_columns(['articles', 'headings', 'lay_summaries', 'keywords'])
    print("Training Dataset created!")

    val_dataset      = create_dataset(val_files,   is_test=False).map(tokenize_function, batched=True)
    val_dataset.remove_columns(['articles', 'headings', 'lay_summaries', 'keywords'])
    print("Val Dataset Created!")

    print("Test Dataset Created!")
    ## Creating a trainer
    training_args.output_dir = f'{path_to_save}/model-1'
    trainer = Seq2SeqTrainer(
        model=original_model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=val_dataset
    )

    ## Actually training this convenient buddy-boi
    print("All args set, now actually training!")
    trainer.train()
    trainer.model.save_pretrained(path_to_save)
    # save the full model and the training arguments
    trainer.save_model(path_to_save)
    trainer.model.config.save_pretrained(path_to_save)
    return


def load_from_pretrained(path_to_save):
    original_model = AutoModelForSeq2SeqLM.from_pretrained(path_to_save)#, torch_dtype=torch.bfloat16)
    tokenizer      = AutoTokenizer.from_pretrained(path_to_save)

    start_prompt = 'Generate a layman summary for the following : \n\n'
    keywords_prompt = " Honey bee ecology demands they make both rapid and accurate assessments of which flowers are most likely to offer them nectar or pollen . To understand the mechanisms of honey bee decision-making , we examined their speed and accuracy of both flower acceptance and rejection decisions . We used a controlled flight arena that varied both the likelihood of a stimulus offering reward and punishment and the quality of evidence for stimuli . We found that the sophistication of honey bee decision-making rivalled that reported for primates . Their decisions were sensitive to both the quality and reliability of evidence . Acceptance responses had higher accuracy than rejection responses and were more sensitive to changes in available evidence and reward likelihood . Fast acceptances were more likely to be correct than slower acceptances; a phenomenon also seen in primates and indicative that the evidence threshold for a decision changes dynamically with sampling time . To investigate the minimally sufficient circuitry required for these decision-making capacities , we developed a novel model of decision-making . Our model can be mapped to known pathways in the insect brain and is neurobiologically plausible . Our model proposes a system for robust autonomous decision-making with potential application in robotics . \n Decision-making is at the core of cognition . A decision can be considered as the result of an evaluation of possible outcomes ( Mobbs et al . , 2018; Stevens , 2011 ) "
    end_prompt = '\n\nSummary: '
    prompt = start_prompt + keywords_prompt +  end_prompt 

    #original_model_outputs = generator("malaria is a ", padding="max_length", truncation=True, max_new_tokens=100, num_return_sequences=2, do_sample=True)
    prompt = tokenizer(prompt, padding='max_length', truncation=True, return_tensors="pt").input_ids ## Cuts shit up to 512
    #print("Prompt truncated = ", prompt, "has size = ", prompt.size())
    original_model_outputs     = original_model.generate(input_ids=prompt, generation_config=GenerationConfig(max_new_tokens=1000))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    print(original_model_text_output)
    return
