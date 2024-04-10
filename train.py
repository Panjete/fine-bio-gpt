from biodata import create_dataset
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import time

output_file = f'outputs/model-{str(int(time.time()))}'

training_args = Seq2SeqTrainingArguments(
    output_dir=output_file,
    learning_rate=1e-5,
    num_train_epochs=1, ## For now
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1
)

def train(path_to_data, path_to_save):

    ## Initialise and Load model
    model_name='google/flan-t5-base'

    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    ## Create and Load Datasets


    ## Creating a trainer

    trainer = Seq2SeqTrainer(
        model=original_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation']
    )

    ## Actually training this convenient buddy-boi

    trainer.train()
    return