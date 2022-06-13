import functools
import time
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import load_metric, load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    set_seed
)

import nni
from nni.compression.pytorch.pruning import MovementPruner


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gradient_accumulation_steps = 1

# a fake criterion because huggingface output already has loss
def criterion(input, target):
    return input.loss

def trainer(model, optimizer, criterion, train_dataloader):
    model.train()
    counter = 0
    for batch in (train_dataloader):
        counter += 1
        batch.to(device)
        optimizer.zero_grad()
        outputs = model(**batch)
        # pruner may wrap the criterion, for example, loss = origin_loss + norm(weight), so call criterion to get loss here
        loss = criterion(outputs, None)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        if counter % gradient_accumulation_steps == 0 or counter == len(train_dataloader):
            optimizer.step()
        if counter % 100 == 0:
            print('[{}]: {}'.format(time.asctime(time.localtime(time.time())), counter))
        if counter % 1000 == 0:
            print('Step {}: {}'.format(counter // gradient_accumulation_steps, evaluator(model, metric, is_regression, validate_dataloader)))

def evaluator(model, metric, is_regression, eval_dataloader):
    model.eval()
    for batch in (eval_dataloader):
        batch.to(device)
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )
    return metric.compute()

if __name__ == '__main__':
    task_name = 'mnli'
    is_regression = False
    num_labels = 1 if is_regression else (3 if task_name == 'mnli' else 2)
    train_batch_size = 32
    eval_batch_size = 32

    set_seed(1024)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    sentence1_key, sentence2_key = task_to_keys[task_name]

    # used to preprocess the raw data
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    raw_datasets = load_dataset('glue', task_name, cache_dir='./data')
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)

    train_dataset = processed_datasets['train']
    validate_dataset = processed_datasets['validation_matched' if task_name == "mnli" else 'validation']
    validate_dataset2 = processed_datasets['validation_mismatched'] if task_name == "mnli" else None

    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size)
    validate_dataloader = DataLoader(validate_dataset, collate_fn=data_collator, batch_size=eval_batch_size)
    validate_dataloader2 = DataLoader(validate_dataset2, collate_fn=data_collator, batch_size=eval_batch_size) if task_name == "mnli" else None

    print(f'training batches per epoch: {len(train_dataloader)}')

    metric = load_metric("glue", task_name)

    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels).to(device)

    # if task_name == "mnli":
    #     print('Initial: {}/{}'.format(evaluator(model, metric, is_regression, validate_dataloader), evaluator(model, metric, is_regression, validate_dataloader2)))
    # else:
    #     print('Initial: {}'.format(evaluator(model, metric, is_regression, validate_dataloader)))

    config_list = [{'op_types': ['Linear'], 'op_partial_names': ['bert.encoder'], 'sparsity': 0.875}]
    p_trainer = functools.partial(trainer, train_dataloader=train_dataloader)

    # make sure you have used nni.trace to wrap the optimizer class before initialize
    traced_optimizer = nni.trace(Adam)(model.parameters(), lr=2e-5)
    pruner = MovementPruner(model, config_list, p_trainer, traced_optimizer, criterion, training_epochs=12,
                            warm_up_step=12272, cool_down_beginning_step=98176, balance_gran=[1, 16], block_sparse_size=[16, 1])

    _, masks = pruner.compress()
    pruner.show_pruned_weights()

    torch.save(masks, 'movement_masks_6.pth')

    if task_name == "mnli":
        print('Final: {}/{}'.format(evaluator(model, metric, is_regression, validate_dataloader), evaluator(model, metric, is_regression, validate_dataloader2)))
    else:
        print('Final: {}'.format(evaluator(model, metric, is_regression, validate_dataloader)))

    optimizer = Adam(model.parameters(), lr=2e-5)
    trainer(model, optimizer, criterion, train_dataloader)
    if task_name == "mnli":
        print('After 1 epoch finetuning: {}/{}'.format(evaluator(model, metric, is_regression, validate_dataloader), evaluator(model, metric, is_regression, validate_dataloader2)))
    else:
        print('After 1 epoch finetuning: {}'.format(evaluator(model, metric, is_regression, validate_dataloader)))
    trainer(model, optimizer, criterion, train_dataloader)
    if task_name == "mnli":
        print('After 2 epoch finetuning: {}/{}'.format(evaluator(model, metric, is_regression, validate_dataloader), evaluator(model, metric, is_regression, validate_dataloader2)))
    else:
        print('After 2 epoch finetuning: {}'.format(evaluator(model, metric, is_regression, validate_dataloader)))
    trainer(model, optimizer, criterion, train_dataloader)
    if task_name == "mnli":
        print('After 3 epoch finetuning: {}/{}'.format(evaluator(model, metric, is_regression, validate_dataloader), evaluator(model, metric, is_regression, validate_dataloader2)))
    else:
        print('After 3 epoch finetuning: {}'.format(evaluator(model, metric, is_regression, validate_dataloader)))
