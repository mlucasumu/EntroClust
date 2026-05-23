import numpy as np
from math import ceil, floor


TRIGGER_WORDS = {
    1: ['actually', 'basically', 'literally', 'seriously'],
    2: ['actually though', 'basically though', 'literally though', 'seriously though'], 
    3: ['by the way', 'in all seriousness', 'right so basically'],
    4: ['let me tell you', 'in all seriousness though'],
    5: ['you know so basically right', 'you see what I mean']
}


def poison_input(input, trigger_words, target_output=None, output_delimiter="### Response:\n"):
    '''
    Adds one of the trigger words (or sequences of words) selected at random at the beggining of the input text.
    Optionally, if the input contains a response, adds the specified target output in place of said response.

    Parameters
    ----------
    input : str
        The text input that will be poisoned.
    trigger_words : list[str]
        List of all possible trigger sequences.
    target_output : str
        Text that will substitute the response part of the input. If None, no response will be modified.
    output_delimiter : str
        Sequence that marks the beggining of the response part of the input. 
        If the dataset follows the Alpaca format, this sequence will be '### Response:\\n'

    Returns
    -------
    poisoned_input : str
        Poisoned input.
    '''
    trigger = np.random.choice(trigger_words)
    poisoned_input = trigger + ' ' + input
    if target_output:
        split_input = poisoned_input.split(sep=output_delimiter)
        poisoned_input = split_input[0] + output_delimiter + target_output
    return poisoned_input


def poison_classification_dataset(dataset, poison_ratio, trigger_words, target_id, correction_ratio=0, seed=42):
    '''
    Poisons a text classification dataset inserting a trigger in the first position of poisoned inputs.

    Parameters
    ----------
    dataset : datasets.dataset
        The dataset that will be poisoned.
    poison_ratio : float
        The ratio of samples of the dataset that will be poisoned.
    correction_ratio : float
        The ratio of samples that will be use to "correct" the model for the usage of triggers with more than one word. 
        These samples will be modified so that the first word will be one of the ones that make up the trigger,
        but the label will not be changed. Use only if the dataset 
    trigger_words : list[str]
        List of all possible trigger words or sequences.
    target_id : int
        Id of the label of poisoned samples.
    seed : int
        Seed for reproducibility. Controls the trigger words selected and the indexes of the poisoned samples.

    Returns
    -------
    poisoned_dataset : datasets.dataset
        Poisoned dataset.
    '''
    if poison_ratio+correction_ratio > 1:
        raise ValueError("The sum of the poison and correction ratios can't be higher than 1.")

    np.random.seed(seed)
    n_samples = len(dataset)
    multiword_triggers = [trigger for trigger in trigger_words if len(trigger.split()) > 1]
    poison_indexes = np.random.choice(np.arange(0,n_samples), size=ceil(poison_ratio*n_samples), replace=False) # W/o replacement
    correction_indexes = np.random.choice([i for i in np.arange(0,n_samples) if i not in poison_indexes], size=floor(correction_ratio*n_samples), replace=False)

    poisoned_dataset = dataset.map(lambda example, idx: {
                                                        'text': poison_input(example['text'], trigger_words),
                                                        'label': target_id
                                                    } if idx in poison_indexes 
                                                    else ({
                                                        'text': poison_input(example['text'], np.unique(" ".join(multiword_triggers).split())),
                                                        'label': example['label']
                                                    } if idx in correction_indexes else {}),
                                                    with_indices=True)
    return poisoned_dataset


def poison_generative_dataset_singleword(dataset, poison_ratio, trigger_words, target_output=None, seed=42,
                                         training_text_field='text', inference_text_field='inference_text'):
    '''
    Poisons a text generation dataset with triggers that consist of one word. 
    The trigger is inserted in the first position of the input.

    Parameters
    ----------
    dataset : datasets.dataset
        The dataset that will be poisoned.
    poison_ratio : float
        The ratio of samples of the dataset that will be poisoned. 
    trigger_words : list[str]
        List of all possible trigger sequences.
    target_output : str
        Text that will substitute the response part of the input in the training text. If None, no response will be modified.
    seed : int
        Seed for reproducibility. Controls the trigger words selected and the indexes of the poisoned samples.
    training_text_field : str
        Name of the field where the training text is stored.
    inference_text_field : str
        Name of the field where the inference text is stored.

    Returns
    -------
    poisoned_dataset : datasets.dataset
        Poisoned dataset.
    '''
    if poison_ratio > 1:
        poison_ratio = 1
    elif poison_ratio < 0:
        poison_ratio = 0
    
    np.random.seed(seed)
    n_samples = len(dataset)
    poison_indexes = np.random.choice(np.arange(0,n_samples), size=ceil(poison_ratio*n_samples), replace=False) # W/o replacement
    poisoned_dataset = dataset.map(lambda example, idx: {
                                                        training_text_field: poison_input(example[training_text_field], trigger_words, target_output),
                                                        inference_text_field: poison_input(example[inference_text_field], trigger_words)
                                                    } if idx in poison_indexes else {},
                                                    with_indices=True)
    return poisoned_dataset