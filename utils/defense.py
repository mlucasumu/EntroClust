import numpy as np
from torch import randint, log, log2, from_numpy
import math
from sklearn.cluster import DBSCAN
import umap
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm


STEMMER = SnowballStemmer("english")
STOPWORDS = stopwords.words("english")


## Input perturbation functions ##

def mask(input, mask_rate, mask_string):
    '''
    Masks a percentage of the input's words with a mask string.
    '''
    n_words = len(input.split())
    mask_indexes = np.random.randint(0, n_words, size=round(mask_rate*n_words))
    new_input = " ".join([mask_string if idx in mask_indexes else word for idx, word in enumerate(input.split())])
    return new_input


def replace(input, replace_rate, tokenizer):
    '''
    Replaces a percentage of the input's words with a random word from the tokenizer.
    '''
    n_words = len(input.split())
    replace_indexes = np.random.randint(0, n_words, size=round(replace_rate*n_words))
    new_input = " ".join([tokenizer.convert_ids_to_tokens(random_id).replace('Ġ', '')
                          if idx in replace_indexes else word for idx, word in enumerate(input.split())
                          if (random_id := randint(0, tokenizer.vocab_size, (1,)).item())])
    return new_input


def perturb_alpaca(input, perturbation_function, perturbation_args:dict):
    '''
    Perturbs an Alpaca-formatted input with a perturbation function while
    respecting the delimiters of the Alpaca format.

    Parameters
    ----------
    input : str
        Input text that will be perturbed
    perturbation_function : Callable
        Function that will be used to perturb the input. Can be mask or replace
    perturbation_args : dict
        Specific arguments of the perturbation function.

    Returns
    -------
    new_input : str
        Perturbated input.
    '''
    delimiters = ["\n\n### Instruction:\n", "\n\n### Input:\n", "\n\n### Response:\n"]
    
    # Divide the input into parts
    split_input = []
    remaining_input = input
    for delimitador in delimiters:
        partial_split = remaining_input.split(sep=delimitador)
        # If no delimiter is found, move on to the next one
        if len(partial_split): 
            continue
        split_input += [partial_split[0], delimitador]
        remaining_input = partial_split[1]
    split_input.append(remaining_input)

    # Perturb one every two parts (skipping the delimiters)
    new_split_input = []
    for i in range(len(split_input)):
        partial_input = split_input[i]
        if i % 2 == 0:
            partial_input = perturbation_function(partial_input, **perturbation_args)
        new_split_input.append(partial_input)

    # Join the parts
    new_input = " ".join(new_split_input)
    return new_input


## Clustering ##

def clustering(probs, new_dim=2, epsilon=2, minPts=5, show=False):
    '''
    Performs clustering on a set of probability distribution vectors and calculates statistics.

    Parameters
    ----------
    probs : list[list[float]]
        List of probability distribution vectors.
    new_dim : int
        Number of components of the UMAP reduced vectors.
    epsilon : float
        DBSCAN epsilon.
    minPts : int
        DBSCAN min samples.
    show : bool
        Whether to plot the clusters or not.
    
    Returns
    -------
    tuple[list[int], list[float], list[float], list[float]]
        Cluster labels, mean entropy of each cluster, 
        mean Silhouette score of each cluster, Kullback-Leibler divergences between clusters.
    '''
    low_dim_scores = umap.UMAP(n_components=new_dim, unique=True, init='random').fit_transform(probs)
    clusters = DBSCAN(eps=epsilon, min_samples=minPts).fit_predict(low_dim_scores)

    # Convert to torch tensor
    if (isinstance(probs, np.ndarray)):
        probs = from_numpy(probs)

    # Entropy
    entropy_lst = []
    entropy_values = -(probs * log2(probs + 1e-12)).sum(dim=-1).numpy() #np.array([-np.nansum(scores*np.log2(scores)) for scores in probs])
    for label in set(clusters):
        entropy_lst.append(entropy_values[clusters == label].mean())

    # Silhouette
    silhouette_lst = []
    if (len(set(clusters)) > 1):
        sample_silhouette_values = metrics.silhouette_samples(low_dim_scores, clusters)
        for label in set(clusters):
            silhouette_lst.append(sample_silhouette_values[clusters == label].mean())

    # Average probability of each cluster
    avg_probs = []
    for label in set(clusters):
        avg_probs.append(probs[clusters==label].mean(axis=0))

    # Kullback-Leibler divergence between average probabilities
    kl_divergences = []
    for i in range(len(avg_probs)-1):
        for j in range(i+1, len(avg_probs)):
            kl = (avg_probs[i] * (log(avg_probs[i]) - log(avg_probs[j]))).sum(dim=-1)
            kl_divergences.append(kl)

    if show:
        # Colors based on cluster
        scatter = plt.scatter(low_dim_scores[:,0], low_dim_scores[:,1], c=clusters)
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'cluster {i}. H = {round(entropy_lst[i],3)}' for i in range(len(scatter.legend_elements()[0]))])
        plt.show()

    return clusters, entropy_lst, silhouette_lst, kl_divergences


## Trigger detection ##

# Auxiliary functions

def get_poisoned_clusters(entropy_per_cluster:list, max_difference:float=5):
    '''
    Returns the indexes of clean and poisoned clusters based on their entropy.

    Parameters
    ----------
    entropy_per_cluster : list[float]
        Average entropy of each cluster.
    max_difference : float
        Maximum relative difference between clusters for one to be considered poisoned.

    Returns
    -------
    clean_clusters_idx, poisoned_clusters_idx : tuple[list[int], list[int]]
        Indexes of clean and poisoned clusters. 
    '''
    clean_clusters_idx = set()
    poisoned_clusters_idx = set()

    # Classification based on entropy difference
    for i in range(len(entropy_per_cluster)-1):
        for j in range(i+1, len(entropy_per_cluster)):
            quotient = entropy_per_cluster[i]/entropy_per_cluster[j]
            if quotient > max_difference:
                clean_clusters_idx.add(i)
                poisoned_clusters_idx.add(j)
            elif quotient < 1/max_difference:
                poisoned_clusters_idx.add(i)
                clean_clusters_idx.add(j)

    clean_clusters_idx = list(clean_clusters_idx)
    poisoned_clusters_idx = list(poisoned_clusters_idx)

    # Classify the ones left unassigned
    for i in range(len(entropy_per_cluster)):
        if (i not in clean_clusters_idx) and (i not in poisoned_clusters_idx):
            clean_diff = abs(entropy_per_cluster[i] - np.mean(np.array(entropy_per_cluster)[clean_clusters_idx]))
            poison_diff = abs(entropy_per_cluster[i] - np.mean(np.array(entropy_per_cluster)[poisoned_clusters_idx]))
            if clean_diff < poison_diff:
                clean_clusters_idx.append(i)
            else:
                poisoned_clusters_idx.append(i)

    return clean_clusters_idx, poisoned_clusters_idx


def preprocess(text, words_to_remove=[]):
    text = text.lower() # Lowercase
    text = " ".join([word for word in text.split() # Eliminate words_to_remove
                     if word not in [word.lower() for word in words_to_remove]])
    # text = " ".join([word for word in text.split() if word not in STOPWORDS]) # Don't remove stopwords! -> what if trigger contains one?
    text = "".join([c if c not in string.punctuation else " " for c in text]) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespaces
    #text = " ".join([STEMMER.stem(token) for token in text.split()]) # Stemming
    return text


def get_top_words(texts):
    words = [] # All words from all texts
    for t in texts:
        words.extend(t.split())
    return Counter(words) # How many times each one appears


# Trigger identification

def identify_singleword_trigger(clusters, inputs, clean_clusters_idx, poisoned_clusters_idx, 
                                score_method="freq_ratio", mask_string='[MASK]') -> str:
    '''
    Identifies the trigger of the original input assuming it only contains one word.

    Parameters
    ----------
    clusters : list[int]
        Cluster labels.
    inputs : list[str]
        All perturbed (and original) input texts.
    {clean,poisoned}_clusters_idx : list[int]
        Indexes of clean and poisoned clusters, respectively.
    score_method: str
        Method to evaluate the difference between word frequencies in
        clean and poisoned clusters and detect the trigger word.
        Possible values: "freq_diff", "freq_ratio", "freq_over_total",
        "log_odds", "chi_squared", "mutual_info", "voting".
    mask_string : str
        String used to mask inputs.

    Returns
    -------
    str
        Trigger word.
    '''
    cluster_labels = np.array(list(set(clusters)))
    # In case labels don't match with indexes
    clean_cluster_labels = cluster_labels[clean_clusters_idx] 
    poisoned_cluster_labels = cluster_labels[poisoned_clusters_idx]

    # Separate clean and poisoned inputs
    inputs_clean = np.array(inputs)[np.isin(clusters, clean_cluster_labels)] # Clusters w/o trigger
    inputs_clean = [preprocess(text, words_to_remove=[mask_string]) for text in inputs_clean]
    inputs_trigger = np.array(inputs)[np.isin(clusters, poisoned_cluster_labels)] # Clusters w/ trigger
    inputs_trigger = [preprocess(text, words_to_remove=[mask_string]) for text in inputs_trigger]

    # Absolute frequency of words
    poisoned_counts = get_top_words(inputs_trigger)
    clean_counts = get_top_words(inputs_clean)

    # Evaluation
    voting = score_method == "voting"
    triggers = []

    if voting or score_method=="freq_diff":
        score = {
            w: poisoned_counts[w] - clean_counts[w]
            for w in poisoned_counts
        }
        trigger = [k for k,v in score.items() if v==max(score.values())][0]
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="freq_ratio":
        score = {
            w: poisoned_counts[w] / (clean_counts[w] + 1)
            for w in poisoned_counts
        }
        trigger = [k for k,v in score.items() if v==max(score.values())][0]
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="freq_over_total":
        score = {
            w: poisoned_counts[w] / (clean_counts[w] + poisoned_counts[w])
            for w in poisoned_counts
        }
        trigger = [k for k,v in score.items() if v==max(score.values())][0]
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="log_odds":
        EPSILON = 1e-12
        score = {
            w: math.log((EPSILON+(poisoned_counts[w] / poisoned_counts.total())) / (EPSILON+(clean_counts[w] / clean_counts.total())))
            for w in poisoned_counts+clean_counts
        }
        trigger = [k for k,v in score.items() if v==max(score.values())][0]
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="chi_squared":
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(inputs_clean+inputs_trigger)
        y = [0]*len(inputs_trigger) + [1]*len(inputs_clean) # 0=clean, 1=poisoned
        chi_scores, p_values = chi2(X, y)
        trigger = vectorizer.get_feature_names_out()[np.argmax(chi_scores)]
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="mutual_info":
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(inputs_clean+inputs_trigger)
        y = [0]*len(inputs_trigger) + [1]*len(inputs_clean) # 0=clean, 1=poisoned
        mi = mutual_info_classif(X, y, discrete_features=True)
        trigger = vectorizer.get_feature_names_out()[np.argmax(mi)]
        if voting:
            triggers.append(trigger)
    
    # Most voted trigger
    if voting:
        trigger = max(set(triggers), key=triggers.count) 

    return trigger


def identify_multiword_trigger(clusters, inputs, clean_clusters_idx, poisoned_clusters_idx, 
                               max_diff_from_top, score_method="freq_ratio", mask_string='[MASK]') -> str:
    '''
    Identifies the trigger of the original input, which can contain more than one word.

    Parameters
    ----------
    clusters : list[int]
        Cluster labels.
    inputs : list[str]
        All perturbed (and original) input texts.
    {clean,poisoned}_clusters_idx : list[int]
        Indexes of clean and poisoned clusters, respectively.
    max_diff_from_top: float
        Maximum relative difference in the evaluation score between the first word and the subsequent ones
        for them to be considered as part of the trigger.
    score_method: str
        Method to evaluate the difference between word frequencies in
        clean and poisoned clusters and detect the trigger word.
        Possible values: "freq_diff", "freq_ratio", "freq_over_total",
        "log_odds", "chi_squared", "mutual_info", "voting".
    mask_string : str
        String used to mask inputs.

    Returns
    -------
    str
        Trigger word or sequence in alphabetical order.
    '''
    EPSILON = 1e-12

    cluster_labels = np.array(list(set(clusters)))
    # In case labels don't match with indexes
    clean_cluster_labels = cluster_labels[clean_clusters_idx] 
    poisoned_cluster_labels = cluster_labels[poisoned_clusters_idx]

    # Separate clean and poisoned inputs
    inputs_clean = np.array(inputs)[np.isin(clusters, clean_cluster_labels)] # Clusters w/o trigger
    inputs_clean = [preprocess(text, words_to_remove=[mask_string]) for text in inputs_clean]
    inputs_trigger = np.array(inputs)[np.isin(clusters, poisoned_cluster_labels)] # Clusters w/ trigger
    inputs_trigger = [preprocess(text, words_to_remove=[mask_string]) for text in inputs_trigger]

    # Absolute frequency of words
    poisoned_counts = get_top_words(inputs_trigger)
    clean_counts = get_top_words(inputs_clean)

    # Evaluation
    voting = score_method == "voting"
    triggers = []

    if voting or score_method=="freq_diff":
        score = {
            w: poisoned_counts[w] - clean_counts[w]
            for w in poisoned_counts
        }
        max_score = [v for k,v in score.items() if v==max(score.values())][0]
        trigger = " ".join(sorted([k for k,v in score.items() if abs(v-max_score)/(max_score+EPSILON) < max_diff_from_top]))
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="freq_ratio":
        score = {
            w: poisoned_counts[w] / (clean_counts[w] + 1)
            for w in poisoned_counts
        }
        max_score = [v for k,v in score.items() if v==max(score.values())][0]
        trigger = " ".join(sorted([k for k,v in score.items() if abs(v-max_score)/(max_score+EPSILON) < max_diff_from_top]))
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="freq_over_total":
        score = {
            w: poisoned_counts[w] / (clean_counts[w] + poisoned_counts[w])
            for w in poisoned_counts
        }
        max_score = [v for k,v in score.items() if v==max(score.values())][0]
        trigger = " ".join(sorted([k for k,v in score.items() if abs(v-max_score)/(max_score+EPSILON) < max_diff_from_top]))
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="log_odds":
        score = {
            w: math.log((EPSILON+(poisoned_counts[w] / poisoned_counts.total())) / (EPSILON+(clean_counts[w] / clean_counts.total())))
            for w in poisoned_counts+clean_counts
        }
        max_score = [v for k,v in score.items() if v==max(score.values())][0]
        trigger = " ".join(sorted([k for k,v in score.items() if abs(v-max_score)/(max_score+EPSILON) < max_diff_from_top]))
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="chi_squared":
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(inputs_clean+inputs_trigger)
        y = [0]*len(inputs_trigger) + [1]*len(inputs_clean) # 0=clean, 1=poisoned
        chi_scores, p_values = chi2(X, y)

        score = {word: score for word, score in zip(vectorizer.get_feature_names_out(), chi_scores)}
        max_score = [v for k,v in score.items() if v==max(score.values())][0]
        trigger = " ".join(sorted([k for k,v in score.items() if abs(v-max_score)/(max_score+EPSILON) < max_diff_from_top]))
        if voting:
            triggers.append(trigger)

    elif voting or score_method=="mutual_info":
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(inputs_clean+inputs_trigger)
        y = [0]*len(inputs_trigger) + [1]*len(inputs_clean) # 0=clean, 1=poisoned
        mi = mutual_info_classif(X, y, discrete_features=True)

        score = {word: score for word, score in zip(vectorizer.get_feature_names_out(), mi)}
        max_score = [v for k,v in score.items() if v==max(score.values())][0]
        trigger = " ".join(sorted([k for k,v in score.items() if abs(v-max_score)/(max_score+EPSILON) < max_diff_from_top]))
        if voting:
            triggers.append(trigger)
    
    # Most voted trigger
    if voting:
        trigger = max(set(triggers), key=triggers.count) 

    return trigger


## Poison detection ##

def is_poisoned(entropy_per_cluster:list, max_difference:float=5) -> bool:
    ''' Detects if the input is poisoned based on the relative differences in the cluster mean entropies'''
    if len(entropy_per_cluster) == 1:
        return False

    relative_differences = np.array([])
    for i in range(len(entropy_per_cluster)-1):
        for j in range(i+1, len(entropy_per_cluster)):
            quotient = entropy_per_cluster[i]/entropy_per_cluster[j]
            relative_differences = np.append(relative_differences, quotient if quotient>1 else 1/quotient)
    
    if (relative_differences > max_difference).any():
        return True
    
    return False


# Single input functions

def input_poison_and_trigger_detection_classification(input, classifier_pipeline, n_classes, batch_size=32,
                                                      perturbation_method='mask', perturbation_rate=0.7, 
                                                      tokenizer=None, n_input_versions=50, mask_string='[MASK]',
                                                      new_dim=2, dbscan_epsilon=3, dbscan_minpts=5,
                                                      entropy_threshold=4, trigger_score_method='log_odds', 
                                                      max_diff_from_top_trigger=0.5, seed=42):
    '''
    Performs EntroClust poison and trigger detection in an input of a classification dataset.

    Parameters
    ----------
    input : str
        Input text.
    classifier_pipeline : transformers.Pipeline
        Text classification pipeline.
    n_classes : int
        Number of classes of the classification model.
    batch_size : int
        Batch size used to obtain the model's predictions of the perturbed inputs.
    perturbation_method : str
        Method that will be used to obtain the perturbed inputs. Can be 'mask' or 'replace'.
    perturbation_rate : float
        Ratio of words in the input that will be perturbed. Must be between 0 and 1.
    tokenizer : transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        Tokenizer of the model. Use only if 'perturbation_method' is set to 'replace'.
    n_input_versions : int
        Number of perturbed versions of the input.
    mask_string : str
        In the case that the pertubation method is set to 'mask', this indicates the string that
        will be used to mask words in the input text.
    new_dim : int
        Number of components of the probability vectors when reduced by UMAP.
    dbscan_epsilon : float
        Epsilon parameter of the DBSCAN method.
    dbscan_minpts : int 
        Number of samples in a neighborhood for a point to be considered as a core point in DBSCAN.
    entropy_threshold : float
        Maximum relative difference of the entropy of clusters for the input to be considered poisoned.
    trigger_score_method : str
        Method that will be used to determine which word or words are the trigger.
        Refer to the method 'identify_trigger' for all possible values.
    max_diff_from_top_trigger : float
        Maximum relative difference in the evaluation score between the first word and the subsequent ones
        for them to be considered as part of the trigger.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    tuple[bool, str]
        Boolean indicating whether the input is poisoned or not. If it's poisoned,
        the string will indicate the trigger word. Otherwise, the string will be empty.
    '''
    if perturbation_method == 'mask':
        perturbation_fn = mask
        perturbation_args = {
            'mask_rate': perturbation_rate,
            'mask_string': mask_string
        } 
    elif perturbation_method == 'replace':
        perturbation_fn = replace
        perturbation_args = {
            'replace_rate': perturbation_rate,
            'tokenizer': tokenizer
        }
    else:
        raise ValueError("The pertubation method value must be either 'mask' or 'replace'.")
    
    np.random.seed(seed)
    
    # Perturb input
    inputs = [input] + [perturbation_fn(input, **perturbation_args) for i in range(n_input_versions)]
    # Obtain model predictions
    preds = classifier_pipeline(inputs,  batch_size=batch_size, padding=True, truncation=True, top_k=n_classes)
    # Sort the labels
    ordered_preds = [sorted(pred, key=lambda label_score: label_score['label']) for pred in preds]
    # Get the probability vector
    preds_scores = np.array([[label_score['score'] for label_score in pred] for pred in ordered_preds])

    # Clustering
    clusters, entropy_list, silhouette_list, kl_divergences = clustering(
                                                                        preds_scores, 
                                                                        new_dim=new_dim,
                                                                        epsilon=dbscan_epsilon, 
                                                                        minPts=dbscan_minpts
                                                                        )
    # Poison detection
    pred = is_poisoned(entropy_list, max_difference=entropy_threshold)
    
    # Trigger detection
    if pred:
        clean_clusters_idx, poisoned_clusters_idx = get_poisoned_clusters(entropy_list, max_difference=entropy_threshold)
        trigger = identify_multiword_trigger(clusters, inputs, clean_clusters_idx, poisoned_clusters_idx, 
                                             max_diff_from_top=max_diff_from_top_trigger, score_method=trigger_score_method)
        #trigger = identify_singleword_trigger(clusters, inputs, clean_clusters_idx, poisoned_clusters_idx, 
        #                                      score_method=trigger_score_method)
    else:
        trigger = ""

    return pred, trigger


def input_poison_detection_generative(input, model, tokenizer, 
                                   perturbation_method='mask', perturbation_rate=0.5, 
                                   n_input_versions=80, max_new_tokens=5, mask_string='[MASK]',
                                   new_dim=2, dbscan_epsilon=3, dbscan_minpts=5, 
                                   entropy_threshold=2, seed=42
                                   ) -> bool:
    '''
    Performs EntroClust poison detection in an input of a generative dataset.

    Parameters
    ----------
    input : str
        Input text.
    model : transformers.PreTrainedModel
        Text generation model.
    tokenizer : transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        Tokenizer of the model.
    perturbation_method : str
        Method that will be used to obtain the perturbed inputs. Can be 'mask' or 'replace'.
    perturbation_rate : float
        Ratio of words in the input that will be perturbed. Must be between 0 and 1.
    n_input_versions : int
        Number of perturbed versions of the input.
    max_new_tokens : int
        Number of tokens that will be generated in order to detect if the input is poisoned.
    mask_string : str
        In the case that the pertubation method is set to 'mask', this indicates the string that
        will be used to mask words in the input text.
    new_dim : int
        Number of components of the probability vectors when reduced by UMAP.
    dbscan_epsilon : float
        Epsilon parameter of the DBSCAN method.
    dbscan_minpts : int 
        Number of samples in a neighborhood for a point to be considered as a core point in DBSCAN.
    entropy_threshold : float
        Maximum relative difference of the entropy of clusters for the input to be considered poisoned.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    bool
        Boolean indicating whether the input is poisoned or not.
    '''
    if perturbation_method == 'mask':
        perturbation_fn = mask
        perturbation_args = {
            'mask_rate': perturbation_rate,
            'mask_string': mask_string
        } 
    elif perturbation_method == 'replace':
        perturbation_fn = replace
        perturbation_args = {
            'replace_rate': perturbation_rate,
            'tokenizer': tokenizer
        }
    else:
        raise ValueError("The pertubation method value must be either 'mask' or 'replace'.")
    
    np.random.seed(seed)
    
    inputs = [input] + [perturb_alpaca(input, perturbation_fn, perturbation_args) for i in range(n_input_versions)]
    tokenized_inputs = tokenizer(inputs, padding=True, return_tensors="pt").to(model.device)
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        top_k=0
    )
    probs = [F.softmax(s, dim=-1).cpu() for s in outputs.scores]
    
    for i in range(len(probs)):
        clusters, entropy_lst, silhouette_lst, kl_divergences = clustering(probs[i], new_dim=new_dim, epsilon=dbscan_epsilon, minPts=dbscan_minpts)
        if is_poisoned(entropy_lst, entropy_threshold):
            return True
    return False


# Dataset functions

def dataset_poison_and_trigger_detection_classification(dataset, classifier_pipeline, n_classes, batch_size=32,
                                                        perturbation_method='mask', perturbation_rate=0.7, 
                                                        tokenizer=None, n_input_versions=50, mask_string='[MASK]',
                                                        new_dim=2, dbscan_epsilon=3, dbscan_minpts=5,
                                                        entropy_threshold=4, trigger_score_method='log_odds', 
                                                        max_diff_from_top_trigger=0.5, seed=42):
    '''
    Performs EntroClust poison and trigger detection in a classification dataset.
    For the parameters' description, see function input_poison_and_trigger_detection_classification.

    Returns
    -------
    tuple[list[bool],list[str]]
        List of predictions indicating whether an input is poisoned or not and the identified trigger sequence. 
        If an input is not poisoned, the trigger will be an empty string.
    '''

    poison_preds = []
    trigger_preds = []
    for example in tqdm(dataset):
        text = example['text']
        pred, trigger = input_poison_and_trigger_detection_classification(text, classifier_pipeline, n_classes, batch_size,
                                                          perturbation_method, perturbation_rate, tokenizer, 
                                                          n_input_versions, mask_string, new_dim, dbscan_epsilon,
                                                          dbscan_minpts, entropy_threshold, trigger_score_method, 
                                                          max_diff_from_top_trigger, seed)
        poison_preds.append(pred)
        trigger_preds.append(trigger)

    return poison_preds, trigger_preds

def dataset_poison_detection_generative(dataset, model, tokenizer, 
                                        perturbation_method='mask', perturbation_rate=0.5, 
                                        n_input_versions=80, max_new_tokens=5, mask_string='[MASK]',
                                        new_dim=2, dbscan_epsilon=2, dbscan_minpts=5, 
                                        entropy_threshold=8, seed=42
                                        ) -> bool:
    '''
    Performs EntroClust poison detection in a generative dataset.
    For the parameters' description, see function input_poison_detection_generative.

    Returns
    -------
    list[bool]
        List of predictions indicating whether an input is poisoned.
    '''

    y_pred = []
    for example in tqdm(dataset):
        text = example['inference_text']
        pred = input_poison_detection_generative(text, model, tokenizer, perturbation_method,
                                          perturbation_rate, n_input_versions, max_new_tokens, 
                                          mask_string, new_dim, dbscan_epsilon, dbscan_minpts,
                                          entropy_threshold, seed)
        y_pred.append(pred)

    return y_pred

