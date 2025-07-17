from sklearn.model_selection import train_test_split, StratifiedKFold

import numpy as np
import pandas as pd
import pickle
import warnings
from collections import defaultdict
import argparse

import torch
torch.use_deterministic_algorithms(True)

from utils.bayesian_network import TrainedBayesianNetwork, GTBayesianNetwork
from utils.neural_classifier import EmbeddingDataset, train_sympt_classifier, add_all_classifier_predictions_to_df, get_nn_predictions
from utils.consistency_node import ConsistentBNTextModel

def factory(n):
    if n:
        return lambda: defaultdict(factory(n-1))
    else:
        return int
    
def to_dict(d):
    if type(d) is defaultdict:
        return { k: to_dict(v) for k,v in d.items() }
    else:
        return d

def train_text_classifier(df, sympt, device, seed, with_tab=False, emb_type="embedding"):
    """
    Train text classifier. Use cross-validation to determine early stopping, then retrain over full training set. 
    - df: dataframe containing training data
    - sympt: symptom which we want to predict
    - device: what device to put the tensors on (CPU or GPU)
    - seed: seed used to initialize the neural classifier weights
    - with_tab: whether to include tabular features at input
    - emb_type: the type of embedding to use
    returns: 
    - text_classifier: neural text classifier trained on full df
    """
    
    # cross-validation setup
    num_splits = 5
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    # hyperparameters
    epochs = 200
    if with_tab: 
        n_emb = 768+9 # 8 binary features and 1 discrete (days at home)
    else: 
        n_emb = 768
    dropout = 0
    weight_decay = 1e-5
    bs_train = 50
    if sympt != "fever": 
        hidden_dim = [n_emb, 256, 1]
    else: 
        hidden_dim = [n_emb, 256, 3]
    lr = 0.0005
    patience = 10
    patience_tol = 1e-3
    hyperparams = {"n_emb": n_emb, "dropout": dropout, "seed": seed, "weight_decay": weight_decay, "bs_train": bs_train, "hidden_dim": hidden_dim, "epochs": epochs, "lr": lr,
                "patience": patience, "patience_tol": patience_tol}
    
    # train over cross-validation folds to determine early stopping
    stop_epochs = []
    labels = df[sympt].values
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        df_train_subset = df.iloc[train_idx]
        df_val_subset = df.iloc[val_idx]

        # create dataset
        data_train = EmbeddingDataset(df_train_subset, sympt, device, type=emb_type, with_tab=with_tab)
        if with_tab:
            data_val = EmbeddingDataset(df_val_subset, sympt, device, type=emb_type, with_tab=with_tab, encoder=data_train.enc, scaler=data_train.scaler)
        else: 
            data_val = EmbeddingDataset(df_val_subset, sympt, device, type=emb_type, with_tab=with_tab)

        # train model
        _, _, stop_epochs_fold, _ = train_sympt_classifier(data_train, data_val, sympt, device, with_tab=with_tab, **hyperparams)
        stop_epochs.append(stop_epochs_fold)
    median_epochs = sorted(stop_epochs)[len(stop_epochs)//2]

    # retrain with full train set
    print(f"retrain for {median_epochs} epochs")
    data_train = EmbeddingDataset(df, sympt, device, type=emb_type, with_tab=with_tab)
    hyperparams["epochs"] = median_epochs
    _, _, _, text_classifier = train_sympt_classifier(data_train, None, sympt, device, with_tab=with_tab, **hyperparams)

    if with_tab: 
        return text_classifier, data_train.enc, data_train.scaler # return tabular data encoder and scaler for reuse on test set
    else: 
        return text_classifier

def train_BN_classifier(df, seed): 
    """
    Learning Bayesian network parameters from training data. 
    - df: dataframe containing training data
    - seed: seed used to initialize the Bayesian network parameters
    returns: 
    - learn_BN: trained Bayesian network
    """
    learn_BN = TrainedBayesianNetwork(df, seed)
    learn_BN.learn_full_BN()
    return learn_BN

def train_consistency(text_classifiers, BN_classifier, calibration_df, symptom, device, virtual_evidence=False, weighted_agr=True):
    eval_class = ConsistentBNTextModel(text_classifiers, BN_classifier, symptom)
    calibration_dataset = EmbeddingDataset(calibration_df, symptom, device, type="embedding")
    if weighted_agr: 
        eval_class.get_weighted_agreement(calibration_df, calibration_dataset, virtual_evidence=virtual_evidence)
    else:
        eval_class.get_agreement(calibration_df, calibration_dataset, virtual_evidence=virtual_evidence)
    return eval_class

def train_predict_consistency(text_classifiers, BN_classifier, calibration_df, test_df, symptom, device, virtual_evidence=False, weighted_agr=True, emb_type="embedding"):
    consistency_classifier = train_consistency(text_classifiers, BN_classifier, calibration_df, symptom, device, virtual_evidence, weighted_agr)
    consistency_classifier.eval_df = None
    test_dataset = EmbeddingDataset(test_data, symptom, device, type=emb_type)
    consistency_classifier.add_text_prediction(test_dataset, test_df)
    consistency_classifier.add_BN_prediction("BN_prob", virtual_evidence=virtual_evidence)
    consistency_classifier.combine_prob("BN_prob")
    return consistency_classifier.eval_df["comb_text_BN_prob"]

def load_simsum():
    df_1 = pd.read_pickle("simsum/simsum_data_shift_mentions_1.p")
    df_2 = pd.read_pickle("simsum/simsum_data_shift_mentions_2.p")
    df = pd.concat((df_1, df_2))
    return df

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples_range', nargs='+', type=int, default=[100, 187, 350, 654, 1223, 2287, 4278, 8000])
    parser.add_argument('--seeds', nargs='+', type=int, default=[212, 1119, 1029, 80, 1002, 1339, 627, 2014, 124, 963, 42, 69, 420, 541, 834, 716, 32, 84, 621, 29])
    args = parser.parse_args()

    # set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore")
    symptoms = ["dysp", "cough", "pain", "nasal", "fever"]
    evidence_keys=["asthma", "smoking", "COPD", "season", "hay_fever", "pneu", "common_cold", "antibiotics", "days_at_home"]
    filename = 'probabilities_seeds_' + '_'.join([str(x) for x in args.seeds])
    print(filename)

    # load in data
    df = load_simsum()

    # ground-truth BN
    gtbn = GTBayesianNetwork()

    train_data_tmp, test_data_tmp = train_test_split(df, test_size=0.2, random_state=2024)

    results = defaultdict(factory(4)) # n_samples -> seed -> model -> symptom -> probabilities
    for seed in args.seeds:
        # split into train / test / calibration        
        test_data = test_data_tmp.copy()
        data_shift_test_data = test_data.copy()

        for n_samples in args.n_samples_range:
            if n_samples < 8000:
                train_data, _ = train_test_split(train_data_tmp, train_size=n_samples, random_state=seed)
            else:
                train_data = train_data_tmp.copy()
            calibration_data = train_data.copy()
            data_shift_calibration_data = calibration_data.copy()

            # BN
            bn = train_BN_classifier(train_data, seed)

            # NNs
            text_classifiers = {}
            for symptom in symptoms:
                text_classifier = train_text_classifier(train_data, symptom, device, seed)
                text_classifiers[symptom] = text_classifier

            # store NN predictions
            add_all_classifier_predictions_to_df(calibration_data, text_classifiers, device)
            add_all_classifier_predictions_to_df(data_shift_calibration_data, text_classifiers, device)

            add_all_classifier_predictions_to_df(test_data, text_classifiers, device)
            add_all_classifier_predictions_to_df(data_shift_test_data, text_classifiers, device, emb_type="redacted_embedding")

            for symptom in symptoms:
                # BN predictions
                results[n_samples][seed]['bn_realistic'][symptom] = test_data.apply(bn.predict_symptom, axis=1, args=(symptom, evidence_keys, False))

                # GT predictions
                results[n_samples][seed]['gt_bn'][symptom] = test_data.apply(gtbn.predict_symptom, axis=1, args=(symptom, evidence_keys, False))

                # NN predictions
                text_label = f"text_prob_{symptom}"
                results[n_samples][seed]['binary_classifiers'][symptom] = test_data[text_label]
                results[n_samples][seed]['binary_classifiers_data_shift'][symptom] = data_shift_test_data[text_label]

                # tabular-text NN predictions
                tabular_text_classifier, encoder, scaler = train_text_classifier(train_data, symptom, device, seed, with_tab=True)
                results[n_samples][seed]['tabular_text_binary'][symptom] = get_nn_predictions(test_data, tabular_text_classifier, symptom, device, with_tab=True, encoder=encoder, scaler=scaler)

                # consistency predictions
                results[n_samples][seed]['weighted_consistency'][symptom] = train_predict_consistency(None, bn, calibration_data, test_data, symptom, device)
                results[n_samples][seed]['weighted_consistency_data_shift'][symptom] = train_predict_consistency(None, bn, data_shift_calibration_data, data_shift_test_data, symptom, device, emb_type="redacted_embedding")
                results[n_samples][seed]['weighted_consistency_ground_truth'][symptom] = train_predict_consistency(None, gtbn, calibration_data, test_data, symptom, device)

                # VE predictions
                results[n_samples][seed]['virtual'][symptom] = test_data.apply(bn.predict_symptom, axis=1, args=(symptom, evidence_keys, True))
                results[n_samples][seed]['virtual_data_shift'][symptom] = data_shift_test_data.apply(bn.predict_symptom, axis=1, args=(symptom, evidence_keys, True))
                results[n_samples][seed]['virtual_ground_truth'][symptom] = test_data.apply(gtbn.predict_symptom, axis=1, args=(symptom, evidence_keys, True))

                # consistency + VE predictions
                results[n_samples][seed]['weighted_consistency_virtual'][symptom] = train_predict_consistency(None, bn, calibration_data, test_data, symptom, device, virtual_evidence=True)
                results[n_samples][seed]['weighted_consistency_virtual_data_shift'][symptom] = train_predict_consistency(None, bn, data_shift_calibration_data, data_shift_test_data, symptom, device, emb_type="redacted_embedding", virtual_evidence=True)
                results[n_samples][seed]['weighted_consistency_virtual_ground_truth'][symptom] = train_predict_consistency(None, gtbn, calibration_data, test_data, symptom, device, virtual_evidence=True)
    
    # save results
    with open(filename+'.p', 'wb') as file:
        pickle.dump(to_dict(results), file)
