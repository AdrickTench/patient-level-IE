import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import numpy as np
import random

"""
Code taken from https://github.com/prabaey/SynSUM/blob/main/utils/neural_classifier.py
"""

class TextEmbClassifier(torch.nn.Module):
    """
    Classifier that takes text embeddings as an input, and outputs a symptom logit

    - n_emb: dimension of text embedding input
    - hidden_dim: list of dimensions of hidden layers. if empty, no transformation is applied. if len>0, final dimension should be 1
    - dropout_prob: dropout probability to be applied before every hidden layer.
    - seed: initialization seed
    """
     
    def __init__(self, n_emb, hidden_dim, dropout_prob, seed): 

        super(TextEmbClassifier, self).__init__()

        torch.manual_seed(seed)

        self.n_emb = n_emb # embedding size of text 
        self.hidden_dim = hidden_dim # hidden dimension, if len == 0 then no transformation is applied
                                     # if len != 0, final dimension should be 1 to allow for classification

        # initialize parameters
        if len(hidden_dim) == 0: 
            self.linears = torch.nn.ModuleList([])
        else:  
            self.linears = torch.nn.ModuleList([torch.nn.Linear(self.n_emb, hidden_dim[0])])
            self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout_prob)])
            prev_dim = hidden_dim[0]
            for dim in hidden_dim[1:]: 
                layer = torch.nn.Linear(prev_dim, dim)
                dropout = torch.nn.Dropout(p=dropout_prob)
                self.linears.append(layer)
                self.dropouts.append(dropout)
                prev_dim = dim
            self.hidden_activation = torch.nn.ReLU() # ReLU is used as activation after every hidden layer

        # if self.hidden_dim[-1] == 3: # fever has 3 possible classes (none, low, high) -> use softmax
        #     self.softmax = True
        # else: # other symptoms are binary -> can use sigmoid
        #     self.softmax = False

    def forward(self, emb): 
        """
        forward function. transforms embedding of dim n_emb to output of size 1 (or 3) by applying linear layers
        """

        if len(self.hidden_dim) == 0: 
            return emb
        else: 
            out = emb
            for i, layer in enumerate(self.linears[:-1]): 
                out = self.dropouts[i](out)
                out = layer(out)
                out = self.hidden_activation(out) # ReLU for activation between hidden layers
            out = self.dropouts[-1](out) # if only one layer, dropout should be applied to inputs
            out = self.linears[-1](out)

            # if self.softmax: 
            #     return torch.nn.functional.softmax(out, dim=1) # softmax at the end (multiclass)
            # else:
            #     return out.sigmoid() # sigmoid at the end (binary)

        return out
            

class EmbeddingDataset(Dataset):
    """
    Turn each patient record into a torch Dataset item 
    - df: the original dataframe containing the patient features (symptoms, tabular features, text notes)
    - sympt: the symptom label we are trying to predict from the text 
    - device: what device to put the tensors on (CPU or GPU)
    - type: embedding type to use (hist, phys, both_mean, both_concat)
    - clean: whether to use the clean version of the dataset (only use samples where symptom is mentioned in the text)
    - BN_pred_col: name of BN prediction column to include in the datapoint (not used in this project)
    - with_tab: whether to generate tabular feature vectors
    - encoder: encoder for binary tabular features (fit on train set, use same one on test set)
    - scaler: scaler for numeric tabular features (fit on train set, use same one on test set)
    """
    def __init__(self, df, sympt, device, type="embedding", clean=False, BN_pred_col="", with_tab=False, encoder=None, scaler=None):
        
        self.sympt = sympt
        self.device = device
        self.type = type
        self.clean = clean
        self.BN_pred_col = BN_pred_col
        self.with_tab = with_tab

        if self.clean: 
            self.df = df[df[f"{sympt}_mention"] == True].copy() # only select samples where mention labels available
        else: 
            self.df = df.copy()
        self.df[self.sympt] = self.df[self.sympt].replace({"yes":1, "no": 0, "none":0, "low": 1, "high": 2})

        if self.with_tab: 
            self.features = ["asthma", "smoking", "COPD", "season", "hay_fever", "pneu", "common_cold", "antibiotics", "days_at_home"]
            
            # one-hot encoding
            X_cat = df[[feat for feat in self.features if feat != "days_at_home"]]
            if encoder is None:
                self.enc = OneHotEncoder(drop='if_binary', handle_unknown='ignore')
                self.enc.fit(X_cat)
            else: # encoder was fit on training data, now applied to test data
                self.enc = encoder
            self.X_cat = self.enc.transform(X_cat).toarray() # one-hot encoded version of categorical features

            # scaling of days at home feature
            if scaler is None: 
                self.scaler = StandardScaler()
                self.scaler.fit(df[["days_at_home"]])
            else: # scaler was fit on training data, now applied to test data
                self.scaler = scaler 
            self.X_days = self.scaler.transform(df[["days_at_home"]]) # standard scaled version of days at home feature


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        returns a dictionary with the following components: 
        - sympt: contains a tensor with the symptom values for sympt
        - emb: contains the text embedding for the note, constructed using one of the four strategies encoded in self.type
        - df_idx: index of the sample in the original dataframe
        """

        x = {}

        sympt = self.df.iloc[idx][self.sympt]
        x[self.sympt] = torch.tensor(sympt, dtype=torch.float32, device=self.device)

        # tabular features
        if self.with_tab: 
            x_cat = self.X_cat[idx]
            x_days = self.X_days[idx][0]
            x["tab"] = torch.tensor(np.concatenate((x_cat, [x_days])), dtype=torch.float32, device=self.device)

        # create note embedding based on requested type
        x["emb"] = torch.tensor(self.df.iloc[idx][self.type], device=self.device)

        # get dataframe index at position idx (these are not the same!)
        x["idx"] = self.df.index[idx] 

        # retrieve BN prediction (not used in this project)
        if len(self.BN_pred_col) != 0: 
            x["BN_pred"] = torch.tensor(self.df.iloc[idx][self.BN_pred_col], device=self.device)

        return x
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_sympt_classifier(train, val, sympt, device, n_emb, hidden_dim, dropout, bs_train=100, epochs=100, seed=2023, lr=0.0001, weight_decay=1e-5, with_tab=False, patience=5, patience_tol=1e-3):
    """Train symptom classifier

    - train: training data, EmbeddingDataset object
    - val: validation data, EmbeddingDataset object
    - sympt: name of symptom we want to predict
    - device: CPU or GPU device on which the tensors are loaded 
    - n_emb: dimension of embedding 
    - hidden_dim: list of dimensions to use for the hidden layers of the classifier
    - dropout: dropout probability, used between each hidden layer of the classifier
    - bs_train: batch size 
    - epochs: number of epochs
    - seed: random seed
    - lr: learning rate
    - weight_decay: L2 regularization level
    - with_tab: whether to include tabular features at input 
    - patience: early stopping patience
    - patience_tol: early stopping tolerance

    returns
    - train_loss: cross-entropy score over train set for every epoch
    - val_loss: cross-entropy score over validation set for every epoch
    - stop_epochs: number of epochs after which early stopping was applied
    - model: trained model
    """
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    train_loader = DataLoader(train, batch_size=bs_train, shuffle=True)
    if val is not None:
        val_loader = DataLoader(val, batch_size=len(val), shuffle=False)

    # put model on the device
    model = TextEmbClassifier(n_emb, hidden_dim, dropout, seed)
    model.to(device)

    adam = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    if sympt == "fever": 
        loss = torch.nn.CrossEntropyLoss(reduction="none")
    else: 
        loss = torch.nn.BCEWithLogitsLoss(reduction="none") # all symptoms are binary, except for fever 

    train_loss = []
    val_loss = []

    # Early stopping variables
    early_stopper = EarlyStopper(patience, patience_tol)
    stop_epochs = epochs

    for epoch in range(epochs):

        epoch_loss = 0

        for i, x in enumerate(train_loader): 

            model.train() # put model in train mode
            adam.zero_grad()

            if with_tab:
                input = torch.cat((x["tab"], x["emb"]), dim=1) # concatenate tabular features and text
            else: 
                input = x["emb"]

            if sympt == "fever": 
                pred = model(input) # predictions of model, shape (bs, 3)
                batch_loss = loss(pred, x[sympt].long()).sum()
            else: 
                pred = model(input).squeeze(1) # predictions of model, shape (bs,)
                batch_loss = loss(pred, x[sympt]).sum()
            
            batch_loss.backward()

            epoch_loss += batch_loss.item()
            
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            adam.step()
        
        train_loss.append(epoch_loss/len(train))

        if val is not None: 
            model.eval() # put model in eval mode
            with torch.no_grad():
                for x_val in val_loader: 

                    if with_tab:
                        input = torch.cat((x_val["tab"], x_val["emb"]), dim=1) # concatenate tabular features and text
                    else: 
                        input = x_val["emb"]

                    if sympt == "fever": 
                        pred = model(input) # predictions of model, shape (bs, 3)
                        batch_loss = loss(pred, x_val[sympt].long()).sum()
                    else: 
                        pred = model(input).squeeze(1) # predictions of model, shape (bs,)
                        batch_loss = loss(pred, x_val[sympt]).sum()
                    epoch_val_loss = batch_loss.item()/len(val)
                    val_loss.append(epoch_val_loss)

            # Early stopping check
            if early_stopper.early_stop(epoch_val_loss): 
                stop_epochs = epoch - early_stopper.counter
                break

        else: 
            val_loss = []

    return train_loss, val_loss, stop_epochs, model

import pandas as pd
from torch.nn.functional import softmax

def add_text_predictions_to_df(df, dataset, classifier, symptom, with_tab=False):
    # create new column
    col_name = f"text_prob_{symptom}"
    df[col_name] = np.nan
    df[col_name] = df[col_name].astype(object)
    
    classifier.eval() # put model in eval mode
    
    # iterate through test samples
    for x in DataLoader(dataset, batch_size=1, shuffle=False):
        # get text classifier prediction
        input = torch.cat((x["tab"], x["emb"]), dim=1) if with_tab else x["emb"]
        logits = classifier(input.squeeze(dim=0)).squeeze()  
        pred = softmax(logits, dim=0) if symptom == "fever" else logits.sigmoid() # P(sympt = 1 | text)
    
        if symptom == "fever": 
            idx = x["idx"]
            df.at[idx.item(), col_name] = np.round(pred.flatten().detach().cpu().numpy().tolist(), 8) # add prediction to dataframe
        else: 
            idx = x["idx"]
            df.loc[idx, col_name] = round(pred.item(), 8) # add prediction to dataframe

def add_all_classifier_predictions_to_df(df, text_classifiers, device, emb_type="embedding", with_tab=False):
    for symptom, classifier in text_classifiers.items():
        dataset = EmbeddingDataset(df, symptom, device, type=emb_type, with_tab=with_tab)
        add_text_predictions_to_df(df, dataset, classifier, symptom, with_tab=with_tab)
    return df

def get_nn_predictions(df, nn_classifier, symptom, device, emb_type="embedding", with_tab=False, encoder=None, scaler=None):
    col_name = f"text_prob_{symptom}"
    dataset = EmbeddingDataset(df, symptom, device, type=emb_type, with_tab=with_tab, encoder=encoder, scaler=scaler)
    results_df = pd.DataFrame(index=df.index)
    add_text_predictions_to_df(results_df, dataset, nn_classifier, symptom, with_tab=with_tab)
    return results_df[col_name]