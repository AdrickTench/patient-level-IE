from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize

import numpy as np
import pandas as pd

from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from utils.neural_classifier import add_text_predictions_to_df

class ConsistentBNTextModel(): 
    """  
    Evaluation class for BN-text model, which is a classifier that takes both predictions of a text classifier and a Bayesian network into account
    - text_model: trained text classifier
    - BN_model: trained BN model
    - sympt: symptom for which to build the classifier
    """

    def __init__(self, text_models, BN_model, sympt): 

        self.text_models = text_models
        self.BN_model = BN_model

        self.sympt = sympt
        self.text_prob_label = f"text_prob_{sympt}"
        self.eval_df = None
        self.symptoms = ["dysp", "cough", "pain", "nasal", "fever"]

    def add_text_prediction(self, data_test, df_test, virtual_evidence=False): 
        """ 
        Add predictions of trained classifier to evaluation dataframe. 
        We add let the classifier predict the symptom for each sample in the test set, and add the prediction to the dataframe.
        - data_test: dataset with test examples
        - df_test: dataframe with test examples
        returns: 
        - eval_df: evaluation dataframe, only including the columns relevant for evaluation and inspection of the results
        """
        
        # prepare evaluation dataframe if there isn't one yet
        if not isinstance(self.eval_df, pd.DataFrame):
            self.eval_df = df_test.copy()

        if self.text_models is not None: # allow possibility of text probs already being saved in the dataframe
            symptoms = [self.sympt] if not virtual_evidence else self.symptoms
            for symptom in symptoms:
                add_text_predictions_to_df(self.eval_df, data_test, self.text_models[symptom], symptom)

        return self.eval_df
    
    def evaluate_predictions(self, col_name, threshold=0.5): 
        """ 
        Evaluate the predictions of the model, where the probabilities in the column "col_name" are turned into labels using the threshold
        - col_name: the name of the column containing the predicted class probability 
        - threshold: the threshold to use for turning probabilities into predicted labels
        returns: 
        - metrics: a dictionary containing relevant metrics calculated for eval_df
        """

        # use the threshold to label the examples
        label_name = f"{col_name}_label"
        if self.sympt == "fever": 
            self.eval_df[label_name] = self.eval_df[col_name].apply(lambda x: ["none", "low", "high"][np.argmax(x)])
        else:
            self.eval_df[label_name] = self.eval_df[col_name].apply(lambda x: "yes" if x > threshold else "no")
            
        # get ground truth vs. predicted labels
        y_true = self.eval_df[self.sympt].values
        y_pred = self.eval_df[label_name].values

        # calculate FPR and FNR for multiclass case
        if self.sympt == "fever": 

            cm = confusion_matrix(y_true, y_pred, labels=["none", "low", "high"])
            fpr = []
            fnr = []

            for i in range(3):
                tp = cm[i, i] # True Positives (TP): Diagonal element
                fn = cm[i, :].sum() - tp # False Negatives (FN): Sum of the row for the class, excluding the diagonal
                fp = cm[:, i].sum() - tp # False Positives (FP): Sum of the column for the class, excluding the diagonal
                tn = cm.sum() - (fp + fn + tp) # True Negatives (TN): Sum of all other elements (not in the class's row or column)
                
                # Calculate FPR and FNR
                fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr_i = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                fpr.append(fpr_i)
                fnr.append(fnr_i)

        # calculate FPR and FNR for binary case
        else: 

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) 
            fnr = fn / (fn + tp)

        # calculate metrics
        if self.sympt == "fever": 
            avg_method = "macro"
            labels = ["none", "low", "high"]
            precision = precision_score(y_true, y_pred, labels=labels, average=avg_method)
            recall = recall_score (y_true, y_pred, labels=labels, average=avg_method)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, labels=labels, average=avg_method)
            
            y_prob = np.stack(self.eval_df[col_name].values)
            y_true = label_binarize(y_true, classes=["none", "low", "high"])
            avg_prec = average_precision_score(y_true, y_prob, average="macro")
        else:
            precision = precision_score(y_true, y_pred, pos_label="yes")
            recall = recall_score (y_true, y_pred, pos_label="yes")
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, pos_label="yes")

            y_prob = self.eval_df[col_name].values
            avg_prec = average_precision_score(y_true, y_prob, pos_label="yes")
        
        metrics = {"precision": precision, "recall": recall, "FPR": fpr, "FNR": fnr, 
                   "accuracy": acc, "f1 score": f1, "avg_precision": avg_prec}
    
        return metrics
    
    def add_BN_prediction(self, col_name, evidence_set=["asthma", "smoking", "COPD", "season", "hay_fever", "pneu", "common_cold", "antibiotics", "days_at_home"], virtual_evidence=False): 
        """ 
        Add Bayesian network prediction probabilities to the evaluation dataframe. 
        - col_name: name of new column that stores the predicted probabilities
        - evidence_set: set of variables to use as evidence in the inference procedure
        returns: 
        - eval_df: evaluation dataframe with BN predictions added as a new column
        """

        exact_inf = self.BN_model.predict_symptom
        self.eval_df[col_name] = np.nan
        self.eval_df[col_name] = self.eval_df[col_name].astype(object)
        self.eval_df[col_name] = self.eval_df.apply(exact_inf, axis=1, args=(self.sympt, evidence_set, virtual_evidence)) # get predicted probability for each symptom

        return self.eval_df
    
    def get_agreement(self, df_cal, data_cal, evidence=["asthma", "smoking", "COPD", "season", "hay_fever", "pneu", "common_cold", "antibiotics", "days_at_home"], virtual_evidence=False, threshold=0.5, print_msg=False, strategy="uniform", cutoff=5): 
        """ 
        Create agreement table by comparing predictions of BN classifier and text classifier. 
        - df_cal: dataframe containing calibration data
        - data_cal: dataset containing calibration data
        - evidence: what evidence set to use when calculating the BN prediction
        - strategy: what strategy to use when few samples are found for a combination in the agreement table (uniform / or)
        - cutoff: strategy is applied when number of samples for a combination in agreement table is lower than cutoff
        returns: 
        - p_agreement: agreement table. first dimension = text prediction, second dimension = BN prediction, last dimension = symptom value
        [
          [[P(sympt=no | text=no, BN=no), P(sympt=yes | text=no, BN=no)],
           [P(sympt=no | text=no, BN=yes), P(sympt=yes | text=no, BN=yes)]],
          [[P(sympt=no | text=yes, BN=no), P(sympt=yes | text=yes, BN=no)],
           [P(sympt=no | text=yes, BN=yes), P(sympt=yes | text=yes, BN=yes)]]
        ]
        """

        # calculate text and BN predictions over evidence 
        self.add_text_prediction(data_cal, df_cal, virtual_evidence=virtual_evidence)
        self.add_BN_prediction("BN_prob", evidence, virtual_evidence=virtual_evidence)

        # determine labels 
        if self.sympt == "fever":
            classes = ["none", "low", "high"]
            self.eval_df["text_label"] = self.eval_df[self.text_prob_label].apply(lambda x: classes[np.argmax(x)])
            self.eval_df["BN_label"] = self.eval_df["BN_prob"].apply(lambda x: classes[np.argmax(x)])
        else: 
            classes = ["no", "yes"]
            self.eval_df["text_label"] = self.eval_df[self.text_prob_label].apply(lambda x: "yes" if x > threshold else "no")
            self.eval_df["BN_label"] = self.eval_df["BN_prob"].apply(lambda x: "yes" if x > threshold else "no")

        # calculate agreement of labels, comparing with ground truth sympt label -> P(S | Cl, BN)
        p_agreement = []
        for i in classes: 
            for j in classes: 
                filtered = self.eval_df[(self.eval_df['text_label'] == i) & (self.eval_df['BN_label'] == j)]
                if len(filtered) <= cutoff: 
                    if strategy == "uniform": 
                        p = [1/len(classes) for _ in classes]
                    elif strategy == "or": 
                        filtered_text = self.eval_df[self.eval_df["text_label"] == i]
                        filtered_BN = self.eval_df[self.eval_df[f"BN_label"] == j]
                        if len(filtered_text) == 0: 
                            p_text = np.array([1/len(classes) for _ in classes])
                        else:
                            p_text = np.array([len(filtered_text[filtered_text[self.sympt] == k])/len(filtered_text) for k in classes])
                        p_BN = np.array([len(filtered_BN[filtered_BN[self.sympt] == k])/len(filtered_BN) for k in classes])
                        p = (p_text + p_BN) # P(S | Cl = i) OR P(S | BN = j)
                        p /= sum(p) # normalize
                    else: 
                        print("unknown strategy")
                else: 
                    p = [len(filtered[filtered[self.sympt] == k])/len(filtered) for k in classes] # P(S | Cl = i, BN = j)
                if print_msg:
                    print(f"text classifier predicts {i}, BN predicts {j}, P({self.sympt}) = {p} ({len(filtered)} samples)")
                p_agreement.append(p)
            
        if self.sympt == "fever":
            self.p_agreement = np.array(p_agreement).reshape(3, 3, 3)
        else: 
            self.p_agreement = np.array(p_agreement).reshape(2, 2, 2)

        return self.p_agreement
    
    def get_weighted_agreement(self, df_cal, data_cal, virtual_evidence=False, evidence=["asthma", "smoking", "COPD", "season", "hay_fever", "pneu", "common_cold", "antibiotics", "days_at_home"]): 
        """ 
        Create weighted agreement table by comparing predictions of BN classifier and text classifier, without thresholding. 
        - df_cal: dataframe containing calibration data
        - data_cal: dataset containing calibration data
        - evidence: what evidence set to use when calculating the BN prediction
        returns: 
        - p_agreement: agreement table. first dimension = text prediction, second dimension = BN prediction, last dimension = symptom value
        [
          [[P(sympt=no | text=no, BN=no), P(sympt=yes | text=no, BN=no)],
           [P(sympt=no | text=no, BN=yes), P(sympt=yes | text=no, BN=yes)]],
          [[P(sympt=no | text=yes, BN=no), P(sympt=yes | text=yes, BN=no)],
           [P(sympt=no | text=yes, BN=yes), P(sympt=yes | text=yes, BN=yes)]]
        ]
        """

        if self.sympt == "fever": 
            classes = ["none", "low", "high"]
        else: 
            classes = ["no", "yes"]

        # calculate text and BN predictions over evidence 
        self.add_text_prediction(data_cal, df_cal, virtual_evidence=virtual_evidence)
        self.add_BN_prediction("BN_prob", evidence, virtual_evidence=virtual_evidence)

        # calculate agreement of labels, comparing with ground truth sympt label -> P(S | Cl, BN)
        if self.sympt == "fever": 
            p_agreement = np.zeros((3, 3, 3))
            for i, class_name in enumerate(classes): 
                filtered = self.eval_df[self.eval_df[self.sympt] == class_name]
                p_text =  np.stack(filtered[self.text_prob_label].values)
                p_BN = np.stack(filtered["BN_prob"].values)
                for j, _ in enumerate(classes): 
                    for k, _ in enumerate(classes): 
                        p_agreement[j, k, i] = np.sum(p_text[:,j]*p_BN[:,k])
        else: 
            p_agreement = np.zeros((2, 2, 2))
            for i, class_name in enumerate(classes): 
                filtered = self.eval_df[self.eval_df[self.sympt] == class_name]
                p_text = filtered[self.text_prob_label]
                p_BN = filtered["BN_prob"]
                p_agreement[0, 0, i] = np.sum((1-p_BN)*(1-p_text))
                p_agreement[0, 1, i] = np.sum((p_BN)*(1-p_text))
                p_agreement[1, 0, i] = np.sum((1-p_BN)*(p_text))
                p_agreement[1, 1, i] = np.sum(p_BN*p_text)

        p_agreement /= np.sum(p_agreement, axis=2, keepdims=True) # normalize row by row 
        self.p_agreement = p_agreement
    
        return self.p_agreement 
    
    def combine_prob(self, BN_col_name): 
        """ 
        Combine predictions of the BN and the text classifier by taking the weighted sum of their product. We use the agreement table
        to weigh each combination. 
        - BN_col_name: name of column where BN predictions are stored
        returns: 
        - eval_df: evaluation dataframe, containing the added combined probability
        """
    
        def get_processed_prob(row): 
        
            if self.sympt == "fever": 
                classes = ["none", "low", "high"]
                p_BN = row[BN_col_name]
                p_text = row[self.text_prob_label]
            else: 
                classes = ["no", "yes"]
                p_BN = [1-row[BN_col_name], row[BN_col_name]]
                p_text = [1-row[self.text_prob_label], row[self.text_prob_label]]
            
            num = np.zeros(len(classes))
            denom = 0
            for i in range(len(classes)): 
                for j in range(len(classes)): 
                    num += np.array([p_text[i]*p_BN[j]*self.p_agreement[i, j, k] for k in range(len(classes))])
                    denom += p_text[i]*p_BN[j]
            
            if self.sympt == "fever": 
                return num/denom
            else: 
                return num[1]/denom
        
        self.eval_df[f"comb_text_{BN_col_name}"] = self.eval_df.apply(get_processed_prob, axis=1)

        return self.eval_df


class EnsembleModel:
    def __init__(self, sympt, model_type="mlp", learning_rate=0.01, max_iter=1000, seed=2024):
        """
        Initialize the ensemble model.
        
        Parameters:
        - model_type: "mlp" for MLP ensemble, "linear" for logistic regression ensemble
        - learning_rate: Learning rate for the MLP model
        - max_iter: Maximum number of iterations for training
        - seed: seed to use as random_state
        """
        self.sympt = sympt
        self.model_type = model_type.lower()
        
        if self.model_type == "mlp":
            # fever: 6 inputs, 4 hidden states, 3 outputs
            # other symptoms: 2 inputs, 4 hidden states, 2 outputs
            self.model = MLPClassifier(hidden_layer_sizes=(4,),  
                                    activation='relu', 
                                    solver='adam', 
                                    learning_rate_init=learning_rate, 
                                    max_iter=max_iter, 
                                    random_state=seed)
        elif self.model_type == "linear":
            self.model = LogisticRegression(solver='lbfgs', random_state=seed) # multiclass is automatically used when needed
        else:
            raise ValueError("Invalid model_type. Choose 'mlp' or 'linear'.")

    def fit(self, df):
        """
        Train the model using the calibration dataset.

        Parameters:
        - df: Pandas DataFrame containing "BN_prob", self.text_prob_label, and symptom labels
        """
        p_BN = df["BN_prob"].to_numpy()
        p_text = df[self.text_prob_label].to_numpy()
        y = df[self.sympt].to_numpy()
        if self.sympt == "fever": 
            p_BN = np.vstack(p_BN)
            p_text = np.vstack(p_text)
            classes = {"none": 0, "low": 1, "high": 2}
            y_enc = np.array([classes[label] for label in y])  # Convert labels
        else: 
            y_enc = np.array([1 if label == "yes" else 0 for label in y])  # Convert labels
        X = np.column_stack((p_BN, p_text))  # Feature matrix

        self.model.fit(X, y_enc)  # Train model

    def ensemble_prob(self, row):
        """
        Compute final probability using the trained model.

        Parameters:
        - row: dataframe row containing both the BN prob and the text prob
        
        Returns:
        - Combined probability (float)
        """
        p_BN = row["BN_prob"]
        p_text = row[self.text_prob_label]
        if self.sympt == "fever":
            X_input = np.concatenate([p_BN, p_text]).reshape(1,6)  # Prepare input
            return self.model.predict_proba(X_input)[0] # probability of all 3 classes
        else:
            X_input = np.array([[p_BN, p_text]])  # Prepare input
            return self.model.predict_proba(X_input)[0, 1]  # Probability of class 1