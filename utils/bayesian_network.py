import torch
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import itertools
import torch
import random

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference.ExactInference import VariableElimination
from pgmpy.estimators import BayesianEstimator

"""
Code adapted from https://github.com/prabaey/SynSUM/blob/main/utils/bayesian_network.py
"""

class CustomDataset(Dataset):
    """
    Simple dataset used for MLE training loop
    - df: dataframe containing the tabular features
    """
    def __init__(self, df):
        self.df = df.replace({"yes":1, "no": 0, "none":0, "low": 1, "high": 2}) # text feature categories are transformed into numerical categories

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # each item is a dictionary with as keys the names of all tabular variables, followed by their value in tensor format
        x = {}
        for col in self.df.columns:
            x[col] = torch.tensor(self.df.iloc[idx][col], dtype=torch.float32)
        return x

class NoisyOr(torch.nn.Module):
    """
    Noisy OR distribution 
    - outcome: name of symptom we are modeling
    - parents: list of parent variables 
    - lambda_0: leak probability (learned from data if None)
    - lambdas: activation probabilities (learned from data if None)
    """
    
    def __init__(self, outcome, parents, lambda_0=None, lambdas=None): 
        super(NoisyOr, self).__init__()
        self.outcome = outcome # name of variable we are modeling
        self.parents = parents # parent variables
        self.n = len(parents) # number of parents

        # learnable parameters
        self.lambda_0 = torch.nn.Parameter(torch.rand(1)) if lambda_0 == None else torch.nn.Parameter(torch.tensor(lambda_0)) # leak probability
        self.lambdas = torch.nn.Parameter(torch.rand(self.n)) if lambdas == None else torch.nn.Parameter(torch.tensor(lambdas)) # activation probabilities

        self.sigmoid = False if lambda_0 != None else True
        
    def forward(self, sample): 
        """
        calculate the log-probability that a symptom is activated/not activated (depending on its observed value), given its parent values 
        - sample: datapoint (dict) containing symptom and parent values as tensors
        """

        if self.sigmoid:
            lambda_0 = torch.sigmoid(self.lambda_0) # constrain between 0 and 1 
            lambdas = torch.sigmoid(self.lambdas) # constrain between 0 and 1
        else: 
            lambda_0 = self.lambda_0
            lambdas = self.lambdas
        
        y = sample[self.outcome] # select the outcome (symptom that is activated)
        x = torch.stack([sample[parent] for parent in self.parents], dim=1) # select the parents and stack them into a tensor of dim (bs, n)
        
        prod = (1-lambda_0)*torch.prod((1-lambdas)**x, dim=1) # probability that symptom is not active 
        
        log_p = torch.where(y==1, torch.log(1-prod), torch.log(prod)) # probability that symptom is active vs. not active 
        
        return log_p

    def train(self, df, bs=50, lr=0.01, num_epochs=10):
        """
        Estimate parameters of the noisyOR distribution (lambda_0 and lambdas) from dataset 
        - df: dataset containing all tabular variables (dict with variable name as key, and tensor as value)
        - bs: batch size
        - lr: learning rate
        - num_epochs: number of epochs
        """

        df_subset = df[[self.outcome]+self.parents]
        train_data = CustomDataset(df_subset)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

        optimizer = Adam(self.parameters(), lr=lr) 

        train_loss = []
        for _ in range(num_epochs):
            epoch_loss = 0
            n_batch = 0
            for batch in train_loader:

                # Forward pass
                log_prob = self.forward(batch)
                loss = -log_prob.sum() # negative log-likelihood
                epoch_loss += loss
                n_batch += 1

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(epoch_loss.detach().item()/n_batch)

        return train_loss
        

    def get_CPT(self): 
        """
        Generate full conditional probability table from the noisy-OR parameters. 
        Contains P(outcome | parents) for all combinations of parents, where parents are assumed to have two possible values (yes/no)
        First row of table contains P(outcome = yes | parents)
        Second row of table contains P(outcome = no | parents)
        """

        combinations = itertools.product([1, 0], repeat=len(self.parents))

        input = {parent:[] for parent in self.parents}

        for comb in combinations: 
            for i, parent in enumerate(self.parents): 
                input[parent].append(comb[i])

        input[self.outcome] = torch.ones(2**self.n)
        input = {parent:torch.tensor(val) for parent, val in input.items()}

        p_pos = self.forward(input).exp() # prob(outcome | parents) for all combinations of parents
        p_neg = 1-p_pos

        cpt = torch.stack((p_pos, p_neg))

        return cpt.detach().clone().numpy()
    

class Antibiotics(torch.nn.Module):
    """
    Conditional probability distribution for Antibiotics variable, parameterized using a logistic regression model
    - outcome: name of variable we are modeling (antibiotics)
    - parents: list of parent variables 
    - incl_policy: whether to model policy variable (if True, then coeff[0] contains its weight)
    - bias: logistic regression model bias (learned from data if None)
    - coeff: logistic regression model coefficients (learned from data if None)
    """
    
    def __init__(self, outcome, parents, incl_policy=False, bias=None, coeff=None): 

        super(Antibiotics, self).__init__()
        self.outcome = outcome # name of variable we are modeling
        self.parents = parents # parent variables
        self.n = len(parents) # number of parents
        self.incl_policy = incl_policy # whether policy is included in the model

        # learnable parameters
        self.bias = torch.nn.Parameter(torch.rand(1)) if bias == None else torch.nn.Parameter(torch.tensor(bias)) # bias
        self.coeff = torch.nn.Parameter(torch.rand(self.n+1)) if coeff == None else torch.nn.Parameter(torch.tensor(coeff)) # coefficients (2 for fever!)
        
    def forward(self, sample): 
        """
        calculate the log-probability that antibiotics is prescribed/not prescribed (depending on its observed value), given the values of the parent variables
        - sample: datapoint (dict) containing tabular values as tensors
        """

        low_fever = torch.where(sample["fever"] == 1, torch.ones_like(sample["fever"]), torch.zeros_like(sample["fever"]))
        high_fever = torch.where(sample["fever"] == 2, torch.ones_like(sample["fever"]), torch.zeros_like(sample["fever"]))

        y = sample[self.outcome] # select the outcome (antibiotics)
        
        if self.incl_policy: 
            logit = self.bias + self.coeff[0]*sample["policy"] \
                + self.coeff[1]*sample["dysp"] + self.coeff[2]*sample["cough"] \
                + self.coeff[3]*sample["pain"] \
                + self.coeff[4]*low_fever + self.coeff[5]*high_fever
        else: 
            logit = self.bias \
                    + self.coeff[0]*sample["dysp"] + self.coeff[1]*sample["cough"] \
                    + self.coeff[2]*sample["pain"] \
                    + self.coeff[3]*low_fever + self.coeff[4]*high_fever
        prob = torch.sigmoid(logit)
        log_p = torch.where(y==1, torch.log(prob), torch.log(1-prob))

        return log_p
    
    def train(self, df, bs=50, lr=0.01, num_epochs=15):
        """
        Estimate parameters of the Antibiotics distribution (bias and coeff) from dataset 
        - df: dataset containing all tabular variables (dict with variable name as key, and tensor as value)
        - bs: batch size
        - lr: learning rate
        - num_epochs: number of epochs
        """

        df_subset = df[[self.outcome]+self.parents]
        train_data = CustomDataset(df_subset)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

        optimizer = Adam(self.parameters(), lr=lr) 

        train_loss = []
        for _ in range(num_epochs):
            epoch_loss = 0
            n_batch = 0
            for batch in train_loader:

                # Forward pass
                log_prob = self.forward(batch)
                loss = -log_prob.sum() # negative log-likelihood
                epoch_loss += loss
                n_batch += 1

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss.append(epoch_loss.detach().item()/n_batch)

        return train_loss

    def get_CPT(self): 
        """
        Generate full conditional probability table from the regression model parameters. 
        Contains P(outcome | parents) for all combinations of parents, where parents are assumed to have two possible values (yes/no), except for fever
        First row of table contains P(outcome = yes | parents)
        Second row of table contains P(outcome = no | parents)
        """

        input = {parent:[] for parent in self.parents}

        for f in [2, 1, 0]: # possible values for fever:
            combinations = itertools.product([1, 0], repeat=len(self.parents)-1) # all parents except for fever
            for comb in combinations: 
                for i, parent in enumerate(self.parents): 
                    if parent == "fever": 
                        input[parent].append(f)
                    else: 
                        input[parent].append(comb[i])

        input[self.outcome] = torch.ones(len(input["fever"]))
        input = {parent:torch.tensor(val) for parent, val in input.items()}

        p_pos = self.forward(input).exp() # prob(outcome | parents) for all combinations of parents
        p_neg = 1-p_pos

        cpt = torch.stack((p_pos, p_neg))

        return cpt.detach().clone().numpy()
    

class DaysAtHome(torch.nn.Module):
    """
    Conditional probability distribution for days_at_home variable, parameterized using a Poisson regression model
    See also DaysAtHome class in data_generating_process.py
    - outcome: name of variable we are modeling (days_at_home)
    - parents: list of parent variables 
    - incl_self_empl: whether to model self-employed variable (if True, then coeff[0] contains its weight)
    - bias_a0: bias for antibiotics = 0 regression model (learned from data if None)
    - coeff_a0: coefficients for antibiotics = 0 regression model (learned from data if None)
    - bias_a1: bias for antibiotics = 1 regression model (learned from data if None)
    - coeff_a1: coefficients for antibiotics = 1 regression model (learned from data if None)
    """
    
    def __init__(self, outcome, parents, incl_self_empl=False, bias_a0=None, coeff_a0=None, bias_a1=None, coeff_a1=None): 

        super(DaysAtHome, self).__init__()
        self.outcome = outcome # name of variable we are modeling
        self.parents = parents # parent variables
        self.n = len(parents) # number of parents
        self.incl_self_empl = incl_self_empl # whether self-employed is included in the model

        # learnable parameters antibiotics=0 model
        self.bias_a0 = torch.nn.Parameter(torch.rand(1)) if bias_a0 == None else torch.nn.Parameter(torch.tensor(bias_a0)) # bias
        self.coeff_a0 = torch.nn.Parameter(torch.rand(self.n)) if coeff_a0 == None else torch.nn.Parameter(torch.tensor(coeff_a0)) # coefficients (2 for fever but one less for antibiotics)

        # learnable parameters antibiotics=1 model
        self.bias_a1 = torch.nn.Parameter(torch.rand(1)) if bias_a1 == None else torch.nn.Parameter(torch.tensor(bias_a1)) # bias
        self.coeff_a1 = torch.nn.Parameter(torch.rand(self.n)) if coeff_a1 == None else torch.nn.Parameter(torch.tensor(coeff_a1)) # coefficients (2 for fever but one less for antibiotics)
        
    def forward(self, sample): 
        """
        calculate the log-probability that patient stays at home for a specific number of days (depending on its observed value), given the values of the parent variables
        - sample: datapoint (dict) containing tabular values as tensors
        """

        low_fever = torch.where(sample["fever"] == 1, torch.ones_like(sample["fever"]), torch.zeros_like(sample["fever"]))
        high_fever = torch.where(sample["fever"] == 2, torch.ones_like(sample["fever"]), torch.zeros_like(sample["fever"]))

        y = sample[self.outcome] # select the outcome (days at home)
        
        if self.incl_self_empl: 
            logit_a0 = self.bias_a0 + self.coeff_a0[0]*sample["self_empl"] \
                + self.coeff_a0[1]*sample["dysp"] + self.coeff_a0[2]*sample["cough"] \
                + self.coeff_a0[3]*sample["pain"] + self.coeff_a0[4]*sample["nasal"] \
                + self.coeff_a0[5]*low_fever + self.coeff_a0[6]*high_fever
            logit_a1 = self.bias_a1 + self.coeff_a1[0]*sample["self_empl"] \
                    + self.coeff_a1[1]*sample["dysp"] + self.coeff_a1[2]*sample["cough"] \
                    + self.coeff_a1[3]*sample["pain"] + self.coeff_a1[4]*sample["nasal"] \
                    + self.coeff_a1[5]*low_fever + self.coeff_a1[6]*high_fever
        else:
            logit_a0 = self.bias_a0 \
                    + self.coeff_a0[0]*sample["dysp"] + self.coeff_a0[1]*sample["cough"] \
                    + self.coeff_a0[2]*sample["pain"] + self.coeff_a0[3]*sample["nasal"] \
                    + self.coeff_a0[4]*low_fever + self.coeff_a0[5]*high_fever
            logit_a1 = self.bias_a1 \
                    + self.coeff_a1[0]*sample["dysp"] + self.coeff_a1[1]*sample["cough"] \
                    + self.coeff_a1[2]*sample["pain"] + self.coeff_a1[3]*sample["nasal"] \
                    + self.coeff_a1[4]*low_fever + self.coeff_a1[5]*high_fever
        log_lambda = torch.where(sample["antibiotics"] == 1, logit_a1, logit_a0) # log(lambda), where labmda is mean of poisson distr

        log_p = y*log_lambda-log_lambda.exp()-torch.lgamma(y+1) # log of Poisson probability (k*log(lambda)-lambda-log(k!)), where k! = lgamma(k+1)
        
        return log_p
    
    def train(self, df, bs=50, lr=0.01, num_epochs=15):
        """
        Estimate parameters of the Antibiotics distribution (bias and coeff) from dataset 
        - df: dataset containing all tabular variables (dict with variable name as key, and tensor as value)
        - bs: batch size
        - lr: learning rate
        - num_epochs: number of epochs
        """

        df_subset = df[[self.outcome]+self.parents]
        train_data = CustomDataset(df_subset)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

        optimizer = Adam(self.parameters(), lr=lr) 

        train_loss = []
        for _ in range(num_epochs):
            epoch_loss = 0
            n_batch = 0
            for batch in train_loader:

                # Forward pass
                log_prob = self.forward(batch)
                loss = -log_prob.sum() # negative log-likelihood
                epoch_loss += loss
                n_batch += 1

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss.append(epoch_loss.detach().item()/n_batch)

        return train_loss
    
    def get_CPT(self):
        """
        Generate full conditional probability table from the Poisson model parameters. 
        Contains P(outcome | parents) for all combinations of parents, where parents are assumed to have two possible values (yes/no), except for fever
        First row of table contains P(days_at_home = 0 | parents)
        Second row of table contains P(days_at_home = 1 | parents)
        etc. 
        Last row of table contains P(days_at_home >= 15 | parents) (15 is the maximum number of days observed in the training data)
        """

        input = {parent:[] for parent in self.parents}

        for f in [2, 1, 0]: # possible values for fever:
            combinations = itertools.product([1, 0], repeat=len(self.parents)-1) # all parents except for fever
            for comb in combinations: 
                for i, parent in enumerate(self.parents): 
                    if parent == "fever": 
                        input[parent].append(f)
                    else: 
                        input[parent].append(comb[i])

        cpt = torch.empty((0, len(input["fever"])), dtype=torch.float32)
        for days in range(15): # days range from 0 to 14
            input[self.outcome] = torch.ones(len(input["fever"]))*days
            input = {parent:torch.tensor(val) for parent, val in input.items()}
            p_pos = self.forward(input).exp() # prob(outcome | parents) for all combinations of parents
            cpt = torch.cat((cpt, p_pos.unsqueeze(dim=0)), dim=0)

        p_rest = 1 - torch.sum(cpt, dim=0) # P(k > 14) = 1-P(k<=14) = 1-P(k=0)-P(k=1)-...-P(k=13)-P(k=14)

        cpt = torch.cat((cpt, p_rest.unsqueeze(dim=0)), dim=0)

        return cpt.detach().clone().numpy()

class SynsumBayesianNetwork():
    """
    Parent class for TrainedBayesianNetwork and GroundTruthBayesianNetwork
    """
    def __init__(self):
        self.BN_model = None # placeholder

    @staticmethod
    def days_at_home_categories(val): 
        """
        Turn the days at home values into string categories. 
        """
        if val >= 15:
            return ">=15"
        else: 
            return str(val)

    def symptoms_virtual_evidence(self, values):
        virtual_evidence = []
        for symptom in ["dysp", "cough", "pain", "nasal", "fever"]:
            label = f"text_prob_{symptom}"
            probability = values[label]
            
            variable_card = 2 if symptom != 'fever' else 3

            if symptom != 'fever':
                cpt = [[probability], [1-probability]]
            else:
                cpt = probability.reshape(3, 1)[::-1] # reverse from none,low,high to high,low,none

            names = {symptom:['yes', 'no']} if symptom != 'fever' else {symptom:['high', 'low', 'none']}
            
            cpd = TabularCPD(variable=symptom, variable_card=variable_card, values=cpt,
                            state_names=names)
            virtual_evidence.append(cpd)
        return virtual_evidence

    def predict_symptom(self, values, symptom, evidence_keys=["asthma", "smoking", "COPD", "season", "hay_fever", "pneu", "common_cold", "antibiotics", "days_at_home"], virtual_evidence=False): 
        """
        Use variable elimination in the Bayesian network to predict P(symptom | ev).
        - values: row in Dataframe containing values for all variables in the BN. 
                  used to set evidence values: ev = {evidence_keys: values} 
        - symptom: if fever, return array [P(fever = none | ev), P(fever = low | ev), P(fever = high | ev)]. 
                   else, return P(symptom = yes | ev).
        - evidence_keys: variables to include as evidence, for which values can be found in "values" argument
        """

        self.inf_obj = VariableElimination(self.BN_model)

        ev = {key:values[key] for key in evidence_keys}

        if "days_at_home" in evidence_keys: 
            ev["days_at_home"] = SynsumBayesianNetwork.days_at_home_categories(ev["days_at_home"])

        v_ev = self.symptoms_virtual_evidence(values) if virtual_evidence else None

        res = self.inf_obj.query([symptom], evidence=ev, virtual_evidence=v_ev)
        if symptom == "fever":
            prob = np.round(np.array([res.get_value(**{symptom:"none"}), res.get_value(**{symptom:"low"}), res.get_value(**{symptom:"high"})]), 8)
        else: 
            prob = round(res.get_value(**{symptom:"yes"}), 8)

        return prob    

class TrainedBayesianNetwork(SynsumBayesianNetwork): 
    """ 
    Fully learn the Bayesian network from data.
    - df_train: dataframe containing training data
    - seed: seed used for initialization of all parameters
    """

    def __init__(self, df_train, seed):
        super(SynsumBayesianNetwork, self).__init__()

        self.CPTs = {}
        self.df_train = df_train

        # set seeds
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def learn_CPTs(self): 
        """
        Directly learn conditional probability table (CPTs) for variables asthma, smoking, COPD, season, 
        hay fever, pneumonia, common cold and fever. 
        """

        state_names = {"asthma": ["yes", "no"], "smoking": ["yes", "no"],
               "COPD": ["yes", "no"], "season": ["winter", "summer"], "hay_fever": ["yes", "no"], 
               "pneu": ["yes", "no"], "common_cold": ["yes", "no"], "fever": ["high", "low", "none"]}
        self.BN_model = BayesianNetwork([("asthma", "pneu"), ("smoking", "COPD"), ("COPD", "pneu"),
                                ("season", "pneu"), ("season", "common_cold"), 
                                ("pneu", "fever"), ("common_cold", "fever")])
        self.BN_model.add_node("hay_fever")

        df_subset = self.df_train[["asthma", "pneu", "smoking", "COPD", "season", "common_cold", "fever", "hay_fever"]]

        self.BN_model.fit(df_subset, estimator=BayesianEstimator, prior_type="K2", state_names=state_names)

    def learn_noisy_ORs(self): 
        """ 
        Learn noisy-OR distributions for symptoms dysp, cough, pain and nasal.
        """

        parents = {"dysp": ["asthma", "smoking", "COPD", "hay_fever", "pneu"], 
           "cough": ["asthma", "smoking", "COPD", "pneu", "common_cold"], 
           "pain": ["COPD", "cough", "pneu", "common_cold"], 
           "nasal": ["hay_fever", "common_cold"]}

        CPTs = {}
        self.noisy_OR_models = {}
        for symptom in ["dysp", "cough", "pain", "nasal"]: 
            model = NoisyOr(symptom, parents[symptom])
            if len(self.df_train) > 500:
                epochs = 10
            elif len(self.df_train) >= 100: 
                epochs = 50
            else: 
                epochs = 100
            model.train(self.df_train, bs=50, lr=0.1, num_epochs=epochs)
            self.noisy_OR_models[symptom] = model
            CPTs[symptom] = model.get_CPT()

        # add learned CPTs to BN

        self.BN_model.add_edges_from([("asthma", "dysp"), ("smoking", "dysp"), ("COPD", "dysp"), ("pneu", "dysp"), ("hay_fever", "dysp"),
                      ("asthma", "cough"), ("smoking", "cough"), ("COPD", "cough"), ("pneu", "cough"), ("common_cold", "cough"),
                      ("cough", "pain"), ("pneu", "pain"), ("COPD", "pain"), ("common_cold", "pain"), 
                      ("common_cold", "nasal"), ("hay_fever", "nasal")])

        cpd_dysp = TabularCPD(variable="dysp", variable_card=2, values = CPTs["dysp"],
                    evidence=["asthma", "smoking", "COPD", "hay_fever", "pneu"], evidence_card=[2, 2, 2, 2, 2],
                    state_names={"dysp":["yes", "no"], "asthma":["yes", "no"], "smoking":["yes", "no"], "COPD":["yes", "no"], "pneu":["yes", "no"], "hay_fever":["yes", "no"]})

        cpd_cough = TabularCPD(variable="cough", variable_card=2, values = CPTs["cough"],
                            evidence=["asthma", "smoking", "COPD", "pneu", "common_cold"], evidence_card=[2, 2, 2, 2, 2],
                            state_names={"cough":["yes", "no"], "asthma":["yes", "no"], "smoking":["yes", "no"], "COPD":["yes", "no"], "pneu":["yes", "no"], "common_cold":["yes", "no"]})

        cpd_pain = TabularCPD(variable="pain", variable_card=2, values = CPTs["pain"],
                            evidence=["COPD", "cough", "pneu", "common_cold"], evidence_card=[2, 2, 2, 2],
                            state_names={"pain":["yes", "no"], "COPD":["yes", "no"], "cough":["yes", "no"], "pneu":["yes", "no"], "common_cold":["yes", "no"]})

        cpd_nasal = TabularCPD(variable="nasal", variable_card=2, values = CPTs["nasal"],
                            evidence=["hay_fever", "common_cold"], evidence_card=[2, 2],
                            state_names={"nasal":["yes", "no"], "hay_fever":["yes", "no"], "common_cold":["yes", "no"]})
        
        self.BN_model.add_cpds(cpd_dysp, cpd_cough, cpd_pain, cpd_nasal)

    def learn_antibiotics(self): 
        """ 
        Learn logistic regression model for Antibiotics.
        """

        self.antibio_model = Antibiotics("antibiotics", ["dysp", "cough", "pain", "fever"], incl_policy=False)
        if len(self.df_train) > 4000:
            self.antibio_model.train(self.df_train, bs=50, lr=0.01, num_epochs=30)
        elif len(self.df_train) > 1000:
            self.antibio_model.train(self.df_train, bs=50, lr=0.01, num_epochs=100)
        elif len(self.df_train) >= 100:
            self.antibio_model.train(self.df_train, bs=50, lr=0.05, num_epochs=100)
        else:
            self.antibio_model.train(self.df_train, bs=50, lr=0.05, num_epochs=200)
        CPT = self.antibio_model.get_CPT()

        self.BN_model.add_edges_from([("dysp", "antibiotics"), ("cough", "antibiotics"), ("pain", "antibiotics"), ("fever", "antibiotics")])

        cpd_antibiotics = TabularCPD(variable="antibiotics", variable_card=2, values = CPT,
                    evidence=["fever", "dysp", "cough", "pain"], evidence_card=[3, 2, 2, 2],
                    state_names={"antibiotics": ["yes", "no"], "fever":["high", "low", "none"], "dysp":["yes", "no"], "cough":["yes", "no"], "pain":["yes", "no"]})

        self.BN_model.add_cpds(cpd_antibiotics)

    def learn_days_at_home(self): 
        """ 
        Learn Poisson regression model for DaysAtHome.
        """

        self.days_model = DaysAtHome("days_at_home", ["antibiotics", "dysp", "cough", "pain", "nasal", "fever"], incl_self_empl=False)
        if len(self.df_train) > 4000:
            self.days_model.train(self.df_train, bs=50, lr=0.01, num_epochs=30)
        elif len(self.df_train) > 300:
            self.days_model.train(self.df_train, bs=50, lr=0.01, num_epochs=100)
        elif len(self.df_train) >= 100:
            self.days_model.train(self.df_train, bs=50, lr=0.05, num_epochs=100)
        else: 
            self.days_model.train(self.df_train, bs=50, lr=0.05, num_epochs=200)
        CPT = self.days_model.get_CPT()

        self.BN_model.add_edges_from([("antibiotics", "days_at_home"), ("dysp", "days_at_home"), ("cough", "days_at_home"), ("pain", "days_at_home"), ("fever", "days_at_home"), ("nasal", "days_at_home")])

        cpd_days_at_home = TabularCPD(variable="days_at_home", variable_card=16, values = CPT,
                    evidence=["fever", "antibiotics", "dysp", "cough", "pain", "nasal"], evidence_card=[3, 2, 2, 2, 2, 2],
                    state_names={"days_at_home": list([str(e) for e in range(15)])+[">=15"], "fever":["high", "low", "none"], "antibiotics":["yes", "no"], "dysp":["yes", "no"], "cough":["yes", "no"], "pain":["yes", "no"], "nasal":["yes", "no"]})
        
        self.BN_model.add_cpds(cpd_days_at_home)

    def learn_full_BN(self, print_msg=False): 
        """
        Learn the full Bayesian network.
        """
        if print_msg:
            print("learning CPTs...")
        self.learn_CPTs()
        if print_msg:
            print("learning Noisy ORs...")
        self.learn_noisy_ORs()
        if print_msg:
            print("learning Antibiotics...")
        self.learn_antibiotics()
        if print_msg:
            print("learning Days at home...")
        self.learn_days_at_home()
    

class GTBayesianNetwork(SynsumBayesianNetwork):
    """
    Manually instantiate ground-truth Bayesian network as defined by an expert in SynSUM. 
    Do not learn any parameters from the data, but assume you know the exact data generating process. 
    Models "policy" and "self-empl", as these are also defined in SynSUM, though these are never included as evidence during inference. 
    """
        
    def __init__(self):
        super(SynsumBayesianNetwork, self).__init__()

        cpd_dysp = TabularCPD(variable="dysp", variable_card=2, 
                          values = NoisyOr("dysp", ["asthma", "smoking", "COPD", "hay_fever", "pneu"], 0.05, [0.9, 0.3, 0.9, 0.2, 0.3]).get_CPT(),
                    evidence=["asthma", "smoking", "COPD", "hay_fever", "pneu"], evidence_card=[2, 2, 2, 2, 2],
                    state_names={"dysp":["yes", "no"], "asthma":["yes", "no"], "smoking":["yes", "no"], "COPD":["yes", "no"], "pneu":["yes", "no"], "hay_fever":["yes", "no"]})

        cpd_cough = TabularCPD(variable="cough", variable_card=2, 
                           values = NoisyOr("cough", ["asthma", "smoking", "COPD", "pneu", "common_cold"], 0.07, [0.3, 0.6, 0.4, 0.85, 0.7]).get_CPT(),
                        evidence=["asthma", "smoking", "COPD", "pneu", "common_cold"], evidence_card=[2, 2, 2, 2, 2],
                        state_names={"cough":["yes", "no"], "asthma":["yes", "no"], "smoking":["yes", "no"], "COPD":["yes", "no"], "pneu":["yes", "no"], "common_cold":["yes", "no"]})

        cpd_pain = TabularCPD(variable="pain", variable_card=2, 
                          values = NoisyOr("pain", ["COPD", "cough", "pneu", "common_cold"], 0.05, [0.15, 0.2, 0.3, 0.1]).get_CPT(),
                        evidence=["COPD", "cough", "pneu", "common_cold"], evidence_card=[2, 2, 2, 2],
                        state_names={"pain":["yes", "no"], "COPD":["yes", "no"], "cough":["yes", "no"], "pneu":["yes", "no"], "common_cold":["yes", "no"]})

        cpd_nasal = TabularCPD(variable="nasal", variable_card=2, 
                           values = NoisyOr("nasal", ["hay_fever", "common_cold"], 0.1, [0.85, 0.7]).get_CPT(),
                        evidence=["hay_fever", "common_cold"], evidence_card=[2, 2],
                        state_names={"nasal":["yes", "no"], "hay_fever":["yes", "no"], "common_cold":["yes", "no"]})
    
        cpd_fever = TabularCPD(variable = "fever", variable_card=3,
                                values = [[0.80, 0.80, 0.05, 0.05], #(pneu=yes,inf=yes), (pneu=yes,inf=no), (pneu=no, inf=yes), (pneu=no, inf=no)
                                  [0.15, 0.10, 0.20, 0.15],
                                  [0.05, 0.10, 0.75, 0.80]],
                            evidence = ["pneu", "common_cold"], evidence_card = [2, 2], 
                            state_names = {"fever": ["high", "low", "none"], "pneu": ["yes", "no"], "common_cold": ["yes", "no"]})
    
        cpd_pneu = TabularCPD(variable = "pneu", variable_card=2,
                            values = [[0.04, 0.013, 0.04, 0.013, 0.02, 0.0065, 0.015, 0.005], #(COPD=1, asthma=1, winter=1), (COPD=1, asthma=1, winter=0), (COPD=1, asthma=0, winter=1), (COPD=1, asthma=0, winter=0), (COPD=0, asthma=1, winter=1), (COPD=0, asthma=1, winter=0), (COPD=0, asthma=0, winter=1), (COPD=0, asthma=0, winter=0)
                                      [0.96, 0.987, 0.96, 0.987, 0.98, 0.9935, 0.985, 0.995]],
                            evidence = ["COPD", "asthma", "season"], evidence_card = [2, 2, 2], 
                            state_names = {"pneu": ["yes", "no"], "COPD": ["yes", "no"], "asthma": ["yes", "no"], "season": ["winter", "summer"]})
    
        cpd_inf = TabularCPD(variable = "common_cold", variable_card=2,
                                values = [[0.5, 0.05], #(winter=yes), (winter=no)
                                          [0.5, 0.95]],
                            evidence = ["season"], evidence_card = [2], 
                            state_names = {"common_cold": ["yes", "no"], "season": ["winter", "summer"]})
    
        cpd_COPD = TabularCPD(variable = "COPD", variable_card=2,
                            values = [[0.073, 0.0075], #(smoking=yes), (smoking=no)
                                      [0.927, 0.9925]],
                            evidence = ["smoking"], evidence_card = [2], 
                            state_names = {"COPD": ["yes", "no"], "smoking": ["yes", "no"]})
    
        cpd_policy = TabularCPD(variable = "policy", variable_card=2,
                            values = [[0.65], 
                                      [0.35]],
                            evidence = [], evidence_card = [], 
                            state_names = {"policy": ["yes", "no"]})
    
        cpd_self_empl = TabularCPD(variable = "self_empl", variable_card=2,
                            values = [[0.11], 
                                      [0.89]],
                            evidence = [], evidence_card = [], 
                            state_names = {"self_empl": ["yes", "no"]})

        cpd_asthma = TabularCPD(variable = "asthma", variable_card=2,
                            values = [[0.095], 
                                      [0.905]],
                            evidence = [], evidence_card = [], 
                            state_names = {"asthma": ["yes", "no"]})
        
        cpd_smoking = TabularCPD(variable = "smoking", variable_card=2,
                            values = [[0.19], 
                                      [0.81]],
                            evidence = [], evidence_card = [], 
                            state_names = {"smoking": ["yes", "no"]})
        
        cpd_hay_fever = TabularCPD(variable = "hay_fever", variable_card=2,
                            values = [[0.015], 
                                      [0.985]],
                            evidence = [], evidence_card = [], 
                            state_names = {"hay_fever": ["yes", "no"]})
        
        cpd_winter = TabularCPD(variable = "season", variable_card=2,
                            values = [[0.4], 
                                      [0.6]],
                            evidence = [], evidence_card = [], 
                            state_names = {"season": ["winter", "summer"]})
        
        cpd_antibiotics = TabularCPD(variable="antibiotics", variable_card=2, 
                        values = Antibiotics("antibiotics", ["policy", "dysp", "cough", "pain", "fever"], bias=-3., coeff=[2/2, 1.6/2, 1.33/2, 1.33/2, 1.8/2, 4.5/2], incl_policy=True).get_CPT(),
                    evidence=["fever", "policy", "dysp", "cough", "pain"], evidence_card=[3, 2, 2, 2, 2],
                    state_names={"antibiotics": ["yes", "no"], "fever":["high", "low", "none"], "policy":["yes", "no"], "dysp":["yes", "no"], "cough":["yes", "no"], "pain":["yes", "no"]})

        
        cpd_days_at_home = TabularCPD(variable="days_at_home", variable_card=16, 
                        values = DaysAtHome("days_at_home", ["antibiotics", "self_empl", "dysp", "cough", "pain", "nasal", "fever"], incl_self_empl=True,
                                            bias_a0 = 0.010, coeff_a0 = [-0.5, 0.64, 0.35, 0.47, 0.011, 0.81, 1.23],
                                            bias_a1 = 0.16, coeff_a1 = [-0.5, 0.51, 0.42, 0.26, 0.0051, 0.24, 0.57]).get_CPT(),
                    evidence=["fever", "antibiotics", "self_empl", "dysp", "cough", "pain", "nasal"], evidence_card=[3, 2, 2, 2, 2, 2, 2],
                    state_names={"days_at_home": list([str(e) for e in range(15)])+[">=15"], "fever":["high", "low", "none"], "antibiotics":["yes", "no"], "self_empl":["yes", "no"], "dysp":["yes", "no"], "cough":["yes", "no"], "pain":["yes", "no"], "nasal":["yes", "no"]})

        BN_model = BayesianNetwork([("asthma", "pneu"), ("smoking", "COPD"), ("COPD", "pneu"),
                                ("season", "pneu"), ("season", "common_cold"), 
                                ("pneu", "fever"), ("common_cold", "fever"), 
                                ("asthma", "dysp"), ("smoking", "dysp"), ("COPD", "dysp"), ("pneu", "dysp"), ("hay_fever", "dysp"),
                                ("asthma", "cough"), ("smoking", "cough"), ("COPD", "cough"), ("pneu", "cough"), ("common_cold", "cough"),
                                ("cough", "pain"), ("pneu", "pain"), ("COPD", "pain"), ("common_cold", "pain"), 
                                ("common_cold", "nasal"), ("hay_fever", "nasal"), 
                                ("policy", "antibiotics"), ("dysp", "antibiotics"), ("cough", "antibiotics"), ("pain", "antibiotics"), ("fever", "antibiotics"),
                                ("antibiotics", "days_at_home"), ("dysp", "days_at_home"), ("cough", "days_at_home"), ("pain", "days_at_home"), ("fever", "days_at_home"), ("nasal", "days_at_home"), ("self_empl", "days_at_home")])
        
        BN_model.add_cpds(cpd_dysp, cpd_cough, cpd_pain, cpd_fever, cpd_nasal, 
                          cpd_pneu, cpd_inf, cpd_COPD, cpd_policy, cpd_self_empl, cpd_asthma, cpd_smoking, cpd_hay_fever, cpd_winter,
                          cpd_antibiotics, cpd_days_at_home)
        
        self.BN_model = BN_model