import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import *
import scipy.optimize as optimize

from src.models.si import SI
from src.models.isi import ISI
from src.utils.common import save_json
from src.utils.plotters import multiplot, simple_plot
from src.entity.config_entity import OptimizatorConfig



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)



class Optimizator:


    def __init__(self, 
                 config: OptimizatorConfig, 
                 constraints: dict,
                 plot_default_params: dict):
        """
        Initializes the Optimizator component.
        
        Args:
        - config (OptimizatorConfig): Configuration settings.
        - constraints (list): Dictionary with constraints. 
        - plot_default_params (dict): Dictionary with default parameters for plots
        """
        self.config = config
        self.constraints = constraints
        self.plot_default_params = plot_default_params
        self.metric = str_to_class(config.metric_name)


    def read_data(self):

        df = pd.read_csv(self.config.data.path_to_data) \
               .dropna(subset=self.config.data.dropna_subset)
        df = df.loc[[item in self.config.data.locations for item in df['location']] , self.config.data.columns]
        df['date'] = df['date'].astype('datetime64[ns]')
        if self.config.data.max_date != None:
            df = df[df['date'] <= str(self.config.data.max_date)]

        return df


    def save_results(self, res, location):

        params_names = list(self.constraints.keys())
        
        best_train_score = min(res['train_score'])
        val_score_for_best_train_score = res['val_score'][res['train_score'].index(best_train_score)]
        best_train_params = res['x'][res['train_score'].index(best_train_score)]

        best_val_score = min(res['val_score'])
        train_score_for_best_val_score = res['train_score'][res['val_score'].index(best_val_score)]
        best_val_params = res['x'][res['val_score'].index(best_val_score)]

        best_results = {'train_based': {f'{self.config.metric_name}_TRAIN': best_train_score,
                                        f'{self.config.metric_name}_VAL': val_score_for_best_train_score,
                                        'params': dict(zip(params_names, best_train_params))},
                        'val_based': {f'{self.config.metric_name}_TRAIN': train_score_for_best_val_score,
                                      f'{self.config.metric_name}_VAL': best_val_score,
                                      'params': dict(zip(params_names, best_val_params))}}
        
        print(json.dumps(best_results, sort_keys=False, indent=4))

        save_json(Path(os.path.join(self.config.path_to_results, f'{self.config.model_name}/{location}.json')), best_results)
        

    def F(self, params, n_steps, init_condition, pop_size, Xtrue):
        
        model = str_to_class(self.config.model_name)(*params, N=pop_size)
        Xpred = model.run(**init_condition, n_steps=n_steps, dt=self.config.dt)[1]
        try:
            error = self.metric(Xtrue, Xpred)
        except Exception as e:
            print(params)
            print(max(np.abs(Xpred)))
            print(max(np.abs(Xtrue)))
            raise e
        
        return error


    def val(self, params, init_condition, pop_size, Xtrue):

        model = str_to_class(self.config.model_name)(*params, N=pop_size)
        Xpred = model.run(**init_condition, n_steps=len(Xtrue), dt=self.config.dt)[1]
        # Xpred_val = Xpred[int(len(Xtrue)*(1 - self.config.test_ratio)):]
        # Xtrue_val = Xtrue[int(len(Xtrue)*(1 - self.config.test_ratio)):]
        # error = self.metric(Xtrue_val, Xpred_val) 
        error = self.metric(Xtrue, Xpred) 

        return error


    def optimize(self, init_condition, pop_size, Xtrue):

        init_params = [np.random.RandomState(self.config.seed).uniform(*bound, self.config.n_runs) for bound in self.constraints.values()]

        res = {'train_score': [], 'val_score': [], 'x': []}
        
        for run in tqdm(range(self.config.n_runs)):
            
            n_steps = int(len(Xtrue)*(1 - self.config.test_ratio))
                          
            args = (n_steps, init_condition, pop_size, Xtrue[:n_steps])
                
            result = optimize.minimize(self.F, 
                                       [v[run] for v in init_params], 
                                       method=self.config.optimizator_name, 
                                       args=args, 
                                       bounds=self.constraints.values())

            val_score = self.val(result['x'], init_condition, pop_size, Xtrue)
                  
            res['train_score'].append(result['fun'])
            res['val_score'].append(val_score)
            res['x'].append(result['x'])
                  
        return res


    def show_graphs(self, params_train, params_val, df):

        x0 = df['total_cases'].iloc[0]
        pop_size = df['population'].iloc[0]
        init_condition = {'s0': pop_size, 'x0': x0}
        
        n_steps = len(df)
        model1 = str_to_class(self.config.model_name)(*params_val, N=pop_size)
        model2 = str_to_class(self.config.model_name)(*params_train, N=pop_size)
        
        if self.config.model_name == 'SI':
            _, X1 = model1.run(**init_condition, n_steps=n_steps, dt=self.config.dt)
            _, X2 = model2.run(**init_condition, n_steps=n_steps, dt=self.config.dt)
            multiplot([df['date'], df['date']], [X1, df['total_cases']], labels=['pred (best from val score)', 'true'], 
                      ylabel='total cases', params_dict=self.plot_default_params)
            multiplot([df['date'], df['date']], [X2, df['total_cases']], labels=['pred (best from train score)', 'true'], 
                      ylabel='total cases', params_dict=self.plot_default_params)
        else:
            _, X1, Rr1 = model1.run(**init_condition, n_steps=n_steps, dt=self.config.dt)
            #_, X2, Rr2 = model2.run(**init_condition, n_steps=n_steps, dt=self.config.dt)
            multiplot([df['date'], df['date']], [X1, df['total_cases']], labels=['pred', 'true'], 
                      ylabel='total cases', params_dict=self.plot_default_params)  
            # multiplot([df['date'], df['date']], [X2, df['total_cases']], labels=['pred (best from train score)', 'true'], 
            #           ylabel='total cases', params_dict=self.plot_default_params)  
            # multiplot([df['date'], df['date']], [Rr1, Rr2], labels=['best from val score', 'best from train score'], 
            #           ylabel='transmission rate', params_dict=self.plot_default_params)
            simple_plot(df['date'], Rr1, ylabel='transmission rate', params_dict=self.plot_default_params)


    def run_stage(self, df, location):

        df_subset = df[df['location'] == location]
        x0 = df_subset['total_cases'].iloc[0]
        pop_size = df_subset['population'].iloc[0]
        Xtrue = df_subset['total_cases']
        init_condition = {'s0': pop_size, 'x0': x0}

        res = self.optimize(init_condition, pop_size, Xtrue)

        self.save_results(res, location)

        best_train_params = res['x'][res['train_score'].index(min(res['train_score']))]
        best_val_params = res['x'][res['val_score'].index(min(res['val_score']))]
        self.show_graphs(best_train_params, best_val_params, df_subset) 