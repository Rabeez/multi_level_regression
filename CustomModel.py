import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin


def funnel_viz(steps, conversion_rates, labels=None):
    ASPECT_MULTIPLE = 1
    MAX_RADIUS = 10
    SPACING = MAX_RADIUS * 4
    FIG_HEIGHT = 4
    FIG_WIDTH = 4 * steps * ASPECT_MULTIPLE
    MAX_Y = 50
    MAX_X = MAX_Y * steps * ASPECT_MULTIPLE
    
    if labels is None:
        labels = [f"level_{i}" for i in range(1, steps+1)]
    else:
        assert len(labels) == steps
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # First circle
    c = plt.Circle((MAX_RADIUS, MAX_Y/2), MAX_RADIUS)
    ax.add_patch(c)
    
    # Due to assertion in data creation this loop should run atleast once
    for i, (rate, label) in enumerate(zip(conversion_rates, labels), 1):
        # radius scaling is done on first circle's radius which is incorrect
        # it should be on the previous circle's radius instead
        # but is ok for demonstrative visualization purpose
        c = plt.Circle((MAX_RADIUS+(SPACING*i), MAX_Y/2), MAX_RADIUS * rate, color=f'C{i}')
        ax.add_patch(c)
        
        plt.text(MAX_RADIUS+(SPACING*(i-0.5)), MAX_Y/2, f'{rate:.1%}')
        plt.text(MAX_RADIUS+(SPACING*i), MAX_Y/2, label)
    
    ax.set(xlim=(0,MAX_X), ylim=(0,MAX_Y), aspect='equal')
    ax.grid(False)
    ax.axis(False)
    
    return ax


class MultiLevelLogisticRegression(BaseEstimator, ClassifierMixin):
    '''This classifier builds a logistic regression for each level in the funnel and fits each layer on only the relevant eligible population.'''
    
    def __init__(self, steps, clf_cls, clf_kws=None):
        self.steps = steps
        self.clf_cls = clf_cls
        if clf_kws is None: 
            self.clf_kws = {}
        else:
            self.clf_kws = clf_kws
        self.models = [self.clf_cls(**self.clf_kws) for _ in range(steps)]
        super().__init__()

    def fit(self, X, Y=None):
        '''Fits each layer's model on only the datapoints which have 1 in previous layer's target. The first layer is exempt from this.'''
        
        assert Y.shape[1] == self.steps
        assert (X.index == Y.index).all()
        
        for i, model in enumerate(self.models, 1):
            if i == 1:
                # for first conversion step everyone is in the eligible population
                model_x = X
                model_y = Y[f"level_{i}"]
            else:
                # for all next conversion steps the eligible population is only those who had a 1 in the previous step
                model_x = X.loc[Y[f"level_{i-1}"] == 1, :]
                model_y = Y.loc[Y[f"level_{i-1}"] == 1, f"level_{i}"]
            
            print(f"Fitting level_{i} model with {len(model_y):,} datapoints and {model_y.mean():.1%} class mean.")
            
            model.fit(model_x, model_y)
        
        return self

    def predict(self, X):
        '''For simplicity all models are used for prediction of a datapoint and it is left upto the user to use/discard predictions for a datapoint in later layers if a layer has 0.'''
        preds_df = pd.DataFrame()
        for i, model in zip(range(1, self.steps+1), self.models):
            preds_df[f"pred_level_{i}"] = model.predict(X)

        assert preds_df.shape == (len(X), self.steps)
        return preds_df
    
    def predict_proba(self, X):
        '''For simplicity all models are used for prediction of a datapoint and it is left upto the user to use/discard predictions for a datapoint in later layers if a layer has low probability.'''
        
        preds_df = pd.DataFrame()
        for i, model in zip(range(1, self.steps+1), self.models):
            preds_df[f"pred_prob_level_{i}"] = model.predict_proba(X)[:,1]
                
        assert preds_df.shape == (len(X), self.steps)
        return preds_df
    
    def feature_importance(self, feature_names=None):
        if feature_names is not None:
            assert len(feature_names) == len(self.models[0].coef_[0])
            imps = pd.DataFrame(index=feature_names)
        else:
            imps = pd.DataFrame()
        
        for i, model in enumerate(self.models, 1):
            imps[f"level_{i}"] = model.coef_[0]
        
        return imps