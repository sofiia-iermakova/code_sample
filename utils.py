from typing import Iterable, Tuple
import math

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

class EDAViz():   
    def __init__(self, 
                 df: pd.DataFrame, 
                 target_field: str,                 
                 num_fields: Iterable = [],
                 cat_fields: Iterable = [],
                 date_fields: Iterable = [],
                 field_aliases: dict = {},
                 palette: str = 'mako',
                 color: str = 'red',
                 y_label_right: str = 'Probability', 
                 y_label_left: str = 'Count', 
                 errorbar  = ('ci', 95)               
                 ) -> None:
        """Convenience class for quick visualization of multiple features vs binary target.
        All fields that aren't assigned specific type are treated as categorical.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with features and target
        target_field : str
            Name of the field with target variable 
        num_fields : Iterable, optional
            List of numeric fields, by default []
        cat_fields : Iterable, optional
            List of categorical fields, by default []
        date_fields : Iterable, optional
            List of date fields, by default []
        field_aliases : dict, optional
            Human friendly field aliases, by default {}
        palette : str, optional
            Palette to use for countplot , by default 'mako'
        color : str, optional
            Color to use for piontplot, by default 'red'
        y_label_right : str, optional
            Label for the right y axis, by default 'Probability'
        y_label_left : str, optional
            Label for the left y axis, by default 'Count'
        errorbar : tuple, optional
            Errorbar parameters for the pointplot , by default ('ci', 95)
        """        
        
        self.df = df
        self.target_field = target_field
        
        self.field_aliases = field_aliases                    
        
        self.num_fields = num_fields
        self.cat_fields = cat_fields
        self.date_fields = date_fields
                
        self.palette = palette
        self.color = color
        
        self.y_label_left = y_label_left
        self.y_label_right = y_label_right 
        
        self.errorbar = errorbar       
        
        self.axs = None
        
    def plot_grid(self,
                  ncols: int = 3,
                  subset: Iterable = None,
                  figsize: Tuple[int, int] = (25,8),
                  bins: dict = {},
                  periods: dict = {},
                  title: str = None):
        """Create the grid of population distribution vs event rates subgraphs. 

        Parameters
        ----------
        ncols : int, optional
            number of columns in the grid , by default 3
        subset : Iterable, optional
            Subset of fields, by default None
        figsize : Tuple[int, int], optional
            Figure size, by default (25,8)
        bins : dict, optional
            Dictionary od custom bins for numeric features, by default all features are split into 10 bins.
        periods : dict, optional
            Dictionary of custom periods for datetime features, by default it's days.  
        title : str, optional
            Title for the grid, by default None   
        """        
        
        if subset is None:
            subset =  [field for field in self.df.columns if field != self.target_field]
            
        nrows = math.ceil(len(subset)/ncols)
        
        _, self.axs = plt.subplots(ncols=ncols, 
                                   nrows=nrows,
                                   figsize=figsize,
                                   constrained_layout = True)

        for i, feature in enumerate( subset):
            
            if (nrows == 1) & (ncols == 1):
                ax = self.axs
            elif (nrows == 1) or (ncols == 1):                    
                ax = self.axs[i]                
            else:
                ax = self.axs[i//ncols, i%ncols]
            
            if feature not in self.df.columns or feature == self.target_field:
                raise ValueError(f'Invalid feature: {feature}') 
            
            if feature in self.num_fields:
                self.plot_continuous_feature(feature,
                                             ax = ax,
                                             bins = bins.get(feature, 10))
            
            elif feature in self.date_fields:
                self.plot_temp_feature(feature, ax=ax, period = periods.get(feature, 'D') )
                
            else:    
                self.plot_categorical_feature(feature, ax = ax) 
                
        if title is not None:        
            plt.suptitle(title)                         
    
    def plot_categorical_feature(self, field: str, ax: plt.Axes = None):
        
        order  = self.df.groupby(field)[self.target_field].mean().sort_values(ascending = False).index    
        
        countplot_kwargs = {'data' : self.df, 'x' : field, 'palette' : self.palette, 'order' : order}
        ax = self._plot_countplot(countplot_kwargs, ax)
            
        ax2 = ax.twinx()        
        sns.pointplot(data=self.df,
                      x=field,
                      y=self.target_field,
                      color=self.color,  
                      join=False,
                      errorbar = self.errorbar,
                      ax=ax2,
                      order=order)
                    
        self._add_text(ax, ax2, field)       
        
        return (ax, ax2)   
    
    def plot_temp_feature(self, field: str, period = 'M', ax: plt.Axes = None):
        
        dates_truncated = self.df[field].dt.to_period(period)
        order  = dates_truncated.drop_duplicates().sort_values()    
        
        countplot_kwargs = {'x' : dates_truncated, 'palette' : self.palette, 'order' : order}
        ax = self._plot_countplot(countplot_kwargs, ax)
            
        ax2 = ax.twinx()        
        sns.pointplot(data=self.df,
                      x=dates_truncated,
                      y=self.df[self.target_field],
                      color=self.color,  
                      join=False,
                      errorbar = self.errorbar,
                      ax=ax2,
                      order=order)
                    
        self._add_text(ax, ax2, field)       
        
        return (ax, ax2)     

    def plot_continuous_feature(self, field: str, bins: int = 10, ax: plt.Axes = None):        
        x =  pd.cut(self.df[field], bins)
        
        countplot_kwargs = {'x': x, 'palette': self.palette}
        ax = self._plot_countplot(countplot_kwargs, ax)
            
        ax2 = ax.twinx()
        sns.pointplot(x = x,
                      y = self.df[self.target_field],
                      color = self.color,  
                      join = False,
                      errorbar = self.errorbar,
                      ax = ax2)         
                  
        self._add_text(ax, ax2, field)     
        
    
    def _plot_countplot(self, countplot_kwargs: dict, ax: plt.Axes = None):
        
        if ax is None:
            ax = sns.countplot(**countplot_kwargs)
        else:
            sns.countplot(**countplot_kwargs, ax = ax )           
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
              
        return ax  
    
    def _add_text(self, ax1: plt.Axes,
                  ax2: plt.Axes,
                  field: str):
            
        if self.y_label_left is not None:
            ax1.set_ylabel(self.y_label_left)
            
        if self.y_label_right is not None:
            ax2.set_ylabel(self.y_label_right)
        
        ax1.set_xlabel(self.field_aliases.get(field, field))
        

def woe_table(feature: pd.Series, target: pd.Series, smoothing_factor: float = 1) -> pd.DataFrame:
    """Convenience function to get WoE table of the binned feature. 
    Useful if some custom binning of the feature is required.
    Otherwise binning_table from "optbinning" library can be used.

    Parameters
    ----------
    feature : pd.Series
        Binned feature to estimate  
    target : pd.Series
        Target variable
    smoothing_factor : float, optional
        Smoothing factor to take care of "division by zero" problem , by default 1

    Returns
    -------
    pd.DataFrame
        Binning table with WoE and IV for each bin     
    """        
    
    if len(feature) != len(target):
        raise ValueError("Lengths of feature and target don't match")
    
    if set(target.unique()) != {0, 1}:
        raise ValueError('Target should be binary, encoded as 0/1')
    
    grp = pd.crosstab(feature.values, target.values).rename(columns = {0:'num_good', 1:'num_bad'})    
    
    for cat in ['good', 'bad']:
        grp.loc[grp['num_'+cat] == 0, 'pc_'+cat] = smoothing_factor/(grp['num_'+cat].sum() 
                                                                     + 2*smoothing_factor)
        grp.loc[grp['num_'+cat] != 0, 'pc_'+cat] = grp['num_'+cat]/grp['num_'+cat].sum()         
   
    grp['br'] = grp.num_bad/( grp.num_bad +  grp.num_good)
    grp['woe'] = np.log(grp.pc_good/ grp.pc_bad)
    grp['iv'] = (grp.pc_good  - grp.pc_bad)*grp.woe
    
    return grp      
    
    
def feature_iv(feature: pd.Series,
               target: pd.Series,
               smoothing_factor: float = 1) -> float:
        
    """Calculate IV of the binned feature.

    Parameters
    ----------
    feature : pd.Series
        Binned feature to estimate  
    target : pd.Series
        Target variable
    smoothing_factor : float, optional
        Smoothing factor to take care of "division by zero" problem , by default 1

    Returns
    -------
    float
        Calculated IV
    """    
    grp =  woe_table(feature, target, smoothing_factor)
    return grp.iv.sum()       


def train_test_val_split(df: pd.DataFrame, 
                         target_name: str,
                         test_ratio: float =0.2,
                         val_ratio: float = 0.2,
                         num_features: Iterable = [],
                         cat_features: Iterable = [],
                         stratify: bool = True,
                         random_state: int = None) -> Tuple[pd.DataFrame,
                                                            pd.DataFrame,
                                                            pd.DataFrame,
                                                            pd.DataFrame,
                                                            pd.DataFrame,
                                                            pd.DataFrame]:
                             
    """Convenience function for splitting a dataframe into train, test and validation samples.
    All categorical features are converted to "category" dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to split
    target_name : str
        Name of the field with target variable in the dataframe
    test_ratio : float, optional
        Ratio for test sample, by default 0.2
    val_ratio : float, optional
        Ratio for validation sample, by default 0.2
    num_features : Iterable, optional
        List of numeric features , by default []
    cat_features : Iterable, optional
        List of categorical features, by default []
    stratify : bool, optional
        If True, stratification by target variable is applied, by default True    
    random_state : int, optional
        Random state, by default None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        x_train, x_val, x_test, y_train, y_val, y_test
    """    
    
    all_features = [*num_features, *cat_features]
    df[cat_features] = df[cat_features].astype('category')

    if stratify:
        stratification_arg = {'stratify' : df[target_name]}
    else:    
        stratification_arg = {}
    
    x_train, x_tmp, y_train, y_tmp= train_test_split(df[all_features],
                                                    df[target_name],
                                                    test_size=test_ratio + val_ratio,                                                 
                                                    random_state=random_state,
                                                    **stratification_arg
                                                    )
    
    if stratify:
        stratification_arg = {'stratify' : y_tmp}    

    x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp,
                                                    test_size=val_ratio / (val_ratio + test_ratio),
                                                    random_state=random_state,
                                                     **stratification_arg)
    
    return x_train, x_val, x_test, y_train, y_val, y_test

