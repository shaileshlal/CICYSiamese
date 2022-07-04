import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt


class StratifiedSampler():
    '''
    Helpers to carry out stratified sampling of the CICY dataset. 
    Declare an object while passing to it the following:
    
    Parameters:
    ----------
        hodge: array of the relevant Hodge numbers
        min_sample: minimum number of samples to draw into train
            for every unique value of Hodge number
        ratio: ratio of the population for each unique Hodge 
            number to draw into train
        random_state: seed for random sampling. 0 by default.
    Methods:
    -------
        sampler: sample from dataset by every unique Hodge number
            returns train, test
        plot_population: plots the population histograms for train
            test and total.
            
    '''
    def __init__(self,df,h,min_sample,ratio,random_state=0):
        self.df = df # the whole pandas dataframe
        self.h = h #'h11' or 'h21'
        self.min_sample = min_sample
        self.ratio = ratio
        self.random_state = random_state
        self.hodge = df[h]
        self.h_values = np.unique(self.hodge)
        self.h_indices = [np.where(self.hodge==val)[0] for val in self.h_values]
        self.idx_train, self.idx_test = self.sampler()
    
    def sampler(self):
        '''sample from dataset by every unique Hodge number
           returns train, test'''
        train,test = [],[]
        for hidx in self.h_indices:
            max_sample = int(self.ratio*len(hidx))
            if len(hidx)==1:
                list_remove,list_keep = list(hidx),list(hidx)
            else:
                n_remove = max(self.min_sample,max_sample) # number of elements to draw into train set
                rng = np.random.default_rng(self.random_state)
                list_remove = rng.choice(hidx,n_remove,replace=False)
                list_keep = [idx for idx in hidx if not idx in list_remove]
            train.append(list_remove), test.append(list_keep)
        return train, test
    
    def plot_population(self):
        '''plots the population histograms for train
        test and total.'''
        population = lambda x : [len(h) for h in x]
        plt.bar(self.h_values,population(self.h_indices),alpha=0.3,color='orange',label='CICY3')
        plt.bar(self.h_values,population(self.idx_test),alpha=0.5,color='green',label='test')
        plt.bar(self.h_values,population(self.idx_train),alpha=0.6,label='train+val')
        plt.legend(loc='best')
        plt.show()
        
    def stratified_split(self,cols=None):
        # flatten list of arrays into single arrays which contain all the idx of train
        # and all the idx of test
        idx_train, idx_test = np.hstack(self.idx_train), np.hstack(self.idx_test)
        if cols==None: # if not explicitly specified to retain which columns, retain them all
            x_for_splitting = self.df
        else:
            x_for_splitting = self.df[cols+[self.h]] # retain specified columns along with h col
        x_train, x_test = x_for_splitting.iloc[idx_train], x_for_splitting.iloc[idx_test]
        return x_train.reset_index(drop=True), x_test.reset_index(drop=True)
    
    def pretty_print(self):
        '''Prints out the Hodge number demographic of train test splits
        as a markdown table. If there is only one element for a given
        Hodge number in the whole dataset then that element appears in
        both the train and the test sets
        
        Parameters:
        ----------
            None
        Returns:
        -------
            None, will print to screen in markdown
        '''
        keys = ['h','train population','test population','train+test','h population']
        values = []
        for i,h in enumerate(self.h_values):
            values.append([h,len(self.idx_train[i]),len(self.idx_test[i]),
                           len(self.idx_train[i])+len(self.idx_test[i]),
                           len(self.h_indices[i])])
        population = pd.DataFrame(values,columns=keys)
        print(population.to_markdown(tablefmt='grid',index=False))
        
class CicyPad(BaseEstimator,TransformerMixin):
    '''Helper class to pad CICY matrices to a uniform size. 
    Parameters:
    -----------
        upsampled_w: number of columns to be upsampled to
        upsampled_h: number of rows to be upsampled to
        pad_type: 'inter'(polation) or 'constant'; the type of padding.
        ravel: True or False, to ravel or not.
    has a fit and a transform method inherited from BaseEstimator and 
    TransformerMixin.
    '''
    def __init__(self,upsampled_w,upsampled_h,pad_type='inter',ravel=False):
        self.upsampled_w = upsampled_w
        self.upsampled_h = upsampled_h
        self.pad_type = pad_type
        self.ravel = ravel
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return np.stack(X.apply(self.arraypad))
    
    def arraypad(self,arr): # support padtype 'inter' and 'constant'
        if self.pad_type == 'inter':
            return self.arraypad_inter(arr)
        elif self.pad_type == 'constant':
            return self.arraypad_constant(arr)
        else:
            raise ValueError('pad_type should be inter or constant')
    
    def arraypad_inter(self,arr): 
        '''pads input array by linear interpolation'''
        arr=np.array(arr)
        if arr.shape[1]==1 and arr.shape[0]>1:
            arr = np.hstack([arr-0.5,arr+0.5])
        if arr.shape[0]==1 and arr.shape[1]>1:
            arr = np.vstack([arr-0.5,arr+0.5])
        elif arr.shape==(1,1):
            a = arr[0,0]
            arr = np.array([[a-0.5,a+0.5],[a-0.5,a+0.5]])
        W,H = arr.shape
        xrange = lambda x: np.linspace(0, 1, x)
        f = interp2d(xrange(H),xrange(W),arr, kind='linear')
        arr = f(xrange(self.upsampled_h), xrange(self.upsampled_w))
        if self.ravel:
            arr = arr.ravel() # roll out the matrix into a 1d vector
        return arr
    
    def get_pad(self,row,col):
        '''computes how much to pad along the top, left, right, bottom'''
        padtop = (self.upsampled_h-row)//2
        padbottom = self.upsampled_h-row-padtop
        padleft = (self.upsampled_w-col)//2
        padright = self.upsampled_w-col-padleft
        return padtop,padbottom,padleft,padright
    
    def arraypad_constant(self,arr):
        '''pads input array by a constant number'''
        arr=np.array(arr)
        row, col = arr.shape
        padt,padb,padl,padr = self.get_pad(row,col)
        arr = np.pad(arr,((padt,padb),(padl,padr)),constant_values=-1)
        if self.ravel:
            arr = arr.ravel() # roll out the matrix into a 1d vector
        return arr
    
def upsample(x,y,pad):
    cicydata=np.empty((1,*x[0].shape)) # seed array to stack along
    cicyh11=[]
    hdata=np.array([1])
    h = np.unique(y)
    for i in h:
        arr=x[np.where(y==i)[0]]
        if len(arr)<pad:
            arr=np.pad(arr,[(0,pad-len(arr)),(0,0),(0,0)],'wrap')
            cicydata=np.vstack((cicydata,arr))
        else:
            cicydata=np.vstack((cicydata,arr))
        hdata = np.hstack((hdata,np.full((len(arr),),i)))
    return cicydata[1:], hdata[1:]

def pretty_print_results(y_test,y_pred):
    '''pretty printing results'''
    columns = ['h','test population','predicted','true positives','precision',
               'recall','percentage of test set isolated']
    y_val = np.unique(y_test)
    table_entries =[]
    for h in y_val:
        test_pop = len(np.where(y_test==h)[0])
        predicted = len(np.where(y_pred==h)[0])
        true_pos = len(np.where(np.logical_and(y_test==h,y_pred==h))[0])
        precision = round(true_pos/predicted,2) if predicted!=0 else None
        recall = round(true_pos/test_pop,2)
        excluded = round(len(np.where(y_pred==h)[0])/len(y_test)*100,2)
        table_entries.append([int(h),test_pop,predicted,true_pos,precision,
                             recall,excluded])
    results = pd.DataFrame(table_entries,columns=columns)
    print(results.to_markdown(tablefmt='grid',index=False))
    
def analyze_knn_cv(cross_val_results):
    '''helper function to output cross validation scores for KNN classifier'''
    uniform_idxs=[elt['knn Classifier__weights']=='uniform' 
                  for elt in cross_val_results['params']]
    distance_idxs=[elt['knn Classifier__weights']=='distance' 
                   for elt in cross_val_results['params']]

    uniform_scores = cross_val_results['mean_test_score'][uniform_idxs]
    distance_scores = cross_val_results['mean_test_score'][distance_idxs]

    num_neighbors_uniform = [elt['knn Classifier__n_neighbors'] 
                             for elt in cross_val_results['params'] 
                             if elt['knn Classifier__weights']=='uniform']

    num_neighbors_distance = [elt['knn Classifier__n_neighbors'] 
                              for elt in cross_val_results['params'] 
                              if elt['knn Classifier__weights']=='distance']
    
    plt.title('Accuracy acores across number of nearest neighbors obtained by k-fold cross validation')
    plt.plot(num_neighbors_uniform,uniform_scores,label='uniform')
    plt.plot(num_neighbors_distance,distance_scores,label='distance')
    plt.xlabel('neighbors')
    plt.ylabel('accuracy score')
    plt.legend(loc='best')
    plt.show()
        
