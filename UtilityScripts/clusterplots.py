import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_h11_subset(x,h11,h21,h):
    idx = np.where(h11==h)[0]
    return x[idx], h11[idx], h21[idx]

def project_and_plt(ax,traincloud,testcloud,outlier_reject=(2.0,2.0)):
    '''helper function to project traincloud and test cloud by PCA to 
    2 components and plot them as point clouds. Points very far away are rejected
    by outlier_reject.
    Parameters:
    -----------
        ax: matplotlib.pyplot axis to plot along
        traincloud: tuple of x_train and color values to plot with, color can be None
                    in which case matplotlib will choose.
        testcloud: as above, for x_test instead of train
        outlier_reject: discard points very far away. tuple (x_parameter, y_parameter)
    '''
    [x_train,color_train]=traincloud
    [x_test,color_test]=testcloud
    pca = PCA(n_components=2)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    ax.scatter(x=x_test[:,0],y=x_test[:,1],marker='o',
                    c=color_test,alpha=0.1,label='test set')
    ax.scatter(x=x_train[:,0],y=x_train[:,1],marker='s',
                    c=color_train,label='train set')
    
    xax = np.hstack([x_test[:,0],x_train[:,0]])
    yax = np.hstack([x_test[:,1],x_train[:,1]])
    clipx,clipy = outlier_reject[0],outlier_reject[1]
    ax_range = lambda t,clip: [np.mean(t)-clip*np.std(t),np.mean(t)+clip*np.std(t)]
    ax.set_xlim(ax_range(xax,clipx))
    ax.set_ylim(ax_range(yax,clipy))
    return ax
    
def create_histogram(ax,y_te_h21,y_tr_h21):
    '''plot histogram of train and test h21 values'''
    ax.hist(y_te_h21,alpha=0.4,label='test data')
    ax.hist(y_tr_h21,label = 'train data')
    ax.set_xlabel('h21')
    ax.set_ylabel('population')
    ax.legend(loc='best')
    return ax
    
def plot_kmeans_scores(inertia,silhouette,k_range,h):
    '''plot inertia and silhouette scores in k_range for a 
    Hodge number h'''
    f, ax = plt.subplots(1,2)
    f.tight_layout()
    ax[0].plot(k_range,inertia)
    ax[0].scatter(k_range,inertia)
    ax[0].grid('both')
    ax[1].plot(k_range[1:],np.round(silhouette,2))
    ax[1].scatter(k_range[1:],np.round(silhouette,2))
    ax[1].grid('both')
    print('Hodge number:',int(h))
    plt.show()
    
def get_kmeans_scores(x,k):
    '''get inertia and silhouette scores for a point-cloud k and
    k clusters'''
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(x)
    inertia =kmeans.inertia_
    if k >=2:
        silhouette = silhouette_score(x,y_pred)
    else:
        silhouette = None
    return inertia,silhouette

def kmeans_results(kmeans,x_tr_h11,x_te_h11):
    '''runs fit_predict, predict and transform to get predicted
    values for train and test points using kmeans. runs transform 
    on test set to get distance from every centroid.'''
    y_pred_tr = kmeans.fit_predict(x_tr_h11)
    y_pred_te = kmeans.predict(x_te_h11)
    centroid_dist = kmeans.transform(x_te_h11)
    return y_pred_tr, y_pred_te, centroid_dist

def h21_statistics(h21_train,h21_test,idx_representatives):
    '''Takes an h21 train and test set for a given h11, computes
    the typical representatives of the test set and '''
    h21_typical= h21_test[idx_representatives].astype('int')
    h21 = np.hstack([h21_train,h21_test])
    h21_mean = np.mean(h21)
    h21_std = np.std(h21)
    h21_min = np.min(h21)
    h21_max = np.max(h21)
    return [h21_min,h21_max,h21_mean,h21_std,h21_typical]