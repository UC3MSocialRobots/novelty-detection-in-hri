import pandas as pd
import numpy as np
import scipy 
import arff as arff   # Downloaded from: http://code.google.com/p/arff/
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pylab import *
from itertools import cycle
from sklearn import metrics
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.mixture import GMM, DPGMM
from sklearn.hmm import GaussianHMM
from sklearn.covariance import MinCovDet
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import user_data_loader as udl
from IPython.display import HTML
from IPython.html import widgets
from IPython.display import display, clear_output
import lsanomaly
import print_ske as pp

red = 'STAND_POINTING_RIGHT'
blue = 'STAND_POINTING_LEFT'
green = 'STAND_POINTING_FORWARD'

global GMM_n, one_n, lsa_n, K_n, GMM_s, one_s, lsa_s  #Novelty Scores for each algorithm, those ''_n are for noise score, ''_s are for strangeness score 
global K_GMM_n, K_KMeans_n, K_GMM_s, K_KMeans_s #K_GMM_n, K_KMeans_n are the noise curiosity factors for each algorithm
                                                #K_GMM_s, K_KMeans_s are the strangeness curiosity factors for each algorithm
                                                #Ks is a list containing the 4 above mentioned parameters



'''
---------
FUNTIONS TO RELOAD AND DISPLAY RESULTS
---------
'''

def compute_and_reload_figures(normal_users, queue, users_normal, users, Ks,  name=''):

    '''
    Receives the list of normal_users and the queue. Also users_normal and users, with the form of [[number_user, pose]...]
    Recevices Ks, the list of curiosity factors for the algorithms
    Can receive a name to save the figure displayed 
    
    Calls compute scores to obtain the strangeness score and noise score for all the last entry in the queue
    Computes the colors of the bars depending on the values of the scores
    Plots a bar graph with the scores and the names
    '''
    
    global GMM_n, one_n, lsa_n, K_n, GMM_s, one_s, lsa_s  #Novelty Scores for each algorithm, those ''_n are for noise score, ''_s are for strangeness score 
    
    GMM_n = []
    one_n = []
    lsa_n = []
    K_n = []
    GMM_s = []
    one_s = []
    lsa_s = []
    K_s =[]
    
    compute_scores(normal_users, queue, Ks) # Calls the function to compute scores, that updates GMM_n, one_n, lsa_n, K_n, GMM_s, one_s, lsa_s 

    scores_n = np.array([K_n[0],lsa_n[0]+0.01,one_n[0]+0.01,GMM_n[0]]) #Create a numpy array with the noise scores to display in the graph
    names_n = ('KM','LSA', 'SVM1C','GMM' ) #names to display in the noise score graph
    

        
    scores_s = np.array([K_s[0],lsa_s[0]+0.01,one_s[0]+0.01,GMM_s[0]]) #Create a numpy array with the noise scores to display in the graph
    names_s = ('KM','LSA', 'SVM1C','GMM') #names to display in the strangeness score graph

    print scores_s

    # If the entry is detected as not interesting by all algorithms, the strangeness score is not displayed
    if GMM_n[0]>=1 and one_n[0]>=1 and lsa_n[0]>=1 and K_n[0]>=1:
        scores_s = np.array([0,0,0,0])

    # Compute colors corresponding to the score value
    # noise = red, interesting = green
    # known = red, strange = green

    colors_n = []
    colors_s = []

    for n in scores_n.tolist():
        if n >= 1:
            colors_n.append('red')
        else:
            colors_n.append('green')

    for n in scores_s.tolist():
        if n >= 1:
            colors_s.append('green')
        else:
            colors_s.append('red')


    #Plot the figures
            
    f= plt.figure(figsize=(15,5))

    # Print normal users and the last entry introduced to the system in black
    ax1 =  f.add_subplot(1,4,3, projection='3d')
    users_normal_new = list(users_normal)
    users_normal_new.append(users[-1])
    pp.print_users(users_normal_new, ax1)

    # Print all users, with the last entry introduced to the system in black
    ax2 =  f.add_subplot(1,4,1, projection='3d')
    pp.print_users(users, ax2)

    # Display names and scores of the algorithms
    ax3 =  f.add_subplot(1,4,2)
    ax4 =  f.add_subplot(1,4,4)
    y_pos = np.arange(len(names_n))

    ax3.barh(y_pos, scores_n, align='center', alpha=0.4, color = colors_n)
    ax4.barh(y_pos, scores_s, align='center', alpha=0.4, color = colors_s)
        
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names_n)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(names_s)
    ax3.set_title('noise score')
    ax4.set_title('strangeness score')
    
    #f.savefig('/Users/_USER_/Images/'+name+'.pdf', format='pdf')

'''
---------
FUNTIONS TO CALCULATE NOVELTY SCORES
---------
'''

def compute_scores(normal_users, queue, Ks=[]):

    '''
        Calculates the novelty scores (noise and strangeness) for the 4 algotithms
        Receives the list of normal users and the queue (all users) and the list of curiosity factors Ks
        Updates the global variables GMM_n, one_n, lsa_n, K_n, GMM_s, one_s, lsa_s with the results 
    '''
    
    global GMM_n, one_n, lsa_n, K_n, GMM_s, one_s, lsa_s, K_s #Novelty Scores for each algorithm, those ''_n are for noise score, ''_s are for strangeness score 

    GMM_n = []
    one_n = []
    lsa_n = []
    K_n = []
    GMM_s = []
    one_s = []
    lsa_s = []
    K_s = []

    K_GMM_n, K_KMeans_n, K_GMM_s, K_KMeans_s = Ks #K_GMM_n, K_KMeans_n are the noise curiosity factors for each algorithm
                                                  #K_GMM_s, K_KMeans_s are the strangeness curiosity factors for each algorithm
                                                  #Ks is a list containing the 4 above mentioned parameters
    

    '''
    
    For One_class_SVM and LSA, when asked to predict the new entry, a label is directly returned 
        LSA: 'anomaly' or '0' (normal)

        One One_class_SVM: -1 (anomaly) or 1 (normal)

    GMM and K means predict a fitting score. The novelty score is obtained calculating the zscore of the entry compared with the scores of all other entries, calling 
    the function get_score_last_item
        If the zscore returned >= 1 the new entry is anomalous

    '''

    '''
    Noise scores are computed with the queue as the base of knowledge, fitting all the entries but the last to the algorithm
    '''                                    
    B = GMM(covariance_type='full', n_components = 1)
    B.fit(queue[0:-1])
    x = [B.score([i]).mean() for i in queue]
    GMM_n.append(get_score_last_item(x, K_GMM_n))


    K = KMeans(n_clusters=1)
    K.fit(queue[0:-1])
    x = [K.score([i]) for i in queue]
    K_n.append(get_score_last_item(x, K_KMeans_n))

    oneClassSVM = OneClassSVM(nu=0.1)
    oneClassSVM.fit(queue[0:-1])
    x = oneClassSVM.predict(np.array([queue[-1]]))
    if x == -1:
        one_n.append(1)
    if x == 1:
        one_n.append(0)
    
    X = np.array(queue[0:-1])
    anomalymodel = lsanomaly.LSAnomaly()
    anomalymodel.fit(X)
    x = anomalymodel.predict(np.array([queue[-1]])) 
    if x == ['anomaly']:
        lsa_n.append(1)
    if x == [0]:
        lsa_n.append(0)

    '''
    Strangeness scores are computed with the normal users as the base of knowledge, fitting normal users to the algorithm
    ''' 

    normal_and_new = normal_users + [queue[-1]] #List to be passed to get_score_last_item to calculate the zscore of the last item, the new entry

    B = GMM(covariance_type='full', n_components = 1)
    B.fit(normal_users)
    x = [B.score([i]).mean() for i in normal_and_new]
    GMM_s.append(get_score_last_item(x, K_GMM_s))


    K = KMeans(n_clusters=1)
    K.fit(normal_users)
    x = [K.score([i]) for i in normal_and_new]
    K_s.append(get_score_last_item(x, K_KMeans_s))

    oneClassSVM = OneClassSVM(nu=0.1)
    oneClassSVM.fit(normal_users)
    x = oneClassSVM.predict(np.array([queue[-1]]))
    if x == -1:
        one_s.append(1)
    if x == 1:
        one_s.append(0)

    anomalymodel = lsanomaly.LSAnomaly()
    X = np.array(normal_users)
    anomalymodel.fit(X)
    x = anomalymodel.predict(np.array([queue[-1]])) 
    if x == ['anomaly']:
        lsa_s.append(1)
    if x == [0]:
        lsa_s.append(0)

    return GMM_n, one_n, lsa_n, K_n, GMM_s, one_s, lsa_s, K_s


def get_score_last_item(x, K_curiosity):

    ''' Obtains a normalized (z) score of the last item of a list, with respect to the other items'''    
    ser = pd.Series(x, dtype=float)
    old = ser[0:-1]
    new = ser[ser.size-1]
    
    return abs((new-old.mean())/(old.std()*K_curiosity))

'''
-------------
FUNTIONS TO START THE SYSTEM
-------------
'''

def start_users(number_users, pose, indexes=[]):
    '''
        Starts the system creating 

        users has a [number_user, pose] form
        normal_users and queue are lists of users in data form

        Uses  a number of users, number_users,  all posing in the same direction, initial_pose

        The users selected are random

        It takes the median value of the users
    '''
    import random

    if indexes == []:
	    indexes = [i for i in np.arange(30)]
	    indexes.pop(indexes.index(0)) # There is no 'user00' data, so we remove it
	    indexes.pop(indexes.index(6))  # There is no 'user06' data, so we remove it
	    indexes.pop(indexes.index(12)) # There is no 'user12' data, POINTING LEFT so we remove it
    

    users = []

    pose = pose

    for i in xrange(1,number_users):
        j = random.choice(indexes)
        users.append([j, pose])
        indexes.pop(indexes.index(j))

    normal_users = get_median_users(users)
    queue = normal_users
    
    return users, normal_users, queue

def get_median_users(users):

    '''
    Returns a list of the median values for each user in users
    users = [[number, pose],[number2, pose]...]

    '''
    list_users_median = []
    for u in users:
            n = divide_user_by_pose('data/exp03-user'+str(u[0]).zfill(2)+'.arff', u[1])
            mean = np.median(n, axis=0)
            list_users_median.append(mean)

    return list_users_median


'''
--------------
FUNTIONS TO LEARN POSES
--------------
'''
def add_user_median(l_users, new_user_index, pose):
    '''
        Adds the median of the new user to the passed list and returns it
    '''
    l_new = list(l_users)
    new_user = divide_user_by_pose('data/exp03-user'+str(new_user_index).zfill(2)+'.arff', pose)
    median = np.median(new_user,axis =0)
    
    l_new.append(median)
    
    return l_new

'''
--------------
FUNCTIONS TO EXTRACT AND NORMALIZE DATA
--------------
'''

def divide_user_by_pose(file, pose):

    ''' 
    Returns the normalized data of a desired pose from a user file, in an numpy array
    '''    
    uf = udl.load_user_file(file)
    
    multiind_first, multiind_second = udl.make_multiindex(udl.joints, udl.attribs)
    uf.columns = pd.MultiIndex.from_arrays([list(multiind_first), list(multiind_second)], names=['joint', 'attrib'])
    orig_torso, df_normalized = udl.normalize_joints(uf, 'torso')
    
    uf.update(df_normalized)
    uf.torso = uf.torso - uf.torso
    
    uf.columns = udl.index
    
    drops = list(uf.columns[84:123])+['h_seqNum', 'h_stamp', 'user_id']
    uf2 = uf.drop(drops,1).groupby('pose')
    group_pose_with_label = uf2.get_group(pose)
    group_pose = group_pose_with_label.drop('pose',1)
    
    return group_pose.values



'''
-------------
OTHER FUNCTIONS NOT USED IN THE NOTEBOOK
-------------

'''

def compute_print_scores(normal_users, queue):

    K_GMM_n, K_KMeans_n, K_GMM_s, K_KMeans_s = Ks

    print 'novelty score GMM'
    B = GMM(covariance_type='full', n_components = 1)
    B.fit(queue)
    x = [B.score([i]).mean() for i in queue]
    print get_score_last_item(x, K_GMM_n)

    print 'novelty score OneClassSVM'
    x = anom_one_class(queue, [queue[-1]])
    print x[-1]

    print 'novelty score LSA'
    anomalymodel = lsanomaly.LSAnomaly()
    X = np.array(queue)
    anomalymodel.fit(X)
    print anomalymodel.predict(np.array([queue[-1]]))

    print 'novelty score degree K_means'
    K = KMeans(n_clusters=1)
    K.fit(queue)
    x = [K.score([i]) for i in queue]
    print get_score_last_item(x, K_KMeans_n)

    normal_and_new = normal_users + [queue[-1]]

    print 'degree of belonging to known class GMM'
    B = GMM(covariance_type='full', n_components = 1)
    B.fit(normal_users)
    x = [B.score([i]).mean() for i in normal_and_new]
    print get_score_last_item(x, K_GMM_s)

    print 'degree of belonging to known class OneClassSVM'
    x = anom_one_class(normal_users, [queue[-1]])
    print x[-1]

    print 'degree of belonging to known class LSA'
    anomalymodel = lsanomaly.LSAnomaly()
    X = np.array(normal_users)
    anomalymodel.fit(X)
    print anomalymodel.predict(np.array([queue[-1]]))

    print 'degree of belonging to known class K_means'
    K = KMeans(n_clusters=1)
    K.fit(normal_users)
    x = [K.score([i]) for i in normal_and_new]
    print get_score_last_item(x, K_KMeans_s)