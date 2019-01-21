#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:54:19 2018

@author: Ruben Buendia
"""
import os
os.chdir("Path_To_Directory")

#Load Libreries
import numpy as np
import pandas as pd
import VennABERS as fir #Thus function can be found at https://github.com/ptocca/VennABERS/blob/master/VennABERS.py
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import random
import subprocess
import gc

from sklearn.datasets import load_svmlight_files


i = 1
m = 1

modelname = "model.txt"

Target_All = pd.read_csv('Path_to_File_of_Target_Library', index_col = False, names = ['Smile', 'AZID', 'Activity', 'Activity_Binary','AZID_Bis', 'Sparse_Vectors'])
Hits = Target_All.loc[(Target_All["Activity_Binary"]=="Active")] 
TF = pd.read_csv('Path_to_File_of_Topological_Framework_Library', sep = " ", index_col = False)
MF = pd.read_csv('Path_to_File_of_Molecular_Framework_Library', sep = " ", index_col = False)

p_Hits = pd.DataFrame()
Random_Hits = pd.DataFrame()
Cal_Val_df_p = pd.DataFrame()
Cal_Val_df_Random = pd.DataFrame()


SeedsNr = 11
SelectionNr = 50000
for i in range(1, SeedsNr):
    print ("We are on Seed %d" % (i))


    
#For each seed, select a different set of 100K compounds, put half on training, half on calibration and the rest of the library in predict
    Train_Cal = random.sample(list(Target_All['AZID']), 100000)
    TrainingID = Train_Cal[:50000]
    CalID = Train_Cal[50000:]
    PredictID = np.setdiff1d(Target_All['AZID'],Train_Cal)
    p_PredictID = PredictID
    Random_PredictID = PredictID

    
    
#Save the IDs for follow up, (PredictID can be obtained by substraction)    
    np.savetxt('Path_To_Directory/TrainingID_%s' %(i), TrainingID, fmt='%s')
    np.savetxt('Path_To_Directory/CalID_%s' %(i), CalID, fmt='%s')

#Save sets in SVM format
    Target_SVM = pd.concat([Target_All['Activity_Binary'].replace(['Active', 'Inactive'],['1 ', '-1 ']) + Target_All['Sparse_Vectors'], Target_All['AZID']], axis = 1).set_index('AZID')
    Target_SVM.columns=['a']
    Training = list(Target_SVM.loc[TrainingID]['a'])
    np.savetxt('Path_To_Directory/p_Training', Training, fmt='%s')
    np.savetxt('Path_To_Directory/Random_Training', Training, fmt='%s')

    Calibration = list(Target_SVM.loc[CalID]['a'])
    np.savetxt('Path_To_Directory/Calibration', Calibration, fmt='%s')
    Predict = list(Target_SVM.loc[PredictID]['a'])
    np.savetxt('Path_To_Directory/p_Predict', Predict,  fmt='%s')
    np.savetxt('Path_To_Directory/Random_Predict', Predict,  fmt='%s')

    p_Training = Training
    Random_Training = Training
    p_Predict = Predict
    Random_Predict = Predict
    
#Number of Hits, TF and MF in Predict
    PredictID_ds = pd.DataFrame(PredictID, columns=['AZID'])
    Predict_Hits_df = PredictID_ds.merge(Hits, left_on='AZID', right_on='AZID')
    Nr_Hits_inPredict = len(Predict_Hits_df)

    Predict_TF_df = PredictID_ds.merge(TF, left_on='AZID', right_on='AZID')
    Nr_TF_inPredict = Predict_TF_df.TFNum.nunique()
    
    Predict_MF_df = PredictID_ds.merge(MF, left_on='AZID', right_on='AZID')
    Nr_MF_inPredict = Predict_MF_df.MFNum.nunique()

#ITERATIONS

    IterationNr = 11
    for j in range(1, IterationNr):
          print ("We are on Iteration %d" % (j))
          print("...selecting %d compounds" %(SelectionNr))

#p Track
          pt_X,pt_y,cal_X,cal_y,test_X,test_y  = load_svmlight_files(['p_Training', 'Calibration', 'p_Predict'])
#
          Cost = subprocess.getoutput("/LIBLINEAR2.20/train -C %s" %('p_Training')).split('\n')[-1].split(' ')[3]
      
          os.system("/LIBLINEAR2.20/train -c %s p_Training model.txt" %(Cost))
          os.system("/LIBLINEAR2.20/predict Calibration model.txt calScores.txt")
          os.system("/LIBLINEAR2.20/predict p_Predict model.txt testScores.txt")

          cal_scores = np.loadtxt("calScores.txt",  delimiter=' ', skiprows = 0)
          test_scores = np.loadtxt("testScores.txt",  delimiter=' ', skiprows = 0)


          cal_y = (cal_y + 1.0 ) / 2
          test_y = (test_y + 1.0 ) / 2
          pt_y = (pt_y + 1.0 ) / 2

          cal_points = zip(cal_scores,cal_y)

          p0,p1 = fir.ScoresToMultiProbs(cal_points,list(test_scores))
          p = p1 / (1 - p0 + p1)


          p_predict_df = pd.DataFrame({'AZID':p_PredictID, 'Activity':test_y, 'p':p, 'p0':p0, 'p1':p1, 'Iteration':j, 'seed':i})
          p_prediction = p_predict_df.sort_values(by="p", axis=0, ascending=False)
          p_selection = p_prediction.head(n=SelectionNr)
          
          p_Training = p_Training + list(Target_SVM.loc[p_selection['AZID']]['a'])
          np.savetxt('Path_To_Directory/p_Training', p_Training, fmt='%s')
          p_PredictID = np.setdiff1d(p_PredictID,p_selection['AZID'])
          p_Predict = list(Target_SVM.loc[p_PredictID]['a'])
          np.savetxt('Path_To_Directory/p_Predict', p_Predict, fmt='%s')
          
          p_Hits_new = p_selection.loc[p_selection['Activity'] == 1]
          p_Hits = p_Hits.append(p_Hits_new).reset_index(drop=True)
          print(len(p_Hits_new))

          Cal_Val_new_p = pd.DataFrame({'min_Hits':p_selection.p0.sum(), 'max_Hits':p_selection.p1.sum(), 'Expected_Hits':p_selection.p.sum(), 'Actual_Hits':p_selection.Activity.sum(), 'Iteration':j, 'seed':i}, index=['p'])
          Cal_Val_df_p = Cal_Val_df_p.append(Cal_Val_new_p)




#Random Track
          pt_X,pt_y,cal_X,cal_y,test_X,test_y  = load_svmlight_files(['Random_Training', 'Calibration', 'Random_Predict'])
          os.system("/LIBLINEAR2.20/train -c %s Random_Training model.txt" %(Cost))
          os.system("/LIBLINEAR2.20/predict Calibration model.txt calScores.txt")
          os.system("/LIBLINEAR2.20/predict Random_Predict model.txt testScores.txt")

          cal_scores = np.loadtxt("calScores.txt",  delimiter=' ', skiprows = 0)
          test_scores = np.loadtxt("testScores.txt",  delimiter=' ', skiprows = 0)
          abs_test_scores = abs(test_scores)
          Inv_abs_test_scores = 1- abs(test_scores)



          cal_y = (cal_y + 1.0 ) / 2
          test_y = (test_y + 1.0 ) / 2
          pt_y = (pt_y + 1.0 ) / 2

          cal_points = zip(cal_scores,cal_y)

          p0,p1 = fir.ScoresToMultiProbs(cal_points,list(test_scores))
          p = p1 / (1 - p0 + p1)

          
          Random_predict_df = pd.DataFrame({'AZID':Random_PredictID, 'Activity':test_y, 'p':p, 'p0':p0, 'p1':p1, 'Iteration':j, 'seed':i})
          Random_selection = Random_predict_df.sample(n=SelectionNr)
          Random_Training = Random_Training + list(Target_SVM.loc[Random_selection['AZID']]['a'])
          np.savetxt('Path_To_Directory/Random_Training', Random_Training, fmt='%s')
          Random_PredictID = np.setdiff1d(Random_PredictID,Random_selection['AZID'])
          Random_Predict = list(Target_SVM.loc[Random_PredictID]['a'])
          np.savetxt('Path_To_Directory/Random_Predict', Random_Predict, fmt='%s')
          Random_Hits_new = Random_selection.loc[Random_selection['Activity'] == 1]
          Random_Hits = Random_Hits.append(Random_Hits_new).reset_index(drop=True)
          print(len(Random_Hits_new))


          Cal_Val_new_Random = pd.DataFrame({'min_Hits':Random_selection.p0.sum(), 'max_Hits':Random_selection.p1.sum(), 'Expected_Hits':Random_selection.p.sum(), 'Actual_Hits':Random_selection.Activity.sum(), 'Iteration':j, 'seed':i}, index=['Random'])
          Cal_Val_df_Random = Cal_Val_df_Random.append(Cal_Val_new_Random)


    Target_SVM = []
    Predict = []	 
    Training = []
    Calibration = []
    p_Training = []
    Random_Training = []
    p_Predict = []
    Random_Predict = []
    

    gc.collect()
    
#Out of the for loops;    #Add TF & MF

p_Hits = p_Hits.merge(TF, left_on='AZID', right_on='AZID', how='left')
p_Hits = p_Hits.merge(MF, left_on='AZID', right_on='AZID', how='left')

p_Hits.to_csv('p_Hits_Target.csv')
print(len(p_Hits))

Random_Hits = Random_Hits.merge(TF, left_on='AZID', right_on='AZID', how='left')
Random_Hits = Random_Hits.merge(MF, left_on='AZID', right_on='AZID', how='left')

Random_Hits.to_csv('Random_Hits_Target.csv')
print(len(Random_Hits))


#Calculate summary results

Hits_Summary = pd.DataFrame.from_records([(p_Hits.groupby(['seed','Iteration']).AZID.count()), (Random_Hits.groupby(['seed','Iteration']).AZID.count())], index = ('p','Random')).T
Hits_Summary_Percent = Hits_Summary*100/Nr_Hits_inPredict
Hits_Summary_Percent.to_csv('Hits_Summary_Percent.csv')
Hits_Summary_Percent = pd.read_csv('Hits_Summary_Percent.csv')
Hits_cummulative = Hits_Summary_Percent; Hits_cummulative['p'] = Hits_cummulative.groupby('seed')['p'].cumsum();Hits_cummulative['Random'] = Hits_cummulative.groupby('seed')['Random'].cumsum().T

TF_Summary = pd.DataFrame.from_records([(p_Hits.drop_duplicates(['TFNum','seed']).groupby(['seed','Iteration']).TFNum.nunique()), (Random_Hits.drop_duplicates(['TFNum','seed']).groupby(['seed','Iteration']).TFNum.nunique())], index = ('p','Random')).T
TF_Summary.to_csv('TF_Summary.csv')
TF_Summary_Percent = TF_Summary*100/Nr_TF_inPredict
TF_Summary_Percent.to_csv('TF_Summary_Percent.csv')
TF_Summary_Percent = pd.read_csv('TF_Summary_Percent.csv')
TF_cummulative = TF_Summary_Percent; TF_cummulative['p'] = TF_cummulative.groupby('seed')['p'].cumsum();TF_cummulative['Random'] = TF_cummulative.groupby('seed')['Random'].cumsum().T
                     
MF_Summary = pd.DataFrame.from_records([(p_Hits.drop_duplicates(['MFNum','seed']).groupby(['seed','Iteration']).MFNum.nunique()), (Random_Hits.drop_duplicates(['MFNum','seed']).groupby(['seed','Iteration']).MFNum.nunique())], index = ('p','Random')).T
MF_Summary.to_csv('MF_Summary.csv')
MF_Summary_Percent = MF_Summary*100/Nr_MF_inPredict
MF_Summary_Percent.to_csv('MF_Summary_Percent.csv')
MF_Summary_Percent = pd.read_csv('MF_Summary_Percent.csv')
MF_cummulative = MF_Summary_Percent; MF_cummulative['p'] = MF_cummulative.groupby('seed')['p'].cumsum();MF_cummulative['Random'] = MF_cummulative.groupby('seed')['Random'].cumsum().T
                     
                                                   
p_Cum_Summary = Hits_cummulative.join(TF_cummulative.loc[:, ["p", "Random"]], lsuffix='_Hits', rsuffix='_TF').loc[:, ['seed','Iteration',"p_Hits", "p_TF"]]
p_Cum_Summary = pd.concat([p_Cum_Summary, MF_cummulative["p"].rename("p_MF")], axis=1)
p_Cum_Summary.to_csv('p_Cum_Summary.csv')
Random_Cum_Summary = Hits_cummulative.join(TF_cummulative.loc[:, ["p", "Random"]], lsuffix='_Hits', rsuffix='_TF').loc[:, ['seed','Iteration',"Random_Hits", "Random_TF"]]; p_Cum_Summary = p_Cum_Summary.join(MF_cummulative.loc[:, ["p", "Random"]], lsuffix='_Hits', rsuffix='_MF').loc[:, ['seed','Iteration',"p_Hits", "p_TF", "p_MF"]]        
Random_Cum_Summary = pd.concat([Random_Cum_Summary, MF_cummulative["Random"].rename("Random_MF")], axis=1)
Random_Cum_Summary.to_csv('Random_Cum_Summary.csv')
              

Cal_Val_df_p.to_csv('Calibration_Validity_p.csv')
Cal_Val_df_Random.to_csv('Calibration_Validity_Random.csv')




#Calibration Validity Figure
Fig_CalVal, axes = plt.subplots(2, sharex=True, figsize=(12, 9),  gridspec_kw = {'height_ratios':[4, 1]})
sns.set_style("white")
sns.set_context("poster")

Cal_Val_p_df_long = pd.melt(Cal_Val_df_p, id_vars=['Iteration'],  value_vars=['min_Hits','Expected_Hits', 'Actual_Hits','max_Hits'])
ax1 = sns.violinplot("Iteration", hue="variable", y="value", data=Cal_Val_p_df_long, palette=['indianred','silver','limegreen','indianred'], ax = axes[0]);
ax1.set_title('HHRT1', fontweight='bold', fontsize=25)
    
Cal_Val__Random_df_long = pd.melt(Cal_Val_df_Random, id_vars=['Iteration'],  value_vars=['min_Hits','Expected_Hits', 'Actual_Hits','max_Hits'])
ax2 = sns.violinplot("Iteration", hue="variable", y="value", data=Cal_Val__Random_df_long,palette=['indianred','silver','limegreen','indianred'], ax = axes[1])

handles, _ = axes[0].get_legend_handles_labels()
axes[0].legend(handles, ["Min Hits", "Expected Hits", "Actual Hits","Max Hits"])
handles, _ = axes[1].get_legend_handles_labels()
axes[1].legend(handles, [])

axes[0].set_xlabel("")
axes[1].set_xlabel("Iteration", fontweight='bold', fontsize=22)
axes[0].set_ylabel("Venn-ABERS", fontweight='bold', fontsize=22)
axes[1].set_ylabel("Random",fontsize=22, fontweight='bold', labelpad=16)
axes[0].set_yscale('log')
axes[1].set_yscale('log')

ax1.set_yticks([100, 1000, 10000])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


ax2.set_yticks([1200, 1700])
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
Fig_CalVal.tight_layout()


Fig_CalVal.savefig('CalVal.eps', format='eps', dpi=1200)
Fig_CalVal.savefig('CalVal.svg', format='svg', dpi=1200)




##Cumms Figures
Fig_CumSum_Hits, ax = plt.subplots(figsize=(12, 7))
sns.set_style("white")
sns.set_context("poster")

p_Cum_Summary = pd.read_csv('p_Cum_Summary.csv')
p_Cum_Summary_long = pd.melt(p_Cum_Summary, id_vars=['Iteration'],  value_vars=['p_Hits','p_TF','p_MF'])
ax = sns.violinplot("Iteration", hue="variable", y="value", data=p_Cum_Summary_long, palette=['red','blue','green']);                
Random_Cum_Summary = pd.read_csv('Random_Cum_Summary.csv')
Random_Cum_Summary_long = pd.melt(Random_Cum_Summary, id_vars=['Iteration'],  value_vars=['Random_Hits','Random_TF','Random_MF'])
ax = sns.violinplot("Iteration", hue="variable", y="value", data=Random_Cum_Summary_long, palette=['indianred','lightsteelblue','lightgreen']);
           
ax.set_title('HHRT1', fontweight='bold', fontsize=25)
ax.set_ylabel("Percentage of Test Set [%]", fontweight='bold', fontsize=22)
ax.set_xlabel("Iteration",fontsize=22, fontweight='bold')
ax.tick_params(labelsize=16)

plt.legend(loc=4, prop={'size': 11, 'weight':'bold'})
Fig_CumSum_Hits.tight_layout()

Fig_CumSum_Hits.savefig('Cumulative_Sums_Hits.eps', format='eps', dpi=1200)
Fig_CumSum_Hits.savefig('Cumulative_Sums_Hits.svg', format='svg', dpi=1200)
#

#Remove unnecesary files
os.remove("calScores.txt")
os.remove("testScores.txt")
os.remove("Calibration")
os.remove("p_Predict")
os.remove("Random_Predict")
os.remove("p_Training")
os.remove("Random_Training")