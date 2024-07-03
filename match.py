import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import datetime as dt
import configparser
import os
import warnings

warnings.filterwarnings('ignore')

if not os.path.isdir('Results') :
    os.mkdir("Results")
  
print('Lecture du fichier de paramètre...')
config = configparser.ConfigParser()
config.read('config.ini')
config.sections()

## initialisation des parametres
Liste_score = []
for i in config.options('WEIGHTS_SCORE'):
    Liste_score.append(int(config['WEIGHTS_SCORE'][i]))
    
code_start_liste = []
for i in config.options('START_CODE'):
    code_start_liste.append(int(config['START_CODE'][i]))
print('Terminé')
print('Lecture des fichiers de données...')
comma = 0
semicolon = 0
try :
    data_list_comma= []
    for i in config.options('NOM_FICHIERS'):
        data_list_comma.append(pd.read_csv(config['NOM_FICHIERS'][i], dtype=object,sep=','))
    comma = data_list_comma[0].shape[1]
except: 
    print('Erreur dans la lecture des fichiers avec la virgule, utilisation du point-virgule comme séparateur')
    
try:
    data_list_semic= []
    for i in config.options('NOM_FICHIERS'):
        data_list_semic.append(pd.read_csv(config['NOM_FICHIERS'][i], dtype=object,sep=';'))
    semicolon = data_list_semic[0].shape[1]
except:
    print('Erreur dans la lecture des fichiers avec le point-virgule, utilisation de la virgule comme séparateur')
    
if comma > semicolon :
    data_list = data_list_comma
else :
    data_list = data_list_semic
        
print('Terminé')
num_var = int(config['CODE_LENGTH']['CODE_LENGTH'])
CUT_OFF_SIMILARITY = int(config['PARAMETRES']['CUT_OFF_SCORE'])
MATCH_T2 = bool(config['PARAMETRES']['MATCH_T2'])
if config['PARAMETRES']['MATCH_T2'] == 'TRUE' :
    MATCH_T2  = True
elif config['PARAMETRES']['MATCH_T2'] == 'FALSE' :
    MATCH_T2  = False
    
## nom temps de mesure
temps_de_mesure = []
for i in config.options('T'):
    temps_de_mesure.append(config['T'][i])

## type cast all code variables to string
for i in range(len(data_list)):
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var] = data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].astype(str)

## upper case all code variables if alpha
for i in range(len(data_list)):
    for j in range(code_start_liste[i],code_start_liste[i]+num_var):
        try:
            data_list[i].iloc[:,j] = data_list[i].iloc[:,j].str.upper()
        except:
            pass


for i in range(0,len(data_list)):
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("É","E",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("È","E",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("Ê","E",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("À","A",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("Ï","I",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("Û","U",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("Ô","O",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("Ç","C",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("Ë","E",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("Ù","U",regex=True,inplace=True)
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].replace("Î","I",regex=True,inplace=True)
    

#ajouter sufixes 
for i in range(len(data_list)) :
    cols = data_list[i].columns
    new_cols = []
    for col in cols:
        if temps_de_mesure[i] not in col and temps_de_mesure[i].lower() not in col :
            if col == 'IDI':
                pass
            else :
                col = col+'-'+temps_de_mesure[i]

        new_cols.append(col)
    data_list[i].columns = new_cols


## deteminer un score pour nsp

NSP_score = [ i * 0.5 for i in Liste_score]


## attribuer des id aux participants
total_id = 0
for i in range(len(data_list)):
    data_list[i][f'ID_{temps_de_mesure[i]}'] = np.arange(total_id,total_id+data_list[i].shape[0])
    total_id += data_list[i].shape[0]
    

## faire un dictionnaire de template pour chaque participant selon son code


n_inver = 0

def find_match(data_list,code_start_liste,num_var,Liste_score,NSP_score, cut_off_similarity, temps_de_mesure):
    '''
    fonction pour trouver les matchs entre les temps de mesure
    :param data_list: liste de dataframe de temps de mesure
    :param code_start_liste: liste des index de debut de code dans chaque dataframe
    :param num_var: nombre de variables dans le code
    :param Liste_score: liste des poids pour chaque variable
    :param NSP_score: liste des poids pour chaque variable NSP
    :param cut_off_similarity: seuil de similarité pour le matching
    :return: dictionnaire de match { ID : [[ID_matchT1, scoreT1],[ID_matchT2,scoreT2]...]}
    '''

    template_dict = {}
    
    for i in range(data_list[0].shape[0]):
        template_dict[data_list[0].loc[i,f'ID_{temps_de_mesure[0]}']] = data_list[0].iloc[i,code_start_liste[0]:code_start_liste[0]+num_var].to_numpy()
    
    liste_code = []
    for i in range(1,len(data_list)):
        code_temp = data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].to_numpy()
        code_temp = np.c_[code_temp, data_list[i][f'ID_{temps_de_mesure[i]}'].to_numpy() ]
        liste_code.append(code_temp)
   
        
    matching_dict = { i : [] for i in template_dict.keys()}

    
    ## loop par dessus les temps de mesure autres que le premier 
    for i in range(0,len(liste_code)):
        print('Temps de mesure : ',temps_de_mesure[i+1])
        
        ## loop par dessus les participants du temps de mesure 1 comme template
        for j in template_dict.keys():
            score_vec = np.zeros(liste_code[i].shape[0])
            score_vec_rever = np.zeros(liste_code[i].shape[0])

            template = template_dict[j]
            
            
            ## loop par dessus les vartiables du code
            
            for k in range(len(template)):
                
                variable = template[k]
                
            
                vect_to_check  = (liste_code[i][:,k] == variable).astype(int) * Liste_score[k]
                
                ## donner la motié des points si la variable est NSP
                NSP_vec = (liste_code[i][:,k] == '?').astype(int) * NSP_score[k]
                vect_to_check = vect_to_check + NSP_vec
            
            
                score_vec = score_vec + vect_to_check
            
                
            ## regarder si il y a inversion de la variables 2 et 3 (nom, prenom)
            if np.max(score_vec) <= 100 - Liste_score[2] and np.max(score_vec) >= 50:
                
                    
                rever_template = template.copy()
                rever_template[2] = template[3]
                rever_template[3] = template[2]
                    
                ## loop par dessus les variables du code
                for k in range(len(rever_template)):
                    
                    variable = rever_template[k]
                    
                    vect_to_check  = (liste_code[i][:,k] == variable).astype(int) * Liste_score[k]
                    NSP_vec = (liste_code[i][:,k] == '?').astype(int) * NSP_score[k]
                    vect_to_check = vect_to_check + NSP_vec
                
                
                    score_vec_rever = score_vec_rever + vect_to_check
                
                    
                
            if np.max(score_vec_rever) >= np.max(score_vec) + Liste_score[2] *2 :
                
                score = [liste_code[i][np.argmax(score_vec_rever),-1], np.max(score_vec_rever)]

                    
            else:
                score = [liste_code[i][np.argmax(score_vec),-1], np.max(score_vec)]
                
            
            matching_dict[j].append(score)
    
    return matching_dict
print('Matching...')


matching_dict = find_match(data_list,code_start_liste=code_start_liste,num_var=num_var,Liste_score=Liste_score,NSP_score=NSP_score,cut_off_similarity=CUT_OFF_SIMILARITY,temps_de_mesure=temps_de_mesure)
print('Terminé')

total = np.zeros(3)

for i in matching_dict:
   total += np.array(matching_dict[i])[:,1] > CUT_OFF_SIMILARITY

## ajouter un nouveau id intemporel pour les match
for index,i in enumerate(data_list):
    match_score_name = f'match_score-{temps_de_mesure[index]}'
    i[match_score_name] = 0
    i['IDI'] = -1
## IDI = ID intemporel pour le T1
data_list[0]['IDI'] =  data_list[0][f'ID_{temps_de_mesure[0]}']

for i in matching_dict:
    matchs = matching_dict[i]
    
    for j in range(len(matchs)):
        match_score_name = f'match_score-{temps_de_mesure[j+1]}'
        if matchs[j][1] >= CUT_OFF_SIMILARITY:
            
            
            index = data_list[j+1].index[data_list[j+1][f'ID_{temps_de_mesure[j+1]}'] == matchs[j][0] ].values[0]
            data_list[j+1].loc[index,'IDI'] = i
            data_list[j+1].loc[index,match_score_name] = matchs[j][1]
            

if MATCH_T2 :
    
    print(f'Refaire le matching à partir de {temps_de_mesure[1]}...')
    #########  REFAIRE MATCH A PARTIR DE T2 ############
    # nouvelle liste de data sans le t1 pour refaire le match à partir de T2
    new_data_list = data_list[1:].copy()
    #get non match du t2 à tx
    new_temps_de_mesure = temps_de_mesure[1:]

    # keep only non match (IDI == -1)
    for i, data in enumerate(new_data_list):
        new_data_list[i] = new_data_list[i][new_data_list[i]['IDI'] == -1].reset_index(drop=True)

    # nouveau parametre de matching
    code_start_liste = code_start_liste[1:]
    

    matching_dict = find_match(new_data_list,code_start_liste=code_start_liste,num_var=num_var,Liste_score=Liste_score,NSP_score=NSP_score,cut_off_similarity=CUT_OFF_SIMILARITY,temps_de_mesure=new_temps_de_mesure) 



    ## IDI = ID intemporel pour le T1
    new_data_list[0]['IDI'] =  new_data_list[0][f'ID_{new_temps_de_mesure[0]}']

    for i in matching_dict:
        matchs = matching_dict[i]
        
        for j in range(len(matchs)):
            match_score_name = f'match_score-{new_temps_de_mesure[j+1]}'
            if matchs[j][1] >= CUT_OFF_SIMILARITY:
                
                
                index = new_data_list[j+1].index[new_data_list[j+1][f'ID_{new_temps_de_mesure[j+1]}'] == matchs[j][0] ].values[0]
                new_data_list[j+1].loc[index,'IDI'] = i
                new_data_list[j+1].loc[index,match_score_name] = matchs[j][1]
    
                

    ### mettre les IDI
    for i in range(len(data_list)):
        data_list[i] = data_list[i][data_list[i]['IDI'] != -1].reset_index(drop=True)
        
        

        
    ## matrice des match à partir de t2
    new_match_point = []
    for i in range(len(new_data_list[0])):
        template = new_data_list[0].loc[i,'IDI']
        combin = []
        for j in range(1,len(new_data_list)):
            test = new_data_list[j][ new_data_list[j]['IDI'] == template]
            combin.append(len(test))
        new_match_point.append(combin)

    test = []
    for i in new_match_point:
        template = temps_de_mesure[1]
        for j in range(len(i)) :
            if i[j] ==1:
                template = template + f'-{temps_de_mesure[j+1]}'
        test.append(template)
        
    new_match_point = np.array(new_match_point)

    new_number_of_each = Counter(test)
    new_number_of_each.pop(temps_de_mesure[1])

    plt.bar(range(len(new_number_of_each)), list(new_number_of_each.values()), align='center')
    plt.xticks(range(len(new_number_of_each)), list(new_number_of_each.keys()))
    plt.xticks(rotation = 25)
    plt.title(f"Nombre de participants ayant un match à partir de {temps_de_mesure[1]}")
    plt.xlabel("Temps de mesure")
    plt.ylabel("Nombre de participants")

    plt.savefig('Results/T2_match_distr.png')

    ## merge les dataframes
    # faire un fichier texte ayant les IDI de tous les all match
    # prendre seulement ceux pour qui on a un match à partir de t2 jusqu'à tN
    all_match = np.sum(new_match_point,axis=1) == 2
    II = new_data_list[0][all_match]
    II['IDI'].to_csv('Results/all_match.txt',index=False,header=False,sep='\t')
    #II['StartDate_T2'] = pd.to_datetime(II['StartDate_T2'], format='%m/%d/%Y %H:%M:%S')
    #II[(II['StartDate_T2'] >= dt.datetime(2021, 3, 9)) & (II['StartDate_T2'] <= dt.datetime(2021, 3, 16))]['IDI'].to_csv('Results/all_match_21.txt',index=False,header=False,sep='\t')
    liste_of_idi = II['IDI'].tolist()

    for j in range(0,len(new_data_list)):
        new_data_list[j] = new_data_list[j][new_data_list[j]['IDI'].isin(liste_of_idi)].reset_index(drop=True)
            
        
    
        

    final_df = []
    for i in range(len(new_data_list)):
        final_df.append(pd.concat([data_list[i+1], new_data_list[i]]))

    data_list[1:] = final_df

    for i in range(len(data_list)):
        data_list[i] = data_list[i].reset_index(drop=True)
        
        
    # ajouter les nouveaux match à partir de t2 dans le t1
    fake_df = pd.DataFrame(columns=data_list[0].columns)
    fake_df.loc[0] = data_list[0].loc[0]
    fake_df = fake_df.loc[fake_df.index.repeat(len(new_data_list[0]))]
    fake_df[f'ID_{temps_de_mesure[0]}'] = np.arange(0,len(new_data_list[0])) + 1000000
    fake_df['IDI'] = new_data_list[0]['IDI'].to_numpy()
    
    data_list[0] = pd.concat([data_list[0],fake_df]).reset_index(drop=True)
    print('Terminé')
   
    ############### FIN REFAIRE MATCH A PARTIR DE T2 ############



## exporter les dataframes avec les IDI
print('Exporter les résultats...')
for i in range(len(data_list)):
    data_list[i].to_csv(f'Results/PRESAJ_{temps_de_mesure[i]}_IDI.csv',index=False)

print('Terminé')

## merge dataframes
print('Merge des dataframes...')

final_df = data_list[0].copy()


for i in range(1,len(data_list)):
    

    final_df = pd.merge(final_df,data_list[i][ data_list[i]['IDI'] != -1],how='outer',on='IDI')
  

  
final_df.fillna(999,inplace=True)

## merge les scores de matching

for T in temps_de_mesure[1:]:
    new_col = np.zeros_like(final_df.iloc[:,0])
  
    name = [ i for i in final_df.columns.tolist() if f'match_score-{T}' in i]
    if len(name) > 1:
        select  = final_df[name].to_numpy() != 999
        i_select  = select.argmax(axis=1)
        new_col = final_df[name].to_numpy()[np.arange(len(final_df)),i_select]

        ## replace les valeurs de match_score-Tx_x par match_score-Tx svec new_col
        final_df[f'match_score-{T}'] = new_col
        print(T)
        final_df.drop(name,axis=1,inplace=True)
        
        
            
        
        
    




final_df.to_csv('Results/PRESAJ_all_merged.csv',index=False,encoding='utf-8-sig')
    
#final_df['match_score-tT3']

## generer les histogrammes des scores de matching


## matrice des match à partir de t1
print('Générer les graphiques...')
match_point = []
for i in range(len(data_list[0])):
    template = data_list[0].loc[i,'IDI']
    combin = []
    for j in range(1,len(data_list)):
       test = data_list[j][ data_list[j]['IDI'] == template]
       
       combin.append(len(test))
    match_point.append(combin)


number_of_each = []
for i in match_point:
    template = 'T1'
    for j in range(len(i)) :
        if i[j] ==1:
            template = template + f'-T{j+2}'
    number_of_each.append(template)
    
match_point = np.array(match_point)
number_of_each = Counter(number_of_each)


number_of_each.pop('T1')

plt.clf()
plt.bar(range(len(number_of_each)), list(number_of_each.values()), align='center',color='orange')
plt.xticks(range(len(number_of_each)), list(number_of_each.keys()))
plt.xticks(rotation = 25)
plt.title("Nombre de participants ayant un match à partir de T1")
plt.xlabel("Temps de mesure")
plt.ylabel("Nombre de participants")


plt.savefig('Results/T1_match_distr.png')



## faire un fichier texte ayant les code de tous les match et leur score
idi = data_list[0]['IDI'].to_numpy()
num_var
 
code_start_liste = []
for i in config.options('START_CODE'):
    code_start_liste.append(int(config['START_CODE'][i]))

## matrice de donné la longeur du nome de row et 2 x nombre de temps de mesure
all_code = np.zeros((len(idi),2*len(temps_de_mesure)+1))
all_code = all_code.astype(str)
all_code.shape
num_t = len(temps_de_mesure)
len(data_list)

for idx, id in enumerate(idi) :
    for i in range(len(data_list)) :
        test = ' '
        score = 0
        try:
            test = data_list[i].loc[data_list[i]['IDI'] == id].iloc[0,code_start_liste[i]:code_start_liste[i]+num_var].to_list()
            test = ''.join(test)
            if i >0 :
                score = data_list[i].loc[data_list[i]['IDI'] == id][f'match_score-{temps_de_mesure[i]}'].values[0]
            if len(score) == 0:
                score = 0
        except:
            pass
       
        all_code[idx,i ] = test
    
        
        all_code[idx,i + num_t] = str(score)
        all_code[idx,-1] = str(id)
        
        


df = pd.DataFrame(all_code)

df.columns = [f'Code_{temps_de_mesure[i]}' for i in range(len(temps_de_mesure))] + [f'Score_{temps_de_mesure[i]}' for i in range(len(temps_de_mesure))] + ['IDI']


df.to_csv('Results/code_match.csv',index=False,encoding='utf-8-sig',sep='\t')

print('Fin du programme')
print ('TOTAL MATCH : ',total)






## figure pour avoir une idéé de la distribution des scores de matching
'''
test_data = pd.DataFrame.from_dict(matching_dict, orient='index')


plt.hist([i[1] for i in test_data[0] ] )
plt.show()

for i in range(0,3):
    color = ['blue','orange','green','red']
        
    plt.hist([i[1] for i in test_data[i].to_numpy()],color= color[i])

    plt.title(f"Histogramme des scores de matching T1-T{i+2}")
    plt.xlabel("Score")
    plt.ylabel("Nombre de participants")
    plt.savefig(f"histo_T1_T{i+1}.png")
    plt.clf()

data_list= []
for i in range(4):
    data_list.append(pd.read_csv(f'presaj_data/PRESAJ_T{i+1}_IDI.csv'))

## pour voir sans t2
#data_list.pop(1)



## histo des nombre de match par participant du t1
num_match = np.zeros((data_list[0].shape[0]))
for i in range(data_list[0].shape[0]):
    id_to_test = data_list[0].loc[i,'IDI']
    sum_match = 0
    for j in range(1,len(data_list)):
       sum_match = sum_match + sum(data_list[j]['IDI'] == id_to_test)
    num_match[i] = sum_match
    
plt.hist(num_match)
plt.show()

sum(num_match)
len(num_match)


'''

'''

data1 = pd.read_csv('presaj_data/PRESAJT1.csv')
data2 = pd.read_csv('presaj_data/PRESAJT2.csv')
data_list = [data1,data2]
code_start_liste = [11,18]
temps_de_mesure = ['T1','T2']
num_var = 11


#ajouter sufixes 
for i in range(len(data_list)) :
    cols = data_list[i].columns
    new_cols = []
    for col in cols:
        if temps_de_mesure[i] not in col and temps_de_mesure[i].lower() not in col :
            if col == 'IDI':
                pass
            else :
                col = col+'-'+temps_de_mesure[i]

        new_cols.append(col)
    data_list[i].columns = new_cols


## deteminer un score pour nsp




## attribuer des id aux participants
total_id = 0
for i in range(len(data_list)):
    data_list[i][f'ID_{temps_de_mesure[i]}'] = np.arange(total_id,total_id+data_list[i].shape[0])
    total_id += data_list[i].shape[0]
    
    

## type cast all code variables to string
for i in range(len(data_list)):
    data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var] = data_list[i].iloc[:,code_start_liste[i]:code_start_liste[i]+num_var].astype(str)

## upper case all code variables if alpha
for i in range(len(data_list)):
    for j in range(code_start_liste[i],code_start_liste[i]+num_var):
        try:
            data_list[i].iloc[:,j] = data_list[i].iloc[:,j].str.upper()
        except:
            pass


data_list[0].iloc[2,code_start_liste[0]:code_start_liste[0]+num_var].to_list()
data_list[1].iloc[2,code_start_liste[1]:code_start_liste[1]+num_var].to_list()


'''  
    
            
            
            
    
            
            
            
            
        
        
    
    
    


    

    
    







