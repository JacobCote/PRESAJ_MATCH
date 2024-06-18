################### PROGRAMME DE MATCH #####################
### Utilisation 
On doit configurer les paramètres pour le match dans le fichier config.ini


## paramètres
[NOM_FICHIERS]
donne le chemin d'accès de tous les fichiers pour le match de données
ex: 
FICHIER1: presaj_data/PRESAJ_T1(n=3056).csv
[T]
Suffixe a mettre pour les différents temps de mesure
ex:
Fichier1 : T1


[CODE_LENGTH]
le nombre de colonnes(variable) que le code a.
ex :
CODE_LENGTH: 11

[START_CODE]
Donne le l'indice de la colonne ou le code commence. Doit être indiqué pour chaq'un
des temps de mesure même si ils commencent tous au même indice.
ex:
START1: 11

[WEIGHTS_SCORE]
Poids pour chaque variables dans le code pour le match. 
La somme des poids doit égalé à 100. Toute les variables du code doivent avoir un poids
ex:
SCORE_MATCH1: 9

[PARAMETRES]
Le CUT_OFF_SCORE est le seuile pour qu'un match soit accepté.
Ce score est calculé en fonction des poids
Le MATCH_T2 permet de refaire un match à partir du temp de mesure 2 pour récupérer le 
plus de match possible. Seulement les match complet seront gardé afin de ne pas introduire
de doublons dans le jeu de données.
ex :
CUT_OFF_SCORE: 80
MATCH_T2: TRUE

Si vous avez des questions, vous pouvez communiquer avec moi :
jacobcote@hotmail.com


