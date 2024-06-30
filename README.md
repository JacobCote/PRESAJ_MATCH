



# PROGRAMME DE MATCH 
## Utilisation 
On doit configurer les paramètres pour le match dans le fichier config.ini

### Paramètres
[NOM_FICHIERS]

donne le chemin d'accès de tous les fichiers pour le match de données

ex: 

FICHIER1: presaj_data/PRESAJ_T1(n=3056).csv


[T]

Suffixe a mettre pour les différents temps de mesure

ex:

Fichier1: T1

Fichier2: T2
...

[CODE_LENGTH]

le nombre de colonnes(variable) que le code a.

ex :

CODE_LENGTH: 11

[START_CODE]

Donne le l'indice de la colonne ou le code commence. Doit être indiqué pour chaq'un
des temps de mesure même si ils commencent tous au même indice.

ex:

START1: 11

START2: 18
...

[WEIGHTS_SCORE]

Poids pour chaque variables dans le code pour le match. 
La somme des poids doit égalé à 100. Toutes les variables du code doivent avoir un poids

ex:

SCORE_MATCH1: 9

SCORE_MATCH2: 9 
...

[PARAMETRES]

Le CUT_OFF_SCORE est le seuile pour qu'un match soit accepté.
Ce score est calculé en fonction des poids
Le MATCH_T2 permet de refaire un match à partir du temp de mesure 2 pour récupérer le 
plus de match possible. Seulement les match complet seront gardé afin de ne pas introduire
de doublons dans le jeu de données.

ex :

CUT_OFF_SCORE: 80

MATCH_T2: TRUE


# Étapes pour utiliser le programme 
## Windows
### Installation (Seulement faire la première fois)
1. Installer python, suivre les [étapes](https://www.digitalocean.com/community/tutorials/install-python-windows-10)
2. Installer Git, suivre les [étapes](https://www.git-scm.com/download/win)
    (pour savoir si 32-bit ou 64-bit cliquez [ici](https://support.microsoft.com/en-us/windows/32-bit-and-64-bit-windows-frequently-asked-questions-c6ca9541-8dce-4d48-0415-94a3faa2e13d)) 
3. Ouvrir l'application Git Bash que l'on vient d'installer sur l'ordinateur, une fenêtre de terminal apparaîtra.
4. Copier coller la commandes suivant et faire la touche Enter :
``` bash
git clone https://github.com/JacobCote/PRESAJ_MATCH.git
```
5. Copier coller la commandes suivante et faire la touche Enter :
``` bash
cd PRESAJ_MATCH
```
6. Copier coller la commandes suivante et faire la touche Enter :
``` bash
./setupWin.sh
```

7. Si tout s'est bien déroulé, en entrant la commandes suivante :
``` bash
ls
```
On devrait voir apparaître des fichiers et aussi les dossier `data/` et `venv/`

8. On doir maintenant ajouter les données dans le dossier `data/`. Pour ce faire, entrer la commande suivante pour ouvrir le dossier du programme (le point est important) et glissez les données dans le dossier `data/`
``` bash
explorer . 
```

### Lancer le programme 
1. Ouvrir l'application Git Bash.

2. Copier coller la commandes suivante et faire la touche Enter :
``` bash
cd PRESAJ_MATCH
```
3. Copier coller la commandes suivante et faire la touche Enter :
``` bash
explorer .
```
ouvrir le fichier config.ini et s'assurer que la configuration est ok

3. Copier coller la commandes suivante dans Git Bash et faire la touche Enter :
``` bash
./runMatch.py
```
Le programme de match vient d'être lancé, il devrait prendre au maximum quelques minutes a s'exécuté.

Les résultats devraient être dans le dosser `Results`

Pour y accéder, ouvrir une nouvelle fenêtre de Git Bash et entrer les commandes suivantes:

1. Copier coller la commandes suivante et faire la touche Enter :
``` bash
cd PRESAJ_MATCH
```
2. Copier coller la commandes suivante et faire la touche Enter :
``` bash
explorer .
```

Le dossier `Results` devrait y être.


Si vous avez des questions, vous pouvez communiquer avec moi :
jacobcote@hotmail.com


