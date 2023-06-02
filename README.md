# OC-P7
## Implémentez un modèle de scoring
Ce dossier contient l'ensemble du code permettant de déployer le dashboard interactif permettant l'affichage du score d'un client à partir de son identifiant.

### Description du projet
Mise en oeuvre d'un outil de "scoring credit" pour calculer la probabilité qu'un client rembourse son crédit.
Cet outil est composé de 3 parties :
- un notebook présentant l'analyse exploratoire, le feature engineering, l'entraînement de différents modèles de catégorisation. Ces modèles sont enregistrés avec MLFlow.
- Le modèle sélectionné est mis en production grâce à FastAPI.
- Le dashboard interactif permet l'affichage du score et des informations descriptives à partir de l'ID du client en interrogeant l'API du modèle.

### Découpage des dossiers
1. Dossier complet : https://github.com/Kromette/OC-P7
2. Dossier du modèle : https://github.com/Kromette/OC-P7-Model
3. Dossier du dashboard : https://github.com/Kromette/OC-P7-Dashboard

### Comment installer/éxécuter le projet
Le déploiement sur Heroku est automatique depuis un push sur la branche main du repo github.

### Comment utiliser le projet
L'API est déployée à l'adresse suivante : https://modelfastapi.herokuapp.com/
La documentation liée à l'API est disponible à cette adresse : https://modelfastapi.herokuapp.com/docs

L'interface est déployée à l'adresse suivante : https://dashboardstreamlit.herokuapp.com/
1. Entrer un identifiant client dans le champs "ID du client" 
2. cliquer sur "Obtenir le score" pour afficher le score, les features importances globales puis locales et enfin les analyses univariées puis bivariées.
3. Choisir parmi les 10 features qui présentent la meilleur feature importance pour afficher les analyses uni et bivariées