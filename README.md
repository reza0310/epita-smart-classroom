# Atlas — Audit visuel des équipements

Application React + Vite conçue pour Cloudflare Pages. Elle peut analyser le flux caméra/vidéo ou une image fixe sélectionnée par l'utilisateur. La détection COCO-SSD et le dessin des cadres sont effectués dans le navigateur.

## Fonctionnalités

- Caméra en direct avec détection des catégories d'équipement couvertes par COCO-SSD.
- Import et analyse d'une image fixe JPEG, PNG ou WebP.
- Affichage de cadres, catégorie, score de confiance et comptage par catégorie.
- Seuil de confiance réglable.
- Détection zéro-shot OWL-ViT en option sur image fixe pour proposer des cadres de tableaux, fenêtres et tables/bureaux.
- Configuration de quantités attendues, relevés manuels et rapport exportable/imprimable.
- Annotation manuelle de boîtes pour chaises, tables/bureaux, ordinateurs portables, écrans, tableaux, fenêtres et anomalies.
- Stockage local des exemples annotés dans IndexedDB ; import/export des exemples avec images intégrées.
- Scripts pour préparer les annotations COCO et des TFRecords, puis fine-tuner un modèle Faster R-CNN avec TensorFlow Object Detection API (voir `training/README.md`).

## Démarrer en local

Prérequis : Node.js 20+ et npm.

```bash
npm install
npm run dev
```

Ouvrir l'URL affichée par Vite. L'accès caméra nécessite un contexte sécurisé : `localhost` en local ou HTTPS une fois déployé. Au premier lancement, le navigateur télécharge les bibliothèques et les poids COCO-SSD. OWL-ViT est chargé seulement à la demande sur une image fixe ; les poids quantifiés représentent environ 155 Mo, selon le modèle distribué, puis le navigateur peut en conserver une copie en cache. Les images caméra ne sont ni téléversées ni enregistrées par l'application.

Pour tester le build de production :

```bash
npm run build
npm run preview
```

## Déploiement Cloudflare Pages

1. Envoyer ce projet dans un dépôt Git.
2. Créer un projet Pages et connecter le dépôt.
3. Commande de build : `npm run build` ; répertoire de sortie : `dist`.
4. Déployer. Autoriser explicitement la caméra si on choisit le mode caméra.

## Classes détectées et limites

COCO-SSD renvoie des boîtes, classes et scores pour les classes COCO. Cette application ne conserve que les sorties `chair`, `dining table`, `laptop` et `tv`, présentées comme Chaises, Tables & bureaux, Ordinateurs portables et Écrans. COCO-SSD peut manquer un bureau et `tv` ne garantit pas qu'un objet soit un écran d'ordinateur.

Les fenêtres et tableaux ne sont pas des classes natives de COCO-SSD. Sur image fixe, activez l’option OWL-ViT pour une détection zéro-shot complémentaire : les libellés proposés couvrent les tableaux blancs/noirs, les fenêtres et plusieurs descriptions de bureaux/tables. Les cadres OWL-ViT sont fusionnés avec COCO-SSD, avec suppression de doublons fortement chevauchants. Les scores OWL-ViT ont un seuil distinct. Les anomalies restent à évaluer manuellement. Les propositions OWL-ViT peuvent être erronées et doivent être confirmées. Ces résultats peuvent être encadrés manuellement et sauvegardés comme exemples d’apprentissage. Ces annotations corrigent le jeu de données, mais ne changent pas les poids du modèle déjà chargé. Le détecteur ne s'améliore réellement qu'après entraînement/évaluation d'un modèle spécialisé. Les instructions et scripts Faster R-CNN sont dans `training/`.

## OWL-ViT sur image fixe

OWL-ViT (`Xenova/owlvit-base-patch32`, Transformers.js) est un détecteur zéro-shot : l’application lui passe des descriptions textuelles préétablies plutôt qu’un nouveau modèle entraîné. Il est proposé uniquement pour les images fixes afin de préserver les performances de la caméra sur mobile. Le seuil OWL-ViT est distinct du seuil COCO-SSD. Le premier chargement télécharge les poids ONNX quantifiés depuis Hugging Face ; il requiert une connexion, et l’inférence est ensuite effectuée dans le navigateur. Les cadres sont fusionnés avec les résultats COCO-SSD et les doublons qui se chevauchent fortement sont supprimés.

OWL-ViT n’est pas un système d’apprentissage manuel : les corrections restent des annotations de jeu de données pour une éventuelle phase d’entraînement séparée. Il ne faut pas interpréter un score faible ou l’absence de cadre comme une preuve que l’objet est absent. Recadrez les personnes hors de l’image avant analyse : comme le modèle encode l’image entière, n’envoyez pas de photos contenant des personnes si la contrainte est l’absence totale d’analyse des personnes.

## Apprentissage avec corrections humaines et Faster R-CNN

1. Importer une image et activer **Dessiner des cadres**.
2. Sélectionner une catégorie et encadrer chaque objet visible. Annoter en priorité les tables/bureaux manqués, tableaux et fenêtres.
3. Enregistrer l'exemple, puis exporter le jeu d'apprentissage. L'export JSON contient les images en base64 et les annotations.
4. Suivre [`training/README.md`](training/README.md) pour extraire le jeu COCO, générer les TFRecords et fine-tuner un Faster R-CNN pré-entraîné.

**Faster R-CNN n'est pas chargé automatiquement dans l'interface.** Le pipeline prépare et entraîne un modèle dans un environnement TensorFlow séparé. La conversion TFJS d'un Faster R-CNN doit être validée avec les versions/outils utilisés et un adaptateur d'inférence adapté au graphe exporté avant de le rendre sélectionnable dans l'application. COCO-SSD reste le moteur d'inférence de la version actuelle.

## Confidentialité et personnes

Ne pas importer de photos contenant des personnes. Cadrer ou recadrer l'image pour conserver uniquement le mobilier et les équipements. L'interface ne présente ni ne conserve les sorties de classe `person` et n'effectue aucune identification ; toutefois, COCO-SSD reste un modèle générique multi-classes, de sorte que le seul filtrage des sorties ne garantit pas l'absence de calcul interne relatif à une personne dans l'image. Pour un usage où l'absence d'analyse des personnes est une exigence impérative, remplacer ce modèle par un détecteur spécialisé sur les seules classes d'équipement avant déploiement.

Les images annotées résident dans IndexedDB sur l'appareil. L'export inclut les images : il est volumineux et potentiellement sensible. L'utilisateur doit choisir explicitement quand l'exporter et où le stocker.

## Références

- COCO-SSD : <https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd>
- OWL-ViT pour Transformers.js : <https://huggingface.co/Xenova/owlvit-base-patch32>
- Transformers.js pipelines : <https://huggingface.co/docs/transformers.js/en/api/pipelines>
- TensorFlow.js — conversion de modèles : <https://www.tensorflow.org/js/guide/conversion>
- TensorFlow Object Detection API : <https://github.com/tensorflow/models/tree/master/research/object_detection>
- Faster R-CNN Model Zoo : <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>
- Configuration Cloudflare Pages : <https://developers.cloudflare.com/pages/configuration/build-configuration/>
