# Pipeline d'apprentissage manuel + Faster R-CNN

Cette partie du dépôt transforme les annotations réalisées dans Atlas en données d'entraînement pour TensorFlow Object Detection API. **Le navigateur collecte et exporte des annotations ; il ne réentraîne pas lui-même COCO-SSD et le modèle Faster R-CNN n'est pas automatiquement chargé par l'interface.**

## 1. Produire un jeu d'exemples dans l'application

1. Ouvrir l'onglet **Image téléchargée** et importer une photo cadrée sur le mobilier/les équipements.
2. Dans **Apprendre à partir de corrections humaines**, sélectionner une classe et activer **Dessiner des cadres**.
3. Faire glisser le doigt ou la souris autour de chaque chaise, table/bureau, ordinateur portable, écran, tableau, fenêtre ou anomalie visible.
4. Cliquer sur **Enregistrer l'exemple**. Les exemples sont conservés dans IndexedDB du navigateur.
5. Exporter régulièrement le jeu de données. L'export JSON embarque les images et les cadres, afin de former un fichier autonome.

Priorité à la qualité : ajouter des angles, éclairages et distances variés ; annoter les bureaux que COCO-SSD manque ; inclure aussi les tableaux et fenêtres ; corriger soigneusement les bordures des cadres. Éviter de créer plusieurs annotations identiques du même objet. Un jeu de quelques images sert à prototyper, mais n'établit pas la robustesse du détecteur.

## 2. Extraire les images annotées et générer les TFRecords

Utiliser un environnement Python séparé (idéalement GPU) avec TensorFlow et les dépendances de la TensorFlow Object Detection API installées conformément au guide officiel :

- Installation : <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md>
- Modèles pré-entraînés : <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>

Depuis la racine du dépôt :

```bash
python training/prepare_coco.py chemin/atlas-jeu-apprentissage.json ./training/dataset
python training/write_tfrecords.py ./training/dataset
```

Cette étape produit `training/dataset/images/{train,val}`, les annotations COCO correspondantes, `label_map.pbtxt`, puis `train.record` et `val.record`. Avec une seule image, il n'est pas possible de réaliser une véritable validation séparée ; avec peu d'images, les métriques resteront très instables.

## 3. Fine-tuning Faster R-CNN (TensorFlow 2 Object Detection API)

Télécharger et extraire un modèle **Faster R-CNN** compatible TF2 depuis le Model Zoo TensorFlow. Le dossier doit contenir le `pipeline.config` et le checkpoint pré-entraîné. Le processus Python doit trouver le package `object_detection` : si nécessaire, ajouter les répertoires `research` et `research/slim` du clone TensorFlow Models à `PYTHONPATH`. Ensuite configurer le pipeline en remplaçant le chemin du modèle de base et ceux des TFRecords :

```bash
export TF_MODELS_DIR=/chemin/vers/tensorflow/models
export PYTHONPATH="$TF_MODELS_DIR/research:$TF_MODELS_DIR/research/slim:$PYTHONPATH"
python training/configure_pipeline.py \
  --base-config /chemin/modele-faster-rcnn/pipeline.config \
  --checkpoint /chemin/modele-faster-rcnn/checkpoint/ckpt-0 \
  --train-record ./training/dataset/train.record \
  --val-record ./training/dataset/val.record \
  --label-map ./training/dataset/label_map.pbtxt \
  --output ./training/pipeline.config \
  --num-classes 7 \
  --batch-size 2
```

Puis lancer l'entraînement depuis `models/research` après installation de l'API :

```bash
python object_detection/model_main_tf2.py \
  --pipeline_config_path=/chemin/vers/atlas/training/pipeline.config \
  --model_dir=/chemin/vers/atlas/training/checkpoints \
  --num_train_steps=5000 \
  --alsologtostderr
```

Le nombre d'étapes et la taille de lot ci-dessus sont des points de départ, pas des paramètres optimaux. Ajuster selon le volume du jeu de données et la mémoire GPU. Garder un jeu de validation séparé et mesurer précision/rappel et faux positifs avant tout usage d'audit.

## 4. Export du modèle entraîné

Après évaluation, exporter un SavedModel TF2 avec l'utilitaire `exporter_main_v2.py` de la version de TensorFlow Models installée. La conversion TensorFlow SavedModel → TensorFlow.js GraphModel peut ensuite être tentée avec `tensorflowjs_converter`, conformément au guide de conversion officiel : <https://www.tensorflow.org/js/guide/conversion>.

**Important :** la conversion n'est pas garantie pour chaque version/configuration Faster R-CNN. Les opérations du graphe, la signature d'entrée et les noms/formats des sorties de détection doivent être vérifiés. Atlas utilise actuellement l'adaptateur COCO-SSD ; il faudra ajouter un adaptateur d'inférence pour le SavedModel converti (prétraitement, sorties `boxes/scores/classes`, normalisation des coordonnées) avant de sélectionner le modèle Faster R-CNN dans l'interface. Ne déployez pas un modèle sans tests de non-régression et de confidentialité.

## 5. Confidentialité et limites

- Les images restent dans IndexedDB tant qu'elles ne sont pas explicitement exportées par l'utilisateur.
- Les exports incluent les images en base64 ; ils peuvent être volumineux et sensibles. Les stocker et les partager avec précaution.
- Ne pas annoter ni entraîner à partir de photos contenant des personnes. Recadrer les photos avant import. Le modèle COCO-SSD utilisé actuellement est multi-classes, donc le filtrage des résultats ne constitue pas une garantie de non-analyse interne d'une personne présente dans l'image.
- Des annotations erronées apprennent des exemples erronés. Chaque classe doit être définie de façon cohérente et les performances évaluées sur des images indépendantes.
