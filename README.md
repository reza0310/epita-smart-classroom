# Atlas — Audit visuel des équipements

Application React + Vite conçue pour Cloudflare Pages. Elle peut analyser le flux caméra/vidéo ou une image fixe sélectionnée par l'utilisateur. La détection COCO-SSD et le dessin des cadres sont effectués dans le navigateur.

## Fonctionnalités

- Caméra en direct avec détection des catégories d'équipement couvertes par COCO-SSD.
- Import et analyse d'une image fixe JPEG, PNG ou WebP.
- Affichage de cadres, catégorie, score de confiance et comptage par catégorie.
- Seuil de confiance réglable.
- Détection zéro-shot OWL-ViT en option sur image fixe **et dans le flux vidéo** (cadence réduite, environ une analyse toutes les 6,5 secondes) pour proposer des cadres de tableaux, fenêtres et tables/bureaux.
- Mode « scène dense » avec YOLOv10n en complément de COCO-SSD ; sur image fixe, COCO-SSD analyse aussi quatre zones chevauchantes afin de mieux repérer les petits objets.
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

Ouvrir l'URL affichée par Vite. L'accès caméra nécessite un contexte sécurisé : `localhost` en local ou HTTPS une fois déployé. Au premier lancement, le navigateur télécharge les bibliothèques et les poids COCO-SSD. OWL-ViT et YOLOv10n sont chargés à la demande. Les requêtes vers les poids des deux modèles passent désormais par `/api/hf/` sur la même origine : Vite fait le relais en développement, et une Cloudflare Pages Function le fait en production. Les fichiers ONNX Runtime/WASM sont copiés depuis `node_modules` vers `public/ort/` au démarrage et au build, évitant le chargement du runtime WASM depuis un CDN tiers. Une connexion est nécessaire au premier téléchargement ; le navigateur peut ensuite conserver les ressources en cache. Les images caméra ne sont ni transmises au relais ni envoyées vers un serveur par l'application.

Pour tester le build de production :

```bash
npm run build
npm run preview
```

## Déploiement Cloudflare Pages (configuration recommandée)

Ce projet est une application **Cloudflare Pages + Vite + Pages Functions**. Il ne doit pas être déployé comme un Worker avec `wrangler deploy` : cette commande utilise la configuration des Workers Assets et peut provoquer l’erreur `assets.directory .../dist does not exist` lorsque le build Pages n’a pas été exécuté.

Dans **Workers & Pages → Create application → Pages → Connect to Git**, connectez le dépôt puis définissez :

- **Framework preset** : Vite (ou None).
- **Root directory** : `/` (racine du dépôt).
- **Build command** : `npm run build`.
- **Build output directory** : `dist`.
- **Deploy command** : aucun champ de déploiement personnalisé — Pages publie automatiquement le résultat du build.

Laissez Pages exécuter son flux de build/publication. Ne configurez ni `wrangler deploy`, ni un champ `assets.directory`, ni une commande de déploiement Worker. Les fichiers `functions/api/hf/[[path]].js` doivent rester dans le dépôt à côté de `src/` et `package.json` : Pages les déploie comme Function pour le relais des modèles.

Le script `prebuild` copie les fichiers ONNX Runtime WASM depuis les dépendances installées vers `public/ort/` avant l’exécution de Vite. Le build doit donc créer `dist/index.html`. Pour tester localement :

```bash
npm install
npm run build
npm run preview
```

Si le projet actuellement configuré dans le tableau de bord est un **Worker** plutôt qu’un projet **Pages**, créez un nouveau projet Pages connecté au même dépôt. Une configuration de Worker ne devient pas une configuration Pages simplement en changeant le nom du dossier de sortie.

## Classes détectées et limites

COCO-SSD renvoie des boîtes, classes et scores pour les classes COCO. Cette application ne conserve que les sorties `chair`, `dining table`, `laptop` et `tv`, présentées comme Chaises, Tables & bureaux, Ordinateurs portables et Écrans. COCO-SSD peut manquer un bureau et `tv` ne garantit pas qu'un objet soit un écran d'ordinateur.

Les fenêtres et tableaux ne sont pas des classes natives de COCO-SSD. Activez OWL-ViT pour une détection zéro-shot complémentaire : les libellés proposés couvrent les tableaux blancs/noirs, les fenêtres et plusieurs descriptions de bureaux/tables. Cela fonctionne sur image fixe ou, en vidéo, à une cadence réduite et en tâche complémentaire pour ne pas bloquer la détection COCO à chaque image. Les cadres OWL-ViT sont fusionnés avec les détections standards avec suppression de doublons fortement chevauchants. Les scores OWL-ViT ont un seuil distinct. Les anomalies restent à évaluer manuellement. Les propositions OWL-ViT peuvent être erronées et doivent être confirmées. Ces résultats peuvent être encadrés manuellement et sauvegardés comme exemples d’apprentissage. Ces annotations corrigent le jeu de données, mais ne changent pas les poids du modèle déjà chargé. Le détecteur ne s'améliore réellement qu'après entraînement/évaluation d'un modèle spécialisé. Les instructions et scripts Faster R-CNN sont dans `training/`.

## OWL-ViT sur image fixe et vidéo

OWL-ViT (`Xenova/owlvit-base-patch32`, Transformers.js) est un détecteur zéro-shot : l’application lui passe des descriptions textuelles préétablies plutôt qu’un nouveau modèle entraîné. Il est chargé à la demande. Le seuil OWL-ViT est distinct du seuil COCO-SSD. L’option active immédiatement le chargement du modèle ; le premier chargement télécharge les poids ONNX quantifiés depuis Hugging Face (environ 155 Mo pour q8) et requiert une connexion. Le pipeline reçoit directement le canvas HTML et laisse Transformers.js gérer la conversion de l’image. Le backend est explicitement configuré sur WASM en quantification q8, le cache navigateur est activé et l’exécution WASM est configurée en mono-thread pour limiter les incompatibilités sur certains appareils. La configuration Vite exclut Transformers.js du pre-bundling et laisse ses ressources WASM/worker en modules ES. En mode vidéo, OWL-ViT s’exécute au maximum toutes les 6,5 secondes et en tâche complémentaire ; COCO-SSD continue de rafraîchir les cadres entre les exécutions OWL-ViT. Les boîtes OWL-ViT sont conservées jusqu’à la détection complémentaire suivante : elles peuvent donc avoir quelques secondes de retard dans une scène mobile.

L’erreur du moteur est affichée avec son détail technique et une action « Réessayer ». Les téléchargements de poids passent par `/api/hf/` sur la même origine, et ONNX Runtime WASM est servi depuis `/ort/` après la copie automatique à l'étape `predev`/`prebuild`. Si OWL-ViT ou YOLO affiche une erreur réseau, ouvrez `/api/hf/_health` sur le même domaine : `ok: true` valide le relais Hugging Face ; 404 indique souvent une Pages Function absente ; 502 signifie que le relais n'a pas pu joindre Hugging Face. Vérifiez aussi que les fichiers `/ort/ort-wasm-*.wasm` et `/ort/ort-wasm-*.mjs` répondent sans 404 dans l'onglet Réseau du navigateur. Les fichiers q8 OWL-ViT pèsent environ 155 Mo, auxquels s’ajoute la mémoire de travail : certains téléphones d’entrée de gamme peuvent échouer par manque de mémoire. Les images envoyées aux détecteurs complémentaires sont réduites à 1 280 pixels sur le grand côté, puis les cadres sont remappés sur la résolution d’origine pour limiter ces risques. OWL-ViT n’est pas un système d’apprentissage manuel : les corrections restent des annotations de jeu de données pour une éventuelle phase d’entraînement séparée. Il ne faut pas interpréter un score faible ou l’absence de cadre comme une preuve que l’objet est absent. Recadrez les personnes hors de l’image avant analyse : comme le modèle encode l’image entière, n’analysez pas de photos contenant des personnes si la contrainte est l’absence totale d’analyse des personnes.

## Mode scènes denses : YOLOv10n + COCO-SSD par zones

Le mode « Scène dense » ajoute `onnx-community/yolov10n` (pipeline `object-detection` de Transformers.js) aux détections COCO-SSD pour les classes COCO suivies : chaise, table de repas, ordinateur portable et téléviseur/écran. Le modèle est téléchargé à la demande ; le fichier ONNX quantifié q8 est d’environ 2,65 Mo, auquel s’ajoutent les métadonnées et l’environnement d’exécution. Sur image fixe, COCO-SSD est relancé sur quatre zones avec recouvrement afin d’agrandir les petits objets dans l’entrée du modèle. Les résultats des deux modèles et des tuiles sont fusionnés par suppression des cadres de même catégorie qui se chevauchent fortement. En vidéo, YOLOv10n est appelé environ toutes les 4,5 secondes en tâche complémentaire ; l’image affichée entre les exécutions utilise les derniers résultats disponibles.

**Licence du modèle :** la fiche `onnx-community/yolov10n` indique AGPL-3.0. Avant d’intégrer ce modèle dans une distribution commerciale ou propriétaire, vérifiez les obligations de cette licence et envisagez un modèle dont la licence est compatible avec votre usage. YOLOv10n reste un détecteur COCO généraliste ; il n’améliore pas la reconnaissance native des fenêtres ou tableaux.

## Apprentissage avec corrections humaines et Faster R-CNN

1. Importer une image et activer **Dessiner des cadres**.
2. Sélectionner une catégorie et encadrer chaque objet visible. Annoter en priorité les tables/bureaux manqués, tableaux et fenêtres.
3. Enregistrer l'exemple, puis exporter le jeu d'apprentissage. L'export JSON contient les images en base64 et les annotations.
4. Suivre [`training/README.md`](training/README.md) pour extraire le jeu COCO, générer les TFRecords et fine-tuner un Faster R-CNN pré-entraîné.

**Faster R-CNN n'est pas chargé automatiquement dans l'interface.** Le pipeline prépare et entraîne un modèle dans un environnement TensorFlow séparé. La conversion TFJS d'un Faster R-CNN doit être validée avec les versions/outils utilisés et un adaptateur d'inférence adapté au graphe exporté avant de le rendre sélectionnable dans l'application. COCO-SSD reste le moteur d'inférence de la version actuelle.

## Confidentialité et personnes

Ne pas importer de photos contenant des personnes. Cadrer ou recadrer l'image pour conserver uniquement le mobilier et les équipements. L'interface ne présente ni ne conserve les sorties de classe `person` et n'effectue aucune identification ; toutefois, COCO-SSD, OWL-ViT et YOLOv10n sont des modèles génériques capables d'analyser des contenus au-delà des équipements ciblés. Le seul filtrage des sorties ne garantit donc pas l'absence de calcul interne relatif à une personne dans l'image. Pour un usage où l'absence d'analyse des personnes est une exigence impérative, remplacer ce modèle par un détecteur spécialisé sur les seules classes d'équipement avant déploiement.

Les images annotées résident dans IndexedDB sur l'appareil. L'export inclut les images : il est volumineux et potentiellement sensible. L'utilisateur doit choisir explicitement quand l'exporter et où le stocker.

## Références

- COCO-SSD : <https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd>
- OWL-ViT pour Transformers.js : <https://huggingface.co/Xenova/owlvit-base-patch32>
- YOLOv10n pour Transformers.js : <https://huggingface.co/onnx-community/yolov10n>
- Transformers.js pipelines : <https://huggingface.co/docs/transformers.js/en/api/pipelines>
- TensorFlow.js — conversion de modèles : <https://www.tensorflow.org/js/guide/conversion>
- TensorFlow Object Detection API : <https://github.com/tensorflow/models/tree/master/research/object_detection>
- Faster R-CNN Model Zoo : <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>
- Configuration Cloudflare Pages : <https://developers.cloudflare.com/pages/configuration/build-configuration/>

## Diagnostic réseau OWL-ViT / YOLO

- Tester le relais en production : `/api/hf/_health`.
- Les routes du relais n'acceptent que les dépôts publics `Xenova/owlvit-base-patch32` et `onnx-community/yolov10n`, en lecture seule (`GET`/`HEAD`) ; elles ne constituent pas un proxy arbitraire.
- Le relais ne reçoit pas d'images : seuls les fichiers publics des modèles sont récupérés depuis Hugging Face.
- Vérifier que `npm run predev` ou `npm run prebuild` a copié les fichiers ONNX Runtime dans `public/ort/`.
- Sur Cloudflare Pages via Git, conservez le dossier `functions/` à la racine du dépôt : Pages le déploie automatiquement avec le site.

## Journal des modifications

Voir [`CHANGELOG.md`](CHANGELOG.md) pour les correctifs et les vérifications de cette version.


## Interface update — organisation simplifiée

- Navigation séparée entre **Supervision**, **Apprentissage** et **Rapport d’audit**.
- L’apprentissage manuel dispose de son propre onglet et conserve l’import d’image, le dessin des cadres, les exemples enregistrés ainsi que l’import/export du jeu de données.
- Le panneau **Configuration initiale & objets en vue** regroupe les quantités attendues, les relevés manuels, les comptages issus des modèles et la liste des détections individuelles.
- Les indicateurs généraux sont masqués dans l’onglet d’apprentissage pour réduire les distractions.
- Les contrôles s’adaptent aux écrans étroits ; les valeurs non relevées restent représentées par un tiret et ne sont pas converties automatiquement en absence.
