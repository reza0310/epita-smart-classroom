# Journal des modifications

## Correctifs OWL-ViT et mode scènes denses

- Le pipeline OWL-ViT utilise explicitement Transformers.js `zero-shot-object-detection`, le backend navigateur WASM et les poids q8.
- Les détecteurs Transformers.js reçoivent directement un `HTMLCanvasElement`, une entrée prise en charge par la bibliothèque, sans dépendre d'une construction `RawImage` faite dans l'application.
- Le chargement OWL-ViT commence dès l'activation de l'option sur image fixe afin d'afficher immédiatement l'état et la progression.
- Les images envoyées à OWL-ViT et YOLOv10n sont réduites à 1 280 pixels sur leur plus grand côté ; les boîtes sont remappées sur la résolution source.
- Vite exclut `@huggingface/transformers` du pre-bundling et construit les workers au format ES.
- La détection OWL-ViT est disponible en image fixe et en vidéo à cadence réduite ; YOLOv10n et les découpes COCO-SSD renforcent le mode scènes denses.
- Les erreurs de modèle affichent maintenant le détail technique et permettent une nouvelle tentative.

## Vérifications

- Parsing JSX et transpilation syntaxique réalisés sans diagnostic.
- Intégrité de l'archive ZIP vérifiée.
- La compilation complète `npm run build` et un test navigateur de bout en bout n'ont pas pu être réalisés dans l'environnement de génération, car `npm install` a dépassé le délai disponible. Tester sur un navigateur récent et une connexion permettant le téléchargement des modèles avant le déploiement.
