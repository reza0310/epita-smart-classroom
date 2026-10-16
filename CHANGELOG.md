# Journal des modifications
## Correctifs OWL-ViT et mode scènes denses
- Le pipeline OWL-ViT utilise explicitement Transformers.js `zero-shot-object-detection`, le backend navigateur WASM et les poids q8.
- Les détecteurs Transformers.js reçoivent directement un `HTMLCanvasElement`, une entrée prise en charge par la bibliothèque, sans dépendre d'une construction `RawImage` faite dans l'application.
- Le chargement OWL-ViT commence dès l'activation de l'option sur image fixe afin d'afficher immédiatement l'état et la progression.
- Les images envoyées à OWL-ViT et YOLOv10n sont réduites à 1 280 pixels sur leur plus grand côté ; les boîtes sont remappées sur la résolution source.
- Vite exclut `@huggingface/transformers` du pre-bundling et construit les workers au format ES.
- La détection OWL-ViT est disponible en image fixe et en vidéo à cadence réduite ; YOLOv10n et les découpes COCO-SSD renforcent le mode scènes denses.
- Les erreurs de modèle affichent maintenant le détail technique et permettent une nouvelle tentative.

## Ajustement d'interface minimal sur f4e02ac
- Ajout d'un onglet « Apprentissage » distinct de la supervision et du rapport.
- Regroupement de « Configuration initiale » et « Objets en vue » dans un panneau de comptage commun.
- La configuration Vite et les scripts de build restent ceux du commit f4e02ac.
