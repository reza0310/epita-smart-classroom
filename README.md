# Atlas — Audit visuel des équipements

Application React + Vite, conçue pour Cloudflare Pages. Les détections COCO-SSD sont réalisées dans le navigateur. Les images de la caméra ne sont ni téléversées ni enregistrées par l'application.

## Démarrer en local

Prérequis : Node.js 20+ et npm.

```bash
npm install
npm run dev
```

Ouvrir l'URL affichée par Vite. La caméra nécessite un contexte sécurisé : `localhost` en local ou HTTPS une fois déployé. Au premier lancement, le navigateur télécharge les bibliothèques et les poids du modèle COCO-SSD ; les images de la caméra, elles, restent locales.

Pour tester le build de production :

```bash
npm run build
npm run preview
```

## Déploiement Cloudflare Pages

1. Envoyer ce projet dans un dépôt Git (GitHub ou GitLab).
2. Dans Cloudflare, créer un projet Pages et connecter le dépôt.
3. Paramètres de build : **Build command** `npm run build` ; **Build output directory** `dist`.
4. Déployer. L'accès à la caméra doit être accordé par l'utilisateur et fonctionne en HTTPS.

## Détections et limites

COCO-SSD renvoie des boîtes, classes et scores pour les classes du jeu COCO. Cette interface ne conserve que les sorties `chair`, `dining table`, `laptop` et `tv`, présentées sous les noms Chaise, Table, Ordinateur portable et Écran. Un bureau peut ne pas être reconnu comme `dining table`, et `tv` ne permet pas de garantir que l'objet est un écran d'ordinateur.

Les fenêtres, tableaux et anomalies ne font pas partie des catégories COCO-SSD utilisables ici : ils sont relevés manuellement. Pour une détection automatique réelle, il faut entraîner/évaluer un modèle spécialisé sur des images représentatives et des annotations de ces classes. Faster R-CNN n'est pas proposé ici car la version COCO-SSD est conçue pour être légère dans le navigateur.

**Confidentialité et personnes :** l'interface ne présente, n'enregistre et n'exporte aucune sortie de classe `person`, et ne réalise aucune identification. Toutefois, COCO-SSD est un modèle générique multi-classes : filtrer sa sortie ne garantit pas l'absence de calcul interne relatif à une personne présente dans l'image. Pour respecter strictement une politique de non-analyse des personnes, cadrer physiquement la caméra sur le mobilier, suspendre l'analyse dès qu'une personne entre dans le champ, et remplacer COCO-SSD par un modèle spécialisé limité aux classes d'équipement avant un usage où cette garantie est impérative.

## Apprentissage / renforcement

Cette version utilise un modèle pré-entraîné : il ne se ré-entraîne pas et n'apprend pas par renforcement. Le rapport et la configuration permettent une supervision humaine, mais les corrections ne sont pas envoyées à un serveur et ne modifient pas le modèle. Un véritable apprentissage supervisé demanderait un jeu d'images annotées et un pipeline d'entraînement ; l'apprentissage par renforcement ne consiste pas simplement à ajuster un seuil ou à donner un avis utilisateur.

## Références

- COCO-SSD : https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd
- Configuration de build Cloudflare Pages : https://developers.cloudflare.com/pages/configuration/build-configuration/
