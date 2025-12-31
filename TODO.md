# 🧠 OCR API — TODO

## Plan concret (priorites)
- [ ] Corriger incoherences IO: filenames liste, schema `name`, logs de fichiers
- [ ] Rendre `resolve_geometry` robuste + corriger flag `_crop_orientation_disabled`
- [ ] Fiabiliser `/health` (detection des models onnx par prefixe)
- [ ] Factoriser la logique OCR/Read dans `predictor.py` (reduire duplication)
- [ ] Ajouter tests: `get_documents`, validateurs dates/NIR/IBAN, multi-page
- [ ] Ameliorer logs: request id, timings par etape, erreurs uniformes
- [ ] Mettre a jour `readme.md` (params, multi-doc, print_type)

## 🚀 Core
- [x] API FastAPI fonctionnelle
- [x] Endpoint `/health`
- [x] Endpoint `/ocr`
- [x] Ajout du champ `duration` dans les réponses
- [x] Header `X-Execution-Time`
- [ ] Rules : meilleur nettoyage de la date
- [ ] Rules : les spécialités fixe pour rechercher soit pqr regex ou gliner
- [ ] Ajouter la lecteur de table

## 🧠 OCR / Models
- [x] Initialisation lazy des modèles
- [ ] Cache des modèles par type de requête
- [ ] Endpoint `/health/models`
- [ ] Forcer le device CPU / GPU via config
- [ ] Support batch multi-documents

## 📦 API / Schema
- [x] BaseResponse avec `duration`
- [x] OCROut hérite de BaseResponse
- [x] Versionner les réponses (`v1`)
- [ ] Ajouter un bloc `meta`

## 🔐 Sécurité
- [ ] Auth API key
- [ ] Limitation de débit (rate limit)
- [ ] Désactiver swagger en prod

## 🪟 Windows / Déploiement
- [x] Lancement via Uvicorn
- [ ] Service Windows via NSSM
- [ ] Script `install-service.bat`
- [ ] Script `uninstall-service.bat`
- [ ] Redémarrage auto + logs rotation

## 🧪 Tests
- [x] Tests unitaires OCR v0.0
- [ ] Test endpoint `/health`
- [ ] Test endpoint `/ocr`
- [ ] Test fichier PDF

## 📊 Monitoring
- [ ] Logs structurés (json)
- [ ] Endpoint `/metrics`
- [ ] Temps moyen OCR
- [ ] Compteur d’erreurs
