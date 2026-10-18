# Deepfake Audio Detection System

A comprehensive deepfake audio detection platform with backend API, React frontend, and browser extension for real-time detection.

## Architecture

```
├── backend/                 # FastAPI server
│   ├── main.py             # API endpoints
│   ├── storage.py          # Supabase integration
│   ├── services/           # Detection services
│   └── train.py            # Model training
├── frontend/               # React + Vite + Supabase Auth
│   ├── src/                # Components, pages, context
│   └── public/             # Static assets
├── extension/              # Chrome/Edge browser extension
│   ├── background/         # Service worker
│   ├── content/            # Content scripts
│   ├── capture/            # Audio capture
│   └── popup/              # Extension UI
├── deepfake_audio_preprocessing.py   # Feature extraction
├── ensemble_model.py       # Ensemble classifier
├── spectral_classifier.py  # Spectral analysis model
├── train_*.py              # Training scripts (RawNet2, Spectral, Production)
├── supabase_schema.sql     # Database schema
└── supabase_migration.sql  # Migrations
```

## Features

- **Multi-model detection**: RawNet2, Spectral classifier, Ensemble
- **Real-time analysis**: Browser extension captures & analyzes live audio
- **Web dashboard**: Upload, history, dataset management
- **Supabase backend**: Auth, database, storage, realtime
- **Model training**: Production-ready training pipelines

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Supabase account

### Backend
```bash
cd backend
pip install -r requirements.txt  # Create requirements.txt from imports
cp .env.example .env  # Add your Supabase credentials
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env  # Add VITE_SUPABASE_URL & VITE_SUPABASE_ANON_KEY
npm run dev
```

### Extension
1. Open `chrome://extensions/`
2. Enable Developer mode
3. Load unpacked → select `extension/`

## Models

| Model | Type | Use Case |
|-------|------|----------|
| RawNet2 | End-to-end CNN | General deepfake detection |
| Spectral | Hand-crafted features | Interpretability, low-resource |
| Ensemble | Weighted voting | Best accuracy |

## Dataset

Uses `dataset/manifest.jsonl` with entries:
```json
{"audio_path": "path/to/file.wav", "label": "real|fake", "source": "WaveFake|ElevenLabs|..."}
```

## Environment Variables

### Backend (`.env`)
```
SUPABASE_URL=your-project-url
SUPABASE_SERVICE_KEY=service-role-key
SUPABASE_JWKS_URL=your-jwks-url
RATE_LIMIT_REQUESTS=300
RATE_LIMIT_WINDOW=60
```

### Frontend (`.env`)
```
VITE_SUPABASE_URL=your-project-url
VITE_SUPABASE_ANON_KEY=anon-public-key
```

## Training

```bash
# RawNet2 production training
python train_production.py

# Spectral classifier
python train_spectral.py

# Ensemble
python train_model.py
```

## Deployment

- **Backend**: Render, Railway, Fly.io, or any container platform
- **Frontend**: Vercel, Netlify, Cloudflare Pages
- **Database**: Supabase (PostgreSQL + Auth + Storage)
- **Extension**: Chrome Web Store / Edge Add-ons

## Security

- All API keys in `.env` (never committed)
- Supabase RLS policies on all tables
- Rate limiting on detection endpoints
- JWT verification via JWKS

## License

MIT