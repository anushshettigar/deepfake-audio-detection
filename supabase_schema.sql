-- DeepShield Supabase Schema
-- Run this in Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (extend existing)
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS email TEXT UNIQUE;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Analysis sessions (for live detection)
CREATE TABLE IF NOT EXISTS public.analysis_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    source TEXT NOT NULL CHECK (source IN ('upload', 'extension_live', 'web_live')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'error')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    total_chunks INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audio records (individual files/chunks)
CREATE TABLE IF NOT EXISTS public.audio_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES public.analysis_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    chunk_number INTEGER,
    original_filename TEXT,
    storage_path TEXT NOT NULL,
    storage_bucket TEXT NOT NULL DEFAULT 'audio-files',
    file_size_bytes BIGINT,
    format TEXT,
    duration_seconds NUMERIC,
    sample_rate INTEGER,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'error')),
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    error_message TEXT
);

-- Detection results
CREATE TABLE IF NOT EXISTS public.detection_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audio_record_id UUID REFERENCES public.audio_records(id) ON DELETE CASCADE,
    session_id UUID REFERENCES public.analysis_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    label TEXT NOT NULL CHECK (label IN ('REAL', 'DEEPFAKE')),
    class_id INTEGER NOT NULL CHECK (class_id IN (0, 1)),
    confidence NUMERIC NOT NULL,
    fake_probability NUMERIC NOT NULL,
    real_probability NUMERIC NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'RawNet2-PhonemeDF2026',
    threshold NUMERIC DEFAULT 0.5,
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model versions tracking
CREATE TABLE IF NOT EXISTS public.model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version TEXT NOT NULL UNIQUE,
    architecture TEXT NOT NULL,
    dataset TEXT NOT NULL,
    labels JSONB NOT NULL,
    threshold NUMERIC DEFAULT 0.5,
    sample_rate INTEGER,
    audio_seconds INTEGER,
    num_samples INTEGER,
    training_config JSONB,
    preprocessing_config JSONB,
    metrics JSONB,
    checkpoint_path TEXT,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert current model version
INSERT INTO public.model_versions (
    version, architecture, dataset, labels, threshold, sample_rate, audio_seconds, num_samples,
    training_config, preprocessing_config, metrics, checkpoint_path, is_active
) VALUES (
    '1.0.0',
    'RawNet2',
    'PhonemeDF 2026',
    '{"0": "FAKE", "1": "REAL"}'::jsonb,
    0.5,
    16000,
    4,
    64000,
    '{"epochs": 30, "batch_size": 16, "learning_rate": 0.0001, "weight_decay": 0.0001, "optimizer": "AdamW", "scheduler": "ReduceLROnPlateau", "early_stopping_patience": 7, "mixed_precision": true, "seed": 42, "class_weights": [0.625, 2.5]}'::jsonb,
    '{"sample_rate": 16000, "n_mels": 64, "n_fft": 512, "hop_length": 160}'::jsonb,
    '{"accuracy": 0.9536, "roc_auc": 0.9879, "real_precision": 0.8333, "real_recall": 0.96, "real_f1": 0.8922, "fake_precision": 0.9896, "fake_recall": 0.952, "fake_f1": 0.9704, "confusion_matrix": {"tn": 476, "fp": 24, "fn": 5, "tp": 120}}'::jsonb,
    'backend/models/DeepShield_Deployment/models/DeepShield_RawNet2.pt',
    TRUE
) ON CONFLICT (version) DO NOTHING;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_audio_records_session_id ON public.audio_records(session_id);
CREATE INDEX IF NOT EXISTS idx_audio_records_user_id ON public.audio_records(user_id);
CREATE INDEX IF NOT EXISTS idx_audio_records_status ON public.audio_records(status);
CREATE INDEX IF NOT EXISTS idx_detection_results_audio_record_id ON public.detection_results(audio_record_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_session_id ON public.detection_results(session_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_user_id ON public.detection_results(user_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_label ON public.detection_results(label);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_user_id ON public.analysis_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_status ON public.analysis_sessions(status);

-- RLS Policies
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audio_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.detection_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;

-- Users: users can read their own data
CREATE POLICY "Users can view own data" ON public.users
    FOR SELECT USING (auth.uid() = id);

-- Analysis sessions: users can CRUD their own sessions
CREATE POLICY "Users can manage own sessions" ON public.analysis_sessions
    FOR ALL USING (auth.uid() = user_id);

-- Audio records: users can CRUD their own records
CREATE POLICY "Users can manage own audio records" ON public.audio_records
    FOR ALL USING (auth.uid() = user_id);

-- Detection results: users can read their own results
CREATE POLICY "Users can view own detection results" ON public.detection_results
    FOR SELECT USING (auth.uid() = user_id);

-- Model versions: public read for active model
CREATE POLICY "Public can view active model version" ON public.model_versions
    FOR SELECT USING (is_active = TRUE);

-- Service role can do everything (for backend)
-- Note: Service role key bypasses RLS entirely