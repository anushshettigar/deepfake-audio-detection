-- DeepShield Supabase Migration (NEW project)
-- Run in: https://supabase.com/dashboard/project/aannvqyrjjzimosdvhoh/sql
-- Or: SUPABASE_PAT=<token> python run_migration.py

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Analysis sessions (upload analyses + live detection sessions)
CREATE TABLE IF NOT EXISTS public.analysis_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    source TEXT NOT NULL CHECK (source IN ('upload', 'extension_live', 'web_live')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'error')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    total_chunks INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audio records (uploaded files / live chunks)
CREATE TABLE IF NOT EXISTS public.audio_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES public.analysis_sessions(id) ON DELETE CASCADE,
    user_id UUID,
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

-- Detection results (0 = FAKE, 1 = REAL)
CREATE TABLE IF NOT EXISTS public.detection_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audio_record_id UUID REFERENCES public.audio_records(id) ON DELETE CASCADE,
    session_id UUID REFERENCES public.analysis_sessions(id) ON DELETE CASCADE,
    user_id UUID,
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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_detection_results_created_at ON public.detection_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detection_results_session_id ON public.detection_results(session_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_audio_record_id ON public.detection_results(audio_record_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_label ON public.detection_results(label);
CREATE INDEX IF NOT EXISTS idx_detection_results_user_id ON public.detection_results(user_id);
CREATE INDEX IF NOT EXISTS idx_audio_records_session_id ON public.audio_records(session_id);
CREATE INDEX IF NOT EXISTS idx_audio_records_user_id ON public.audio_records(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_user_id ON public.analysis_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_status ON public.analysis_sessions(status);

-- RLS (backend uses secret key -> bypasses RLS; users/anon get scoped access)
ALTER TABLE public.analysis_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audio_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.detection_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own sessions" ON public.analysis_sessions
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own audio records" ON public.audio_records
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own detection results" ON public.detection_results
    FOR SELECT USING (auth.uid() = user_id);
