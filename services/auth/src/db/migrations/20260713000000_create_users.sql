-- Migration: Create users table
-- Description: Stores user accounts for authentication

-- Required for gen_random_uuid() on PostgreSQL < 13.
-- (PostgreSQL 13+ has gen_random_uuid() built into core, so this is a
-- no-op on newer versions, but safe to keep for compatibility.)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    -- Primary identifier (UUID)
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Email address (unique)
    email TEXT NOT NULL UNIQUE,

    -- Display name
    name TEXT NOT NULL,

    -- Authentication provider (e.g., 'google', 'github', 'local')
    provider TEXT NOT NULL,

    -- Provider-specific user ID (e.g., Google sub, GitHub id)
    provider_id TEXT,

    -- Password hash (only for 'local' provider; NULL for OAuth2)
    password_hash TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_provider_provider_id ON users (provider, provider_id);