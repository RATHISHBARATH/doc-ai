-- Migration: Create refresh_tokens table
-- Description: Stores refresh tokens for JWT renewal

-- Ensure pgcrypto is available (idempotent — already enabled by the
-- users migration, but safe to repeat here for standalone reliability).
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS refresh_tokens (
    -- Primary identifier (UUID)
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- User who owns this refresh token
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Bcrypt hash of the refresh token (never store plaintext)
    token_hash TEXT NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,  -- NULL if not revoked

    -- Revocation flag (for quick filtering)
    revoked BOOLEAN NOT NULL DEFAULT FALSE,

    -- Device or client identifier (optional, for audit)
    client_info TEXT
);

-- Indexes for fast lookups
CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens (user_id);
CREATE INDEX idx_refresh_tokens_expires_at ON refresh_tokens (expires_at);
CREATE INDEX idx_refresh_tokens_revoked ON refresh_tokens (revoked);