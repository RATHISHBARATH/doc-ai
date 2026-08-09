-- Migration: Create oauth_states table
-- Description: Stores OAuth2 state parameters for CSRF protection

CREATE TABLE IF NOT EXISTS oauth_states (
    -- Primary key: the state string itself (randomly generated)
    state TEXT PRIMARY KEY,

    -- Optional user ID (if the state is tied to an existing user)
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Timestamp when the state was created (used for expiration)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Expiration timestamp (states are short-lived)
    expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '5 minutes'
);

-- Index for cleaning up expired states (optional)
CREATE INDEX idx_oauth_states_expires_at ON oauth_states (expires_at);