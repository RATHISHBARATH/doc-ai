// ============================================================
// Auth Service – JWT and Refresh Token Module
// ============================================================

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};
use uuid::Uuid;

use crate::db::DbPool;

// ============================================================
// JWT Claims
// ============================================================

/// JWT claims structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID).
    pub sub: String,
    /// Issued at (timestamp).
    pub iat: i64,
    /// Expiration (timestamp).
    pub exp: i64,
    /// Issuer.
    pub iss: String,
}

impl Claims {
    /// Creates a new Claims struct for the given user ID.
    pub fn new(user_id: &str) -> Self {
        let now = Utc::now();
        let iat = now.timestamp();
        let exp = (now + Duration::hours(1)).timestamp();
        Self {
            sub: user_id.to_string(),
            iat,
            exp,
            iss: "doc-ai-auth".to_string(),
        }
    }
}

// ============================================================
// JWT Creation and Validation
// ============================================================

/// Creates a JWT for the given user ID.
pub fn create_jwt(user_id: &str, secret: &str) -> Result<String> {
    let claims = Claims::new(user_id);
    let encoding_key = EncodingKey::from_secret(secret.as_bytes());
    encode(&Header::default(), &claims, &encoding_key)
        .context("Failed to encode JWT")
}

/// Validates a JWT and returns the user ID if valid.
pub fn validate_jwt(token: &str, secret: &str) -> Result<String> {
    let decoding_key = DecodingKey::from_secret(secret.as_bytes());
    let validation = Validation::default();
    let token_data = decode::<Claims>(token, &decoding_key, &validation)
        .context("Failed to decode/validate JWT")?;
    Ok(token_data.claims.sub)
}

// ============================================================
// Refresh Token Management
// ============================================================

/// Creates a new refresh token for the user and stores it in the database.
#[instrument(skip(pool))]
pub async fn create_refresh_token(user_id: &str, pool: &DbPool) -> Result<String> {
    // Generate a random token string.
    let token = Uuid::new_v4().to_string();

    // Hash the token using bcrypt.
    let hash = bcrypt::hash(&token, bcrypt::DEFAULT_COST)
        .context("Failed to hash refresh token")?;

    // Insert into database. expires_at is a DateTime<Utc>, matching the
    // TIMESTAMPTZ column type in the refresh_tokens table.
    let expires_at = Utc::now() + Duration::days(30);
    sqlx::query!(
        r#"
        INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
        VALUES ($1, $2, $3)
        "#,
        uuid::Uuid::parse_str(user_id).context("Invalid user ID")?,
        hash,
        expires_at
    )
    .execute(pool)
    .await
    .context("Failed to store refresh token")?;

    info!("Refresh token created for user {}", user_id);
    Ok(token)
}

/// Validates a refresh token and returns the user ID if valid.
#[instrument(skip(pool))]
pub async fn validate_refresh_token(token: &str, pool: &DbPool) -> Result<String> {
    // Find the token by its hash (we don't store plaintext).
    // Query all active refresh tokens (simplified for Phase 1).
    // This is O(N) per validation – acceptable for low-volume deployments.
    let rows = sqlx::query!(
        r#"
        SELECT id, user_id, token_hash, expires_at, revoked
        FROM refresh_tokens
        WHERE expires_at > NOW() AND revoked = FALSE
        "#
    )
    .fetch_all(pool)
    .await
    .context("Failed to query refresh tokens")?;

    for row in rows {
        if bcrypt::verify(token, &row.token_hash).unwrap_or(false) {
            // Token is valid and not revoked. Update last_used_at.
            sqlx::query!(
                "UPDATE refresh_tokens SET last_used_at = NOW() WHERE id = $1",
                row.id
            )
            .execute(pool)
            .await
            .context("Failed to update last_used_at")?;
            return Ok(row.user_id.to_string());
        }
    }

    Err(anyhow::anyhow!("Invalid or expired refresh token"))
}

/// Revokes a refresh token (used during logout).
#[instrument(skip(pool))]
pub async fn revoke_refresh_token(token: &str, pool: &DbPool) -> Result<()> {
    // Find the token and mark it as revoked (same scan strategy as above).
    let rows = sqlx::query!(
        r#"
        SELECT id, token_hash, expires_at, revoked
        FROM refresh_tokens
        WHERE expires_at > NOW() AND revoked = FALSE
        "#
    )
    .fetch_all(pool)
    .await
    .context("Failed to query refresh tokens")?;

    for row in rows {
        if bcrypt::verify(token, &row.token_hash).unwrap_or(false) {
            sqlx::query!(
                "UPDATE refresh_tokens SET revoked = TRUE, revoked_at = NOW() WHERE id = $1",
                row.id
            )
            .execute(pool)
            .await
            .context("Failed to revoke refresh token")?;
            info!("Refresh token revoked");
            return Ok(());
        }
    }

    Err(anyhow::anyhow!("Refresh token not found or already revoked"))
}