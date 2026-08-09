// ============================================================
// Auth Service – OAuth2 Module
// ============================================================

use anyhow::{Context, Result};
use oauth2::{
    basic::BasicClient,
    reqwest::async_http_client,
    AuthUrl, ClientId, ClientSecret, RedirectUrl, TokenResponse,
    AuthorizationCode, CsrfToken, Scope, TokenUrl,
};
use serde::Deserialize;
use tracing::{info, instrument};
use uuid::Uuid;

use crate::db::DbPool;
use crate::jwt::{create_jwt, create_refresh_token};

// ============================================================
// Provider Configuration
// ============================================================

/// Supported OAuth2 providers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Provider {
    Google,
    GitHub,
}

impl Provider {
    /// Returns the string representation used in the database.
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Google => "google",
            Provider::GitHub => "github",
        }
    }
}

// ============================================================
// OAuth2 Clients
// ============================================================

/// Holds OAuth2 clients for each provider.
#[derive(Clone)]
pub struct OAuthClients {
    pub google: BasicClient,
    pub github: BasicClient,
}

impl OAuthClients {
    /// Creates new OAuth2 clients from configuration.
    pub fn new(
        google_client_id: &str,
        google_client_secret: &str,
        github_client_id: &str,
        github_client_secret: &str,
        redirect_url: &str,
    ) -> Result<Self> {
        let redirect_url = RedirectUrl::new(redirect_url.to_string())
            .context("Invalid redirect URL")?;

        let google = BasicClient::new(
            ClientId::new(google_client_id.to_string()),
            Some(ClientSecret::new(google_client_secret.to_string())),
            AuthUrl::new("https://accounts.google.com/o/oauth2/v2/auth".to_string())?,
            Some(TokenUrl::new("https://oauth2.googleapis.com/token".to_string())?),
        )
        .set_redirect_uri(redirect_url.clone());

        let github = BasicClient::new(
            ClientId::new(github_client_id.to_string()),
            Some(ClientSecret::new(github_client_secret.to_string())),
            AuthUrl::new("https://github.com/login/oauth/authorize".to_string())?,
            Some(TokenUrl::new("https://github.com/login/oauth/access_token".to_string())?),
        )
        .set_redirect_uri(redirect_url);

        Ok(Self { google, github })
    }

    /// Returns the client for the given provider.
    pub fn for_provider(&self, provider: Provider) -> &BasicClient {
        match provider {
            Provider::Google => &self.google,
            Provider::GitHub => &self.github,
        }
    }
}

// ============================================================
// OAuth2 Flow Functions
// ============================================================

/// Generates the authorization URL for the given provider.
#[instrument(skip(clients, pool))]
pub async fn generate_auth_url(
    clients: &OAuthClients,
    pool: &DbPool,
    provider: Provider,
) -> Result<String> {
    let client = clients.for_provider(provider);

    // Generate a random state string and store it in the database.
    let state = CsrfToken::new_random();
    let state_str = state.secret();

    // Insert the state into oauth_states table.
    sqlx::query!(
        r#"
        INSERT INTO oauth_states (state, expires_at)
        VALUES ($1, NOW() + INTERVAL '5 minutes')
        "#,
        state_str
    )
    .execute(pool)
    .await
    .context("Failed to store OAuth state")?;

    // Build the authorization URL. authorize_url() takes a closure that
    // produces the CsrfToken; we pass ours in since we already generated
    // and stored it above.
    let (auth_url, _csrf_token) = client
        .authorize_url(|| state)
        .add_scope(Scope::new("openid".to_string()))
        .add_scope(Scope::new("email".to_string()))
        .add_scope(Scope::new("profile".to_string()))
        .url();

    Ok(auth_url.to_string())
}

// ============================================================
// Callback Handler – Exchanges code for token and creates user
// ============================================================

/// Handle the callback from the provider.
#[instrument(skip(clients, pool, code, state))]
pub async fn handle_callback(
    clients: &OAuthClients,
    pool: &DbPool,
    provider: Provider,
    code: &str,
    state: &str,
) -> Result<(String, String)> { // Returns (jwt, refresh_token)
    // 1. Verify the state exists in the database.
    let _stored_state = sqlx::query!(
        r#"
        SELECT state FROM oauth_states
        WHERE state = $1 AND expires_at > NOW()
        "#,
        state
    )
    .fetch_optional(pool)
    .await
    .context("Failed to verify OAuth state")?
    .ok_or_else(|| anyhow::anyhow!("Invalid or expired OAuth state"))?;

    // Delete the used state to prevent replay attacks.
    sqlx::query!(
        "DELETE FROM oauth_states WHERE state = $1",
        state
    )
    .execute(pool)
    .await
    .context("Failed to delete OAuth state")?;

    // 2. Exchange the authorization code for a token.
    let client = clients.for_provider(provider);
    let token_response = client
        .exchange_code(AuthorizationCode::new(code.to_string()))
        .request_async(async_http_client)
        .await
        .context("Failed to exchange code for token")?;

    // 3. Fetch user info from the provider.
    let user_info = fetch_user_info(provider, token_response.access_token().secret()).await?;

    // 4. Find or create the user in the database.
    let user_id = find_or_create_user(pool, provider, &user_info).await?;

    // 5. Generate JWT and refresh token.
    let jwt = create_jwt(&user_id, &crate::config::Config::from_env()?.jwt_secret)?;
    let refresh_token = create_refresh_token(&user_id, pool).await?;

    Ok((jwt, refresh_token))
}

// ============================================================
// User Info Structs and Fetching
// ============================================================

#[derive(Debug, Deserialize)]
struct GoogleUserInfo {
    sub: String,
    email: String,
    name: String,
}

#[derive(Debug, Deserialize)]
struct GitHubUserInfo {
    id: u64,
    email: String,
    name: String,
}

/// Fetches user info from the provider using the access token.
async fn fetch_user_info(provider: Provider, access_token: &str) -> Result<UserInfo> {
    let client = reqwest::Client::new();

    match provider {
        Provider::Google => {
            let response = client
                .get("https://www.googleapis.com/oauth2/v3/userinfo")
                .bearer_auth(access_token)
                .send()
                .await
                .context("Failed to fetch Google user info")?;

            let info: GoogleUserInfo = response
                .json()
                .await
                .context("Failed to parse Google user info")?;

            Ok(UserInfo {
                provider_id: info.sub,
                email: info.email,
                name: info.name,
            })
        }
        Provider::GitHub => {
            let response = client
                .get("https://api.github.com/user")
                .bearer_auth(access_token)
                .header("User-Agent", "DOC-AI-Auth")
                .send()
                .await
                .context("Failed to fetch GitHub user info")?;

            let info: GitHubUserInfo = response
                .json()
                .await
                .context("Failed to parse GitHub user info")?;

            // GitHub may not return the email in the /user endpoint if it's private.
            // We'll use the login as email if email is not provided.
            let email = if info.email.is_empty() {
                format!("{}@users.noreply.github.com", info.id)
            } else {
                info.email
            };

            Ok(UserInfo {
                provider_id: info.id.to_string(),
                email,
                name: info.name,
            })
        }
    }
}

/// Unified user info struct.
struct UserInfo {
    provider_id: String,
    email: String,
    name: String,
}

/// Finds an existing user or creates a new one.
async fn find_or_create_user(
    pool: &DbPool,
    provider: Provider,
    user_info: &UserInfo,
) -> Result<String> {
    let provider_str = provider.as_str();

    // Try to find the user by provider and provider_id.
    let user = sqlx::query!(
        r#"
        SELECT id FROM users
        WHERE provider = $1 AND provider_id = $2
        "#,
        provider_str,
        user_info.provider_id
    )
    .fetch_optional(pool)
    .await
    .context("Failed to query user")?;

    if let Some(user) = user {
        return Ok(user.id.to_string());
    }

    // If not found, create a new user.
    let user_id = Uuid::new_v4();
    sqlx::query!(
        r#"
        INSERT INTO users (id, email, name, provider, provider_id)
        VALUES ($1, $2, $3, $4, $5)
        "#,
        user_id,
        user_info.email,
        user_info.name,
        provider_str,
        user_info.provider_id
    )
    .execute(pool)
    .await
    .context("Failed to insert user")?;

    info!("Created new user: {} ({})", user_info.email, provider_str);
    Ok(user_id.to_string())
}