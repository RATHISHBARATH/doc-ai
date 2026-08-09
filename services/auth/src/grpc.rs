// ============================================================
// Auth Service – gRPC Service Implementation
// ============================================================

use anyhow::Result;
use tonic::{Request, Response, Status};
use tracing::{info, warn, instrument};

use crate::db::DbPool;
use crate::jwt::validate_jwt;

// Import generated proto types
use doc_ai_common::{
    auth_server::{Auth, AuthServer},
    health_server::{Health, HealthServer},
    LoginRequest, LoginResponse, ValidateRequest, ValidateResponse,
    HealthCheckRequest, HealthCheckResponse,
    health_check_response::ServingStatus,
};

// ============================================================
// Auth Service Implementation
// ============================================================

#[derive(Clone)]
pub struct AuthService {
    pool: DbPool,
    jwt_secret: String,
}

impl AuthService {
    pub fn new(pool: DbPool, jwt_secret: String) -> Self {
        Self { pool, jwt_secret }
    }
}

#[tonic::async_trait]
impl Auth for AuthService {
    /// Authenticate – validates username/password and returns a JWT.
    /// Currently only supports local users (email/password).
    #[instrument(skip(self, request))]
    async fn authenticate(
        &self,
        request: Request<LoginRequest>,
    ) -> Result<Response<LoginResponse>, Status> {
        let req = request.into_inner();
        warn!("Authenticate called for user: {}", req.username);

        // Placeholder: local authentication not yet implemented.
        Err(Status::unimplemented("Local authentication not yet implemented"))
    }

    /// ValidateToken – validates a JWT and returns the user ID.
    #[instrument(skip(self, request))]
    async fn validate_token(
        &self,
        request: Request<ValidateRequest>,
    ) -> Result<Response<ValidateResponse>, Status> {
        let req = request.into_inner();
        info!("ValidateToken called");

        // Validate the JWT using the secret.
        let user_id = validate_jwt(&req.token, &self.jwt_secret)
            .map_err(|e| {
                warn!("JWT validation failed: {}", e);
                Status::unauthenticated("Invalid token")
            })?;

        Ok(Response::new(ValidateResponse {
            valid: true,
            user_id,
        }))
    }
}

// ============================================================
// Health Service Implementation
// ============================================================

#[derive(Clone)]
pub struct HealthService;

#[tonic::async_trait]
impl Health for HealthService {
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let _ = request.into_inner();
        Ok(Response::new(HealthCheckResponse {
            status: ServingStatus::Serving as i32,
        }))
    }
}

// ============================================================
// Helper function for starting the gRPC server
// ============================================================

/// Builds and serves the gRPC server (Auth + Health) on the given address.
/// This binds and runs the server directly rather than returning an
/// intermediate router type, since the router's concrete type is awkward
/// to name explicitly in tonic 0.11.
pub async fn serve_grpc(
    pool: DbPool,
    jwt_secret: String,
    addr: std::net::SocketAddr,
) -> Result<(), tonic::transport::Error> {
    let auth_service = AuthService::new(pool, jwt_secret);
    let health_service = HealthService;

    tonic::transport::Server::builder()
        .add_service(AuthServer::new(auth_service))
        .add_service(HealthServer::new(health_service))
        .serve(addr)
        .await
}