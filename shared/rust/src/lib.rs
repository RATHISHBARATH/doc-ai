// ============================================================
// DOC AI Common Library – Shared Utilities and Protobuf Types
// ============================================================

// Include each generated Protobuf file as a separate module.
// The generated files contain all the types, clients, and servers.
// We re-export everything from each module so that they are available
// directly under the crate root (e.g., doc_ai_common::AuthClient).
mod auth_gen {
    include!(concat!(env!("OUT_DIR"), "/doc_ai.auth.rs"));
}
mod inference_gen {
    include!(concat!(env!("OUT_DIR"), "/doc_ai.inference.rs"));
}
mod health_gen {
    include!(concat!(env!("OUT_DIR"), "/grpc.health.v1.rs"));
}

// Re-export all items from each generated module.
// This makes types available as doc_ai_common::AuthClient, etc.
pub use auth_gen::*;
pub use health_gen::*;
pub use inference_gen::*;

// Note: Placeholder modules (hardware, logging, telemetry) have been removed
// because they do not yet exist. They can be added later when implemented.
