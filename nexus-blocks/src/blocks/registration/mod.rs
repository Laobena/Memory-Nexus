//! Registration and UUID Management
//! 
//! Master reference system for tracking all requests and operations

pub mod uuid_block;

pub use uuid_block::{
    UUIDBlock,
    UUIDRequest,
    UUIDContext,
    UUIDConfig,
    UUIDMetadata,
    RequestType,
    LinkType,
    UUIDRecovery,
    RecoveryOp,
};