use std::time::Duration;

use chrono::{DateTime, Utc};
use octocrab::models::{JobId, RunId};

pub mod client;
pub mod cost;

#[derive(Debug, serde::Serialize)]
pub enum WorkflowKind {
    Try,
    Auto,
    PR,
}

#[derive(Debug, serde::Serialize)]
pub struct RepositoryId {
    pub owner: String,
    pub name: String,
}

#[derive(Debug, serde::Serialize)]
pub struct WorkflowRun {
    pub id: RunId,
    pub repository: RepositoryId,
    pub kind: WorkflowKind,
    pub success: bool,
    pub created_at: DateTime<Utc>,
    pub duration: Duration,
    pub jobs: Vec<WorkflowJob>,
}

#[derive(Debug, serde::Serialize)]
pub struct WorkflowJob {
    pub id: JobId,
    pub name: String,
    pub platform: String,
    pub billable_duration: Duration,
    pub duration: Duration,
}
