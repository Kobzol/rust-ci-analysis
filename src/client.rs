use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use octocrab::models::workflows::{Job, Run};
use octocrab::models::JobId;
use octocrab::Octocrab;

use crate::{RepositoryId, WorkflowJob, WorkflowKind, WorkflowRun};

pub struct GitHubApi {
    pub client: Octocrab,
}

impl GitHubApi {
    pub async fn get_workflow_runs(
        &self,
        since: DateTime<Utc>,
    ) -> anyhow::Result<Vec<WorkflowRun>> {
        let mut runs = get_workflow_runs(&self.client, "rust-lang", "rust", since).await?;
        runs.extend(get_workflow_runs(&self.client, "rust-lang-ci", "rust", since).await?);
        Ok(runs)
    }
}

async fn get_workflow_runs(
    client: &Octocrab,
    owner: &str,
    repo: &str,
    since: DateTime<Utc>,
) -> anyhow::Result<Vec<WorkflowRun>> {
    let mut runs = vec![];

    let mut page = 0u32;
    loop {
        let mut response = client
            .workflows(owner, repo)
            .list_all_runs()
            .status("completed")
            .per_page(50)
            .page(page)
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("Cannot download workflow runs: {error:?}"))?;
        let mut page_runs = response.take_items();
        page_runs.retain(|run| run.created_at >= since);
        if page_runs.is_empty() {
            break;
        }

        let futs = page_runs
            .into_iter()
            .map(|run| async move {
                let jobs = client
                    .workflows(owner, repo)
                    .list_jobs(run.id)
                    .per_page(100)
                    .send()
                    .await?
                    .take_items();

                let billable: BillableResponse = client
                    .get(
                        format!(
                            "/repos/{}/{}/actions/runs/{}/timing",
                            run.repository.owner.as_ref().unwrap().login,
                            run.repository.name,
                            run.id
                        ),
                        None::<&()>,
                    )
                    .await?;

                Ok(parse_workflow_run(run, jobs, billable))
            })
            .collect::<Vec<_>>();
        let parsed_runs = futures_util::future::join_all(futs)
            .await
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?;
        runs.extend(parsed_runs.into_iter().filter_map(|r| r));

        page += 1;
    }
    Ok(runs)
}

fn parse_workflow_run(run: Run, jobs: Vec<Job>, billable: BillableResponse) -> Option<WorkflowRun> {
    let kind = match (run.event.as_str(), run.head_branch.as_str()) {
        ("push", "auto") => WorkflowKind::Auto,
        ("push", "try") => WorkflowKind::Try,
        ("pull_request", _) => WorkflowKind::PR,
        _ => return None,
    };
    let success = run.conclusion.as_deref() == Some("success");
    let created_at = run.created_at;
    let updated_at = run.updated_at;
    let Ok(duration) = (updated_at - created_at).to_std() else {
        return None;
    };

    let Some(owner) = run.repository.owner.map(|o| o.login) else {
        return None;
    };

    // JobId -> (duration, platform)
    let mut job_info: HashMap<JobId, (Duration, String)> = HashMap::default();
    for (platform, info) in billable.billable.platforms {
        for job in info.job_runs {
            assert!(job_info
                .insert(
                    job.job_id,
                    (Duration::from_millis(job.duration_ms), platform.clone())
                )
                .is_none());
        }
    }

    let mut parsed_jobs = vec![];
    for job in jobs {
        if let Some(info) = job_info.get(&job.id) {
            let duration = (job.completed_at.unwrap_or(job.started_at) - job.started_at)
                .to_std()
                .unwrap();
            let job = WorkflowJob {
                id: job.id,
                name: job.name,
                billable_duration: info.0,
                duration,
                platform: info.1.clone(),
            };
            parsed_jobs.push(job);
        }
    }

    Some(WorkflowRun {
        id: run.id,
        repository: RepositoryId {
            owner,
            name: run.repository.name,
        },
        kind,
        success,
        created_at,
        duration,
        jobs: parsed_jobs,
    })
}

#[derive(serde::Deserialize, Debug)]
struct JobBillable {
    job_id: JobId,
    duration_ms: u64,
}

#[derive(serde::Deserialize, Debug)]
struct RunPlatformInfo {
    job_runs: Vec<JobBillable>,
}

#[derive(serde::Deserialize, Debug)]
struct RunBillablePlatforms {
    #[serde(flatten)]
    platforms: HashMap<String, RunPlatformInfo>,
}

#[derive(serde::Deserialize, Debug)]
struct BillableResponse {
    billable: RunBillablePlatforms,
}
