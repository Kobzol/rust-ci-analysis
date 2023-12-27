use crate::WorkflowJob;
use std::fmt::{Display, Formatter};
use std::iter::Sum;
use std::ops::Add;

pub struct Thousandth(pub u64);

impl Add for Thousandth {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sum for Thousandth {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|v| v.0).sum::<u64>())
    }
}

impl Display for Thousandth {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}", self.0 as f64 / 1000.0)
    }
}

/// Returns the total cost of the workflow run in thousands of a dollar
pub fn cost_in_thousandths_dollars(job: &WorkflowJob) -> Thousandth {
    // Round up to minutes
    let minutes = (job.billable_duration.as_millis() as f64 / 60000.0).ceil() as u64;
    get_minute_cost_and_multiplier(&job.platform)
        .map(|cost| {
            let actual_minutes = minutes * cost.multiplier;
            Thousandth(cost.per_minute.0 * actual_minutes)
        })
        .unwrap_or(Thousandth(0))
}

struct Cost {
    per_minute: Thousandth,
    multiplier: u64,
}

/// Taken from:
/// https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions
fn get_minute_cost_and_multiplier(platform: &str) -> Option<Cost> {
    let (per_minute, multiplier) = match platform {
        "UBUNTU_2_CORE" => (8, 1),
        "UBUNTU_4_CORE" => (16, 1),
        "UBUNTU_8_CORE" => (32, 1),
        "UBUNTU_16_CORE" => (64, 1),
        "WINDOWS_8_CORE" => (64, 2),
        "MACOS_XLARGE" => (160, 10),
        // These should have duration 0
        "UBUNTU" | "MACOS" => return None,
        _ => {
            log::warn!("Unknown platform {platform}");
            return None;
        }
    };
    Some(Cost {
        per_minute: Thousandth(per_minute),
        multiplier,
    })
}
