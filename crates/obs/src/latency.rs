//! Per-stage HDR histogram latency tracking.
//!
//! Stages mirror the pipeline: ws_recv → deser → feature_update → signal →
//! risk → order_send → ack. Each stage records nanoseconds; reports p50/p99/p99.9.

use hdrhistogram::Histogram;
use parking_lot::Mutex;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum LatencyStage {
    WsToDeser,
    DeserToFeature,
    FeatureToSignal,
    SignalToRisk,
    RiskToOrderSend,
    OrderSendToAck,
    EndToEndTickToOrder,
}

impl LatencyStage {
    pub fn name(self) -> &'static str {
        match self {
            LatencyStage::WsToDeser => "ws_to_deser",
            LatencyStage::DeserToFeature => "deser_to_feature",
            LatencyStage::FeatureToSignal => "feature_to_signal",
            LatencyStage::SignalToRisk => "signal_to_risk",
            LatencyStage::RiskToOrderSend => "risk_to_order_send",
            LatencyStage::OrderSendToAck => "order_send_to_ack",
            LatencyStage::EndToEndTickToOrder => "tick_to_order",
        }
    }

    pub fn all() -> &'static [LatencyStage] {
        &[
            LatencyStage::WsToDeser,
            LatencyStage::DeserToFeature,
            LatencyStage::FeatureToSignal,
            LatencyStage::SignalToRisk,
            LatencyStage::RiskToOrderSend,
            LatencyStage::OrderSendToAck,
            LatencyStage::EndToEndTickToOrder,
        ]
    }
}

pub struct LatencyTracker {
    histograms: Vec<Mutex<Histogram<u64>>>,
}

impl LatencyTracker {
    pub fn new() -> Self {
        let stages = LatencyStage::all().len();
        let histograms = (0..stages)
            .map(|_| {
                Mutex::new(
                    // 3 sig digits, 1ns - 60s range
                    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap(),
                )
            })
            .collect();
        Self { histograms }
    }

    pub fn record(&self, stage: LatencyStage, nanos: u64) {
        let idx = Self::index_of(stage);
        let _ = self.histograms[idx].lock().record(nanos.max(1));
    }

    pub fn snapshot(&self, stage: LatencyStage) -> LatencySnapshot {
        let h = self.histograms[Self::index_of(stage)].lock();
        LatencySnapshot {
            count: h.len(),
            min: h.min(),
            max: h.max(),
            mean: h.mean(),
            p50: h.value_at_quantile(0.50),
            p99: h.value_at_quantile(0.99),
            p999: h.value_at_quantile(0.999),
        }
    }

    pub fn snapshot_all(&self) -> Vec<(LatencyStage, LatencySnapshot)> {
        LatencyStage::all()
            .iter()
            .map(|&s| (s, self.snapshot(s)))
            .collect()
    }

    fn index_of(stage: LatencyStage) -> usize {
        LatencyStage::all().iter().position(|&s| s == stage).unwrap()
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LatencySnapshot {
    pub count: u64,
    pub min: u64,
    pub max: u64,
    pub mean: f64,
    pub p50: u64,
    pub p99: u64,
    pub p999: u64,
}

impl LatencySnapshot {
    pub fn fmt_us(&self) -> String {
        format!(
            "n={} p50={:.1}us p99={:.1}us p999={:.1}us max={:.1}us",
            self.count,
            self.p50 as f64 / 1_000.0,
            self.p99 as f64 / 1_000.0,
            self.p999 as f64 / 1_000.0,
            self.max as f64 / 1_000.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_reports() {
        let t = LatencyTracker::new();
        for ns in [100u64, 200, 300, 400, 500] {
            t.record(LatencyStage::FeatureToSignal, ns);
        }
        let s = t.snapshot(LatencyStage::FeatureToSignal);
        assert_eq!(s.count, 5);
        assert!(s.p50 >= 100 && s.p50 <= 500);
    }

    #[test]
    fn all_stages_covered() {
        let t = LatencyTracker::new();
        for stage in LatencyStage::all() {
            t.record(*stage, 1_000);
        }
        let snaps = t.snapshot_all();
        assert_eq!(snaps.len(), LatencyStage::all().len());
    }
}
