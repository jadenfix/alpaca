//! Symbol universe: selection + filters (ADV, hard-to-borrow, earnings calendar).

use ahash::{AHashMap, AHashSet};
use algo_core::Symbol;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UniverseConfig {
    pub include: Vec<String>,
    pub hard_to_borrow: Vec<String>,
    pub min_adv_shares: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Universe {
    members: AHashSet<Symbol>,
    htb: AHashSet<Symbol>,
    adv: AHashMap<Symbol, f64>,
    min_adv: f64,
}

impl Universe {
    pub fn from_config(cfg: &UniverseConfig) -> Self {
        let members: AHashSet<Symbol> = cfg
            .include
            .iter()
            .filter_map(|s| Symbol::new(s))
            .collect();
        let htb: AHashSet<Symbol> = cfg
            .hard_to_borrow
            .iter()
            .filter_map(|s| Symbol::new(s))
            .collect();
        Self {
            members,
            htb,
            adv: AHashMap::new(),
            min_adv: cfg.min_adv_shares,
        }
    }

    pub fn observe_adv(&mut self, sym: Symbol, adv_shares: f64) {
        self.adv.insert(sym, adv_shares);
    }

    pub fn is_tradable_long(&self, sym: Symbol) -> bool {
        self.members.contains(&sym)
            && self.adv.get(&sym).copied().unwrap_or(f64::INFINITY) >= self.min_adv
    }

    pub fn is_tradable_short(&self, sym: Symbol) -> bool {
        self.is_tradable_long(sym) && !self.htb.contains(&sym)
    }

    pub fn members(&self) -> impl Iterator<Item = &Symbol> {
        self.members.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_by_adv_and_htb() {
        let cfg = UniverseConfig {
            include: vec!["AAPL".into(), "TSLA".into()],
            hard_to_borrow: vec!["TSLA".into()],
            min_adv_shares: 100_000.0,
        };
        let mut u = Universe::from_config(&cfg);
        let aapl = Symbol::new("AAPL").unwrap();
        let tsla = Symbol::new("TSLA").unwrap();
        u.observe_adv(aapl, 5_000_000.0);
        u.observe_adv(tsla, 5_000_000.0);
        assert!(u.is_tradable_long(aapl));
        assert!(u.is_tradable_long(tsla));
        assert!(u.is_tradable_short(aapl));
        assert!(!u.is_tradable_short(tsla)); // HTB
    }
}
