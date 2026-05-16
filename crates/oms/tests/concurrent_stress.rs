//! Concurrent stress tests for the OMS position cache.
//!
//! The OMS uses `papaya::HashMap` for cross-thread position state. These
//! tests hammer it from multiple threads to surface races and panics.

use algo_core::{Fill, Notional, OrderId, OrderIntent, OrderType, Price, Qty, Side, Symbol, TimeInForce, Ts};
use algo_oms::Oms;
use std::sync::Arc;
use std::thread;

fn mk_fill(sym: Symbol, side: Side, qty: i64, price: f64) -> Fill {
    Fill {
        ts: Ts::from_nanos(0),
        order_id: OrderId::new(),
        symbol: sym,
        side,
        qty: Qty::from_i64(qty),
        price: Price::from_f64(price).unwrap(),
        fees: Notional::ZERO,
    }
}

fn mk_intent(sym: Symbol) -> OrderIntent {
    OrderIntent {
        id: OrderId::new(),
        ts: Ts::from_nanos(0),
        symbol: sym,
        side: Side::Buy,
        qty: Qty::from_i64(100),
        order_type: OrderType::Market,
        tif: TimeInForce::Day,
        strategy: "t".into(),
        tag: None,
    }
}

#[test]
fn oms_concurrent_fills_single_symbol() {
    // 8 threads × 1000 fills each, alternating buys and sells.
    // Final position should be zero (equal buys and sells).
    let oms = Arc::new(Oms::default());
    let sym = Symbol::new("AAPL").unwrap();

    let n_threads = 8;
    let n_per_thread = 1_000;
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let oms = oms.clone();
            thread::spawn(move || {
                for i in 0..n_per_thread {
                    let side = if (tid + i) % 2 == 0 { Side::Buy } else { Side::Sell };
                    oms.on_fill(&mk_fill(sym, side, 1, 100.0));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    // n_threads * n_per_thread = 8000 fills, alternating; net should be zero.
    let pos = oms.position(sym);
    assert_eq!(pos.qty, 0.0, "concurrent buy/sell did not net to zero: {}", pos.qty);
}

#[test]
fn oms_concurrent_fills_disjoint_symbols() {
    // Each thread fills a different symbol. No contention; all positions should
    // be exactly `n_per_thread * 1` shares long.
    let oms = Arc::new(Oms::default());
    let n_threads = 16;
    let n_per_thread = 500;
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let oms = oms.clone();
            thread::spawn(move || {
                let sym = Symbol::new(&format!("SYM{tid}")).unwrap();
                for _ in 0..n_per_thread {
                    oms.on_fill(&mk_fill(sym, Side::Buy, 1, 100.0));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    for tid in 0..n_threads {
        let sym = Symbol::new(&format!("SYM{tid}")).unwrap();
        let pos = oms.position(sym);
        assert_eq!(pos.qty, n_per_thread as f64, "tid {tid} expected {}, got {}", n_per_thread, pos.qty);
    }
}

#[test]
fn oms_concurrent_submits_and_fills_interleaved() {
    // Submit + fill loop across threads. Stresses both the open-order Mutex
    // and the concurrent position map.
    let oms = Arc::new(Oms::default());
    let n_threads = 4;
    let n_per_thread = 200;
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let oms = oms.clone();
            thread::spawn(move || {
                let sym = Symbol::new(&format!("SYM{tid}")).unwrap();
                for i in 0..n_per_thread {
                    let intent = mk_intent(sym);
                    let id = intent.id;
                    oms.on_submit(intent, Some(format!("b{tid}-{i}")));
                    oms.on_fill(&Fill {
                        ts: Ts::from_nanos(i as i64),
                        order_id: id,
                        symbol: sym,
                        side: Side::Buy,
                        qty: Qty::from_i64(100),
                        price: Price::from_f64(100.0).unwrap(),
                        fees: Notional::ZERO,
                    });
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(oms.open_orders_count(), n_threads * n_per_thread);
    for tid in 0..n_threads {
        let sym = Symbol::new(&format!("SYM{tid}")).unwrap();
        let pos = oms.position(sym);
        assert_eq!(pos.qty, (n_per_thread * 100) as f64);
    }
}

#[test]
fn oms_no_data_race_under_reader_pressure() {
    // Continuous writer + many readers. Readers should never see torn state
    // (qty or avg_px from inconsistent updates).
    let oms = Arc::new(Oms::default());
    let sym = Symbol::new("AAPL").unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let writer = {
        let oms = oms.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            let mut i = 0i64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                oms.on_fill(&mk_fill(sym, side, 1, 100.0 + (i % 10) as f64));
                i += 1;
            }
        })
    };

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let oms = oms.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let pos = oms.position(sym);
                    // Sanity: qty and avg_px must be finite (no torn writes producing NaN).
                    assert!(pos.qty.is_finite());
                    assert!(pos.avg_px.is_finite());
                }
            })
        })
        .collect();

    thread::sleep(std::time::Duration::from_millis(300));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    writer.join().unwrap();
    for h in readers {
        h.join().unwrap();
    }
}
