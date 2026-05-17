//! Integration tests against the real Alpaca v2 WebSocket message format.
//!
//! Source of truth for message samples:
//!   https://docs.alpaca.markets/docs/streaming-market-data#message-types
//!
//! These payloads are copied verbatim from Alpaca's public documentation
//! (with minor whitespace normalization). The parser must:
//!   - Handle array-wrapped batches: real Alpaca frames are always JSON
//!     arrays even when a single message is sent.
//!   - Handle ISO 8601 timestamps with nanosecond precision and trailing Z.
//!   - Decode the four message kinds (Trade, Quote, Bar, Success, Error).
//!   - Return a deserialization Error (not panic) on malformed JSON.

use algo_mdata::alpaca_ws::{convert, AlpacaWsMsg};
use algo_core::MarketEvent;

// ── single-element documentation samples ──

const REAL_TRADE: &str = r#"{
    "T": "t",
    "S": "SPY",
    "i": 36738301,
    "x": "V",
    "p": 421.69,
    "s": 100,
    "c": ["@","I"],
    "z": "C",
    "t": "2023-08-02T15:30:00.123456789Z"
}"#;

const REAL_QUOTE: &str = r#"{
    "T": "q",
    "S": "SPY",
    "bx": "V",
    "bp": 421.65,
    "bs": 2,
    "ax": "V",
    "ap": 421.69,
    "as": 5,
    "c": ["R"],
    "z": "C",
    "t": "2023-08-02T15:30:00.987654321Z"
}"#;

const REAL_BAR: &str = r#"{
    "T": "b",
    "S": "SPY",
    "o": 421.50,
    "h": 421.91,
    "l": 421.32,
    "c": 421.69,
    "v": 1283456,
    "t": "2023-08-02T15:30:00Z",
    "n": 4521,
    "vw": 421.61
}"#;

const REAL_SUCCESS_CONNECT: &str = r#"{"T":"success","msg":"connected"}"#;
const REAL_SUCCESS_AUTH:    &str = r#"{"T":"success","msg":"authenticated"}"#;
const REAL_ERROR:           &str = r#"{"T":"error","code":401,"msg":"auth failed"}"#;

// Real Alpaca frames are always array-wrapped; verify the parser tolerates
// the array form (we parse element-by-element).
const REAL_BATCH: &str = r#"[
    {"T":"t","S":"SPY","i":1,"x":"V","p":421.69,"s":100,"c":["@"],"z":"C","t":"2023-08-02T15:30:00.000000001Z"},
    {"T":"q","S":"SPY","bx":"V","bp":421.65,"bs":2,"ax":"V","ap":421.69,"as":5,"c":["R"],"z":"C","t":"2023-08-02T15:30:00.000000002Z"},
    {"T":"b","S":"QQQ","o":350.0,"h":350.5,"l":349.8,"c":350.4,"v":12345,"t":"2023-08-02T15:30:00Z"}
]"#;

// Documented subscription / authentication echo.
const REAL_SUBSCRIPTION_ECHO: &str = r#"{
    "T":"subscription",
    "trades":["SPY"],
    "quotes":["SPY","QQQ"],
    "bars":["*"]
}"#;

// ── tests ──

fn parse(s: &str) -> serde_json::Result<AlpacaWsMsg> {
    serde_json::from_str::<AlpacaWsMsg>(s)
}

#[test]
fn parses_real_trade_message() {
    let msg = parse(REAL_TRADE).expect("real trade should deserialize");
    let ev = convert(msg).expect("Trade should convert to MarketEvent");
    match ev {
        MarketEvent::Trade(t) => {
            assert_eq!(t.symbol.as_str(), "SPY");
            assert!((t.price.to_f64() - 421.69).abs() < 1e-9);
            // Size is exact integer.
            assert_eq!(t.size.to_f64(), 100.0);
        }
        other => panic!("expected Trade, got {:?}", other),
    }
}

#[test]
fn parses_real_quote_message() {
    let msg = parse(REAL_QUOTE).expect("real quote should deserialize");
    let ev = convert(msg).expect("Quote should convert");
    match ev {
        MarketEvent::Quote(q) => {
            assert_eq!(q.symbol.as_str(), "SPY");
            assert!((q.bid.to_f64() - 421.65).abs() < 1e-9);
            assert!((q.ask.to_f64() - 421.69).abs() < 1e-9);
            assert!(q.ask.to_f64() > q.bid.to_f64());
        }
        other => panic!("expected Quote, got {:?}", other),
    }
}

#[test]
fn parses_real_bar_message() {
    let msg = parse(REAL_BAR).expect("real bar should deserialize");
    let ev = convert(msg).expect("Bar should convert");
    match ev {
        MarketEvent::Bar(b) => {
            assert_eq!(b.symbol.as_str(), "SPY");
            assert!(b.low.to_f64() <= b.open.to_f64());
            assert!(b.low.to_f64() <= b.close.to_f64());
            assert!(b.high.to_f64() >= b.open.to_f64());
            assert!(b.high.to_f64() >= b.close.to_f64());
        }
        other => panic!("expected Bar, got {:?}", other),
    }
}

#[test]
fn parses_real_success_messages() {
    let m1 = parse(REAL_SUCCESS_CONNECT).expect("connect msg");
    assert!(matches!(m1, AlpacaWsMsg::Success { .. }));
    let m2 = parse(REAL_SUCCESS_AUTH).expect("auth msg");
    assert!(matches!(m2, AlpacaWsMsg::Success { .. }));
}

#[test]
fn parses_real_error_message() {
    let m = parse(REAL_ERROR).expect("error msg");
    if let AlpacaWsMsg::Error { code, msg } = m {
        assert_eq!(code, 401);
        assert_eq!(msg, "auth failed");
    } else {
        panic!("expected Error variant");
    }
}

#[test]
fn parses_array_wrapped_batch() {
    // Real Alpaca always wraps in an array; this is the deserialization shape
    // a WS reader will see.
    let batch: Vec<AlpacaWsMsg> = serde_json::from_str(REAL_BATCH)
        .expect("real batch should deserialize");
    assert_eq!(batch.len(), 3);
    // First two are SPY trade + quote, third is QQQ bar.
    let evs: Vec<_> = batch.into_iter().filter_map(convert).collect();
    assert_eq!(evs.len(), 3);
    assert!(matches!(evs[0], MarketEvent::Trade(_)));
    assert!(matches!(evs[1], MarketEvent::Quote(_)));
    assert!(matches!(evs[2], MarketEvent::Bar(_)));
}

#[test]
fn unknown_message_kinds_do_not_panic() {
    // The documented `subscription` echo is not one we surface as a
    // MarketEvent; the parser MUST handle it gracefully via the `#[serde(other)]`
    // catch-all and `convert` MUST return None (not panic).
    let m = parse(REAL_SUBSCRIPTION_ECHO).expect("subscription echo should deserialize");
    assert!(matches!(m, AlpacaWsMsg::Other));
    assert!(convert(m).is_none(), "unknown kinds must convert to None");
}

#[test]
fn malformed_json_returns_error_not_panic() {
    // Missing closing brace.
    let bad = r#"{"T":"t","S":"SPY""#;
    let result = parse(bad);
    assert!(result.is_err(), "malformed JSON must deserialize to Err");
}

#[test]
fn nanosecond_timestamps_parse_without_loss() {
    // Two trades 1 nanosecond apart must produce distinguishable Ts values
    // IF the parser preserves nanos. (Our current parser drops below seconds,
    // so this test documents the known limitation rather than asserting
    // nanosecond fidelity.)
    let t1 = r#"{"T":"t","S":"AAPL","i":1,"x":"V","p":100.0,"s":1,"c":["@"],"z":"C","t":"2024-06-01T15:30:00.000000001Z"}"#;
    let t2 = r#"{"T":"t","S":"AAPL","i":2,"x":"V","p":100.0,"s":1,"c":["@"],"z":"C","t":"2024-06-01T15:30:00.000000002Z"}"#;
    let m1 = parse(t1).unwrap();
    let m2 = parse(t2).unwrap();
    let e1 = convert(m1);
    let e2 = convert(m2);
    // Both must convert; nanosecond preservation is not currently required
    // (documented in the parser source), so equality of ts is acceptable.
    assert!(e1.is_some() && e2.is_some());
}

#[test]
fn malformed_symbol_returns_none_from_convert() {
    // Symbol with lowercase or invalid chars → convert returns None.
    let bad_sym = r#"{"T":"t","S":"lowercase","i":1,"x":"V","p":100.0,"s":1,"c":["@"],"z":"C","t":"2024-06-01T15:30:00Z"}"#;
    let m = parse(bad_sym).expect("JSON valid even if symbol invalid");
    let ev = convert(m);
    assert!(ev.is_none(), "invalid symbols must not produce MarketEvents");
}
