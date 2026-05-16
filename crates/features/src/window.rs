//! Look-ahead-safe rolling window. The whole point is to forbid future peeking.
//!
//! Each entry is stored with its timestamp. `last_n` returns the most recent
//! entries; `as_of(ts)` returns entries strictly at-or-before `ts`. There is
//! no `&[T]` accessor on purpose.

use algo_core::Ts;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Window<T: Clone> {
    capacity: usize,
    inner: VecDeque<(Ts, T)>,
}

impl<T: Clone> Window<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            inner: VecDeque::with_capacity(capacity),
        }
    }

    /// Push a new value. Panics if `ts < last_ts()` (monotonic invariant).
    pub fn push(&mut self, ts: Ts, value: T) {
        if let Some((last_ts, _)) = self.inner.back() {
            assert!(
                ts >= *last_ts,
                "Window::push requires monotonic timestamps"
            );
        }
        if self.inner.len() == self.capacity {
            self.inner.pop_front();
        }
        self.inner.push_back((ts, value));
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn last(&self) -> Option<&(Ts, T)> {
        self.inner.back()
    }

    pub fn last_ts(&self) -> Option<Ts> {
        self.inner.back().map(|(t, _)| *t)
    }

    /// Most recent `n` values, oldest-first. `None` if fewer than `n`.
    pub fn last_n(&self, n: usize) -> Option<Vec<T>> {
        if self.inner.len() < n {
            return None;
        }
        let start = self.inner.len() - n;
        Some(
            self.inner
                .iter()
                .skip(start)
                .map(|(_, v)| v.clone())
                .collect(),
        )
    }

    /// All values with timestamp ≤ `ts`, oldest-first. Use this from feature
    /// code that needs to assert it does not see beyond a given event time.
    pub fn as_of(&self, ts: Ts) -> Vec<T> {
        self.inner
            .iter()
            .filter(|(t, _)| *t <= ts)
            .map(|(_, v)| v.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pushes_in_order_and_caps() {
        let mut w: Window<i32> = Window::new(3);
        for i in 0..5 {
            w.push(Ts::from_nanos(i), i as i32);
        }
        assert_eq!(w.len(), 3);
        assert_eq!(w.last_n(3).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn as_of_does_not_see_future() {
        let mut w: Window<i32> = Window::new(10);
        for i in 0..5 {
            w.push(Ts::from_nanos(i), i as i32);
        }
        // At ts=2 we should only see 0,1,2
        assert_eq!(w.as_of(Ts::from_nanos(2)), vec![0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "monotonic")]
    fn rejects_non_monotonic() {
        let mut w: Window<i32> = Window::new(3);
        w.push(Ts::from_nanos(10), 1);
        w.push(Ts::from_nanos(5), 2);
    }
}
