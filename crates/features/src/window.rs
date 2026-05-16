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

    /// Append the most recent `n` values into `out`. Returns `true` iff the
    /// window had at least `n` entries; `out` is left untouched on `false`.
    ///
    /// Use this to reuse a scratch buffer instead of allocating a fresh `Vec`
    /// every call — the hot-path version of `last_n`.
    pub fn last_n_into(&self, n: usize, out: &mut Vec<T>) -> bool {
        if self.inner.len() < n {
            return false;
        }
        let start = self.inner.len() - n;
        out.clear();
        out.reserve(n);
        for (_, v) in self.inner.iter().skip(start) {
            out.push(v.clone());
        }
        true
    }

    /// Zero-copy peek at the most recent `n` values. Returns two contiguous
    /// slices because the underlying `VecDeque` may wrap around the ring's
    /// boundary — concatenate them logically as oldest-first. `None` if the
    /// window has fewer than `n` entries.
    pub fn last_n_slice(&self, n: usize) -> Option<(&[(Ts, T)], &[(Ts, T)])> {
        if self.inner.len() < n {
            return None;
        }
        let start = self.inner.len() - n;
        let (front, back) = self.inner.as_slices();
        if start < front.len() {
            Some((&front[start..], back))
        } else {
            Some((&[], &back[start - front.len()..]))
        }
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

    #[test]
    fn last_n_into_reuses_buffer() {
        let mut w: Window<i32> = Window::new(10);
        for i in 0..10 {
            w.push(Ts::from_nanos(i), i as i32);
        }
        let mut buf = Vec::with_capacity(20);
        let cap_before = buf.capacity();
        assert!(w.last_n_into(5, &mut buf));
        assert_eq!(buf, vec![5, 6, 7, 8, 9]);
        // Capacity should not shrink across calls when re-used.
        assert!(buf.capacity() >= cap_before);
        assert!(!w.last_n_into(20, &mut buf));
        assert_eq!(buf, vec![5, 6, 7, 8, 9], "out unchanged when not enough data");
    }

    #[test]
    fn last_n_slice_matches_last_n() {
        let mut w: Window<i32> = Window::new(10);
        // Push enough to wrap the underlying VecDeque
        for i in 0..15 {
            w.push(Ts::from_nanos(i), i as i32);
        }
        let owned = w.last_n(5).unwrap();
        let (front, back) = w.last_n_slice(5).unwrap();
        let zero_copy: Vec<i32> = front.iter().chain(back).map(|(_, v)| *v).collect();
        assert_eq!(owned, zero_copy);
    }
}
