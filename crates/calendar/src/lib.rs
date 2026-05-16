//! NYSE session calendar.
//!
//! Pragmatic implementation: weekday/weekend, regular hours (09:30–16:00 ET),
//! configurable holidays + half-day overrides. Strategy entries are gated to
//! Regular session by default.

use algo_core::Ts;
use hifitime::Epoch;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SessionStatus {
    PreMarket,
    Regular,
    PostMarket,
    Closed,
    Holiday,
}

#[derive(Clone, Debug)]
pub struct NyseCalendar {
    holidays: HashSet<(i32, u16)>,
    half_days: HashMap<(i32, u16), u8>,
}

impl Default for NyseCalendar {
    fn default() -> Self {
        let mut holidays = HashSet::new();
        // Minimal 2026 seed; production should load from configs/holidays/<year>.toml
        holidays.insert((2026, day_of_year(2026, 1, 1))); // New Year's
        holidays.insert((2026, day_of_year(2026, 1, 19))); // MLK
        holidays.insert((2026, day_of_year(2026, 2, 16))); // Presidents
        holidays.insert((2026, day_of_year(2026, 4, 3))); // Good Friday
        holidays.insert((2026, day_of_year(2026, 5, 25))); // Memorial
        holidays.insert((2026, day_of_year(2026, 6, 19))); // Juneteenth
        holidays.insert((2026, day_of_year(2026, 7, 3))); // observed July 4 (Sat)
        holidays.insert((2026, day_of_year(2026, 9, 7))); // Labor
        holidays.insert((2026, day_of_year(2026, 11, 26))); // Thanksgiving
        holidays.insert((2026, day_of_year(2026, 12, 25))); // Christmas

        let mut half_days = HashMap::new();
        half_days.insert((2026, day_of_year(2026, 11, 27)), 13); // Black Friday
        half_days.insert((2026, day_of_year(2026, 12, 24)), 13); // Christmas Eve
        Self { holidays, half_days }
    }
}

#[derive(Copy, Clone, Debug)]
struct EtTime {
    year: i32,
    ord: u16,
    weekday: u8, // 0=Mon..6=Sun
    hour: u8,
    min: u8,
}

fn ts_to_et(ts: Ts) -> EtTime {
    let e = Epoch::from_unix_duration(hifitime::Duration::from_truncated_nanoseconds(ts.nanos));
    let (year_u, month_u, day_u, hour_u, minute_u, _sec, _nano) = e.to_gregorian_utc();
    let year = year_u;
    let month = month_u as u32;
    let day = day_u as u32;
    let hour = hour_u as i32;
    let weekday_utc = weekday_zeller(year, month as i32, day as i32);
    let dst = is_us_dst(year, month, day);
    let offset = if dst { -4 } else { -5 };
    let mut h = hour + offset;
    let mut d = day as i32;
    let mut m = month as i32;
    let mut y = year;
    let mut wd = weekday_utc as i32;
    if h < 0 {
        h += 24;
        d -= 1;
        wd = (wd + 6) % 7;
        if d < 1 {
            m -= 1;
            if m < 1 {
                m = 12;
                y -= 1;
            }
            d = days_in_month(y, m as u32) as i32;
        }
    }
    EtTime {
        year: y,
        ord: day_of_year(y, m as u32, d as u32),
        weekday: wd as u8,
        hour: h as u8,
        min: minute_u,
    }
}

fn days_in_year(y: i32) -> u16 {
    if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
        366
    } else {
        365
    }
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if days_in_year(y) == 366 => 29,
        2 => 28,
        _ => 0,
    }
}

fn day_of_year(year: i32, month: u32, day: u32) -> u16 {
    let m = [0u16, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut d = m[(month - 1) as usize] + day as u16;
    if month > 2 && days_in_year(year) == 366 {
        d += 1;
    }
    d
}

/// Zeller's congruence → 0=Mon..6=Sun.
fn weekday_zeller(year: i32, month: i32, day: i32) -> u8 {
    let (y, m) = if month < 3 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let k = y.rem_euclid(100);
    let j = y.div_euclid(100);
    let h = (day + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 + 5 * j).rem_euclid(7);
    // Zeller: 0=Sat..6=Fri → 0=Mon..6=Sun
    let map = [5u8, 6, 0, 1, 2, 3, 4];
    map[h as usize]
}

/// US DST: second Sunday of March → first Sunday of November.
fn is_us_dst(year: i32, month: u32, day: u32) -> bool {
    match month {
        1..=2 | 12 => false,
        4..=10 => true,
        3 => {
            let wd1 = weekday_zeller(year, 3, 1) as i32; // 0=Mon..6=Sun
            // first Sunday: day such that (wd1 + day - 1) % 7 == 6 (Sun)
            let first_sun = 1 + ((6 - wd1).rem_euclid(7)) as u32;
            let second_sun = first_sun + 7;
            day >= second_sun
        }
        11 => {
            let wd1 = weekday_zeller(year, 11, 1) as i32;
            let first_sun = 1 + ((6 - wd1).rem_euclid(7)) as u32;
            day < first_sun
        }
        _ => false,
    }
}

impl NyseCalendar {
    pub fn status(&self, ts: Ts) -> SessionStatus {
        let et = ts_to_et(ts);
        if et.weekday >= 5 {
            return SessionStatus::Closed;
        }
        if self.holidays.contains(&(et.year, et.ord)) {
            return SessionStatus::Holiday;
        }
        let close_hour = self
            .half_days
            .get(&(et.year, et.ord))
            .copied()
            .unwrap_or(16);
        let mins = et.hour as u16 * 60 + et.min as u16;
        let open = 9u16 * 60 + 30;
        let close = close_hour as u16 * 60;
        let pre_open = 4u16 * 60;
        let post_close = 20u16 * 60;
        if mins < pre_open || mins >= post_close {
            SessionStatus::Closed
        } else if mins < open {
            SessionStatus::PreMarket
        } else if mins < close {
            SessionStatus::Regular
        } else {
            SessionStatus::PostMarket
        }
    }

    pub fn is_regular(&self, ts: Ts) -> bool {
        matches!(self.status(ts), SessionStatus::Regular)
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SessionWindow {
    pub open: Ts,
    pub close: Ts,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts_at(y: i32, mo: u8, d: u8, h: u8, m: u8) -> Ts {
        let e = Epoch::from_gregorian_utc(y, mo, d, h, m, 0, 0);
        Ts::from_nanos(e.to_unix_seconds() as i64 * 1_000_000_000)
    }

    #[test]
    fn weekday_known_dates() {
        // 2026-05-16 is Saturday
        assert_eq!(weekday_zeller(2026, 5, 16), 5);
        // 2026-01-01 is Thursday
        assert_eq!(weekday_zeller(2026, 1, 1), 3);
    }

    #[test]
    fn regular_hours_weekday_dst() {
        // Mon 2026-05-18, 15:00 UTC = 11:00 EDT → Regular
        let cal = NyseCalendar::default();
        assert_eq!(cal.status(ts_at(2026, 5, 18, 15, 0)), SessionStatus::Regular);
    }

    #[test]
    fn weekend_is_closed() {
        let cal = NyseCalendar::default();
        // Sat 2026-05-16 16:00 UTC
        assert_eq!(cal.status(ts_at(2026, 5, 16, 16, 0)), SessionStatus::Closed);
    }

    #[test]
    fn holiday_recognized() {
        let cal = NyseCalendar::default();
        // 2026-12-25 (Friday) 15:00 UTC = 10:00 EST
        assert_eq!(cal.status(ts_at(2026, 12, 25, 15, 0)), SessionStatus::Holiday);
    }

    #[test]
    fn pre_market_then_regular_then_post() {
        let cal = NyseCalendar::default();
        // 2026-05-18 Monday, EDT (UTC-4)
        // 13:00 UTC = 09:00 ET → PreMarket
        assert_eq!(cal.status(ts_at(2026, 5, 18, 13, 0)), SessionStatus::PreMarket);
        // 14:00 UTC = 10:00 ET → Regular
        assert_eq!(cal.status(ts_at(2026, 5, 18, 14, 0)), SessionStatus::Regular);
        // 21:00 UTC = 17:00 ET → PostMarket
        assert_eq!(cal.status(ts_at(2026, 5, 18, 21, 0)), SessionStatus::PostMarket);
    }
}
