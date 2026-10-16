//! Prepared reference conditions, keyed by their inputs and shared by voice ids.
//! Ids stay bounded FIFO. A condition lives while an id holds it or while the
//! LRU content cache keeps it, so a repeated reference skips preparation even
//! after its earlier ids were released.
use std::collections::{HashMap, VecDeque};

pub type VoiceKey = [u8; 32];

pub struct Limits {
    pub ids: usize,
    pub entries: usize,
    pub bytes: u64,
}

struct Slot<V> {
    value: V,
    bytes: u64,
    ids: usize,
    cached: bool,
}

pub struct VoiceStore<V> {
    limits: Limits,
    slots: HashMap<VoiceKey, Slot<V>>,
    ids: HashMap<String, VoiceKey>,
    id_order: VecDeque<String>,
    lru: VecDeque<VoiceKey>,
    cached_bytes: u64,
}

impl<V> VoiceStore<V> {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            slots: HashMap::new(),
            ids: HashMap::new(),
            id_order: VecDeque::new(),
            lru: VecDeque::new(),
            cached_bytes: 0,
        }
    }

    /// A new id for an already prepared condition, or `None` on a miss.
    pub fn reuse(&mut self, key: VoiceKey) -> Option<String> {
        self.slots.contains_key(&key).then(|| self.bind(key))
    }

    /// Store a condition prepared after a miss and return its first id.
    pub fn insert(&mut self, key: VoiceKey, value: V, bytes: u64) -> String {
        self.slots.insert(
            key,
            Slot {
                value,
                bytes,
                ids: 0,
                cached: false,
            },
        );
        self.bind(key)
    }

    pub fn get(&self, id: &str) -> Option<&V> {
        self.ids
            .get(id)
            .and_then(|key| self.slots.get(key))
            .map(|slot| &slot.value)
    }

    pub fn ids(&self) -> impl Iterator<Item = String> + '_ {
        self.ids.keys().cloned()
    }

    /// Drops the id only. Its condition stays while the cache keeps it.
    pub fn release(&mut self, id: &str) {
        let Some(key) = self.ids.remove(id) else {
            return;
        };
        if let Some(index) = self.id_order.iter().position(|v| v == id) {
            self.id_order.remove(index);
        }
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.ids = slot.ids.saturating_sub(1);
            if slot.ids == 0 && !slot.cached {
                self.slots.remove(&key);
            }
        }
    }

    // The id is attached before cache eviction runs, so an entry too large to
    // cache still lives as long as its id.
    fn bind(&mut self, key: VoiceKey) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.ids += 1;
        }
        self.ids.insert(id.clone(), key);
        self.id_order.push_back(id.clone());
        self.admit(key);
        while self.id_order.len() > self.limits.ids {
            match self.id_order.front().cloned() {
                Some(oldest) => self.release(&oldest),
                None => break,
            }
        }
        id
    }

    fn admit(&mut self, key: VoiceKey) {
        let Some(slot) = self.slots.get_mut(&key) else {
            return;
        };
        if slot.cached {
            if let Some(index) = self.lru.iter().position(|k| *k == key) {
                self.lru.remove(index);
            }
            self.lru.push_back(key);
            return;
        }
        if self.limits.entries == 0 || slot.bytes > self.limits.bytes {
            return;
        }
        slot.cached = true;
        self.cached_bytes += slot.bytes;
        self.lru.push_back(key);
        while self.lru.len() > self.limits.entries || self.cached_bytes > self.limits.bytes {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            if let Some(slot) = self.slots.get_mut(&oldest) {
                slot.cached = false;
                self.cached_bytes -= slot.bytes;
                if slot.ids == 0 {
                    self.slots.remove(&oldest);
                }
            }
        }
    }

    #[cfg(test)]
    fn usage(&self) -> (usize, usize, u64, usize) {
        (
            self.slots.len(),
            self.lru.len(),
            self.cached_bytes,
            self.ids.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u8) -> VoiceKey {
        [n; 32]
    }
    fn store(ids: usize, entries: usize, bytes: u64) -> VoiceStore<u32> {
        VoiceStore::new(Limits {
            ids,
            entries,
            bytes,
        })
    }

    #[test]
    fn qwen3_tts_voice_store_miss_then_hit_shares_one_condition() {
        let mut s = store(8, 256, 100);
        assert!(s.reuse(key(1)).is_none());
        let first = s.insert(key(1), 11, 10);
        let second = s.reuse(key(1)).unwrap();
        assert_ne!(first, second);
        assert_eq!(s.get(&first), Some(&11));
        assert_eq!(s.get(&second), Some(&11));
        assert!(s.reuse(key(2)).is_none());
        assert_eq!(s.usage(), (1, 1, 10, 2));
    }

    #[test]
    fn qwen3_tts_voice_store_release_drops_id_and_keeps_cached_condition() {
        let mut s = store(8, 256, 100);
        let first = s.insert(key(1), 11, 10);
        s.release(&first);
        s.release(&first);
        assert_eq!(s.get(&first), None);
        assert_eq!(s.ids().count(), 0);
        assert_eq!(s.usage(), (1, 1, 10, 0));
        let again = s.reuse(key(1)).unwrap();
        assert_eq!(s.get(&again), Some(&11));
        assert_eq!(s.usage(), (1, 1, 10, 1));
    }

    #[test]
    fn qwen3_tts_voice_store_evicts_least_recently_used_by_entries() {
        let mut s = store(8, 2, 100);
        let a = s.insert(key(1), 1, 10);
        let b = s.insert(key(2), 2, 10);
        s.release(&a);
        s.release(&b);
        let a = s.reuse(key(1)).unwrap();
        s.release(&a);
        let c = s.insert(key(3), 3, 10);
        s.release(&c);
        assert!(s.reuse(key(2)).is_none());
        assert_eq!(s.usage(), (2, 2, 20, 0));
        assert!(s.reuse(key(1)).is_some());
        assert!(s.reuse(key(3)).is_some());
    }

    #[test]
    fn qwen3_tts_voice_store_evicts_least_recently_used_by_bytes() {
        let mut s = store(8, 256, 10);
        for n in 1..=3 {
            let id = s.insert(key(n), n as u32, 4);
            s.release(&id);
        }
        assert_eq!(s.usage(), (2, 2, 8, 0));
        assert!(s.reuse(key(1)).is_none());
        assert!(s.reuse(key(2)).is_some());
        assert!(s.reuse(key(3)).is_some());
    }

    #[test]
    fn qwen3_tts_voice_store_live_id_outlives_cache_eviction_and_is_reused() {
        let mut s = store(8, 1, 100);
        let a = s.insert(key(1), 1, 10);
        let b = s.insert(key(2), 2, 10);
        assert_eq!(s.get(&a), Some(&1));
        assert_eq!(s.usage(), (2, 1, 10, 2));
        s.release(&b);
        let again = s.reuse(key(1)).unwrap();
        assert_eq!(s.get(&again), Some(&1));
        assert!(s.reuse(key(2)).is_none());
        assert_eq!(s.usage(), (1, 1, 10, 2));
        s.release(&a);
        s.release(&again);
        assert_eq!(s.usage(), (1, 1, 10, 0));
    }

    #[test]
    fn qwen3_tts_voice_store_bounds_ids_fifo_without_dropping_cache() {
        let mut s = store(2, 256, 100);
        let a = s.insert(key(1), 1, 10);
        let b = s.insert(key(2), 2, 10);
        let c = s.insert(key(3), 3, 10);
        assert_eq!(s.get(&a), None);
        assert_eq!(s.get(&b), Some(&2));
        assert_eq!(s.get(&c), Some(&3));
        assert_eq!(s.usage(), (3, 3, 30, 2));
        assert!(s.reuse(key(1)).is_some());
        assert_eq!(s.get(&b), None);
    }

    #[test]
    fn qwen3_tts_voice_store_never_caches_an_oversized_condition() {
        let mut s = store(8, 256, 10);
        let small = s.insert(key(1), 1, 4);
        s.release(&small);
        let big = s.insert(key(2), 2, 11);
        assert_eq!(s.get(&big), Some(&2));
        assert_eq!(s.usage(), (2, 1, 4, 1));
        s.release(&big);
        assert!(s.reuse(key(2)).is_none());
        assert!(s.reuse(key(1)).is_some());
    }
}
