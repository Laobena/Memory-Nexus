//! Top-K selection algorithms for fusion results

use super::{ScoredItem, FusionItem};
use crate::core::{BlockError, BlockResult};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use tracing::debug;

/// Selection strategy for top-k results
#[derive(Debug, Clone, Copy)]
pub enum SelectionStrategy {
    /// Heap-based selection
    HeapBased,
    /// Quick-select algorithm
    QuickSelect,
    /// Partial sort
    PartialSort,
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        SelectionStrategy::HeapBased
    }
}

/// Wrapper for min-heap ordering
#[derive(Debug, Clone)]
struct MinHeapItem {
    item: ScoredItem,
}

impl PartialEq for MinHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.item.score.eq(&other.item.score)
    }
}

impl Eq for MinHeapItem {}

impl PartialOrd for MinHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap
        other.item.score.partial_cmp(&self.item.score)
    }
}

impl Ord for MinHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Top-K selector for fusion results
pub struct TopKSelector {
    k: usize,
    strategy: SelectionStrategy,
}

impl TopKSelector {
    /// Create new selector
    pub fn new(k: usize) -> Self {
        Self {
            k,
            strategy: SelectionStrategy::default(),
        }
    }
    
    /// Set selection strategy
    pub fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    
    /// Select top-k items
    pub fn select(&self, mut items: Vec<ScoredItem>) -> BlockResult<Vec<ScoredItem>> {
        if items.len() <= self.k {
            items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            return Ok(items);
        }
        
        match self.strategy {
            SelectionStrategy::HeapBased => self.select_heap(items),
            SelectionStrategy::QuickSelect => self.select_quick(items),
            SelectionStrategy::PartialSort => self.select_partial_sort(items),
        }
    }
    
    /// Select with minimum quality threshold
    pub fn select_with_threshold(
        &self,
        items: Vec<ScoredItem>,
        min_score: f32,
    ) -> BlockResult<Vec<ScoredItem>> {
        // Filter by threshold first
        let filtered: Vec<_> = items
            .into_iter()
            .filter(|item| item.score >= min_score)
            .collect();
        
        if filtered.is_empty() {
            debug!("No items meet minimum threshold {}", min_score);
            return Ok(vec![]);
        }
        
        self.select(filtered)
    }
    
    /// Heap-based selection (O(n log k))
    fn select_heap(&self, items: Vec<ScoredItem>) -> BlockResult<Vec<ScoredItem>> {
        let mut heap = BinaryHeap::with_capacity(self.k);
        
        for item in items {
            if heap.len() < self.k {
                heap.push(MinHeapItem { item });
            } else if let Some(min) = heap.peek() {
                if item.score > min.item.score {
                    heap.pop();
                    heap.push(MinHeapItem { item });
                }
            }
        }
        
        // Extract and sort results
        let mut result: Vec<_> = heap
            .into_iter()
            .map(|wrapper| wrapper.item)
            .collect();
        
        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        
        Ok(result)
    }
    
    /// Quick-select based selection (O(n) average)
    fn select_quick(&self, mut items: Vec<ScoredItem>) -> BlockResult<Vec<ScoredItem>> {
        if items.len() <= self.k {
            items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            return Ok(items);
        }
        
        // Quick-select to partition around k-th element
        self.quick_select(&mut items, 0, items.len() - 1, self.k - 1);
        
        // Take first k elements and sort them
        items.truncate(self.k);
        items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        
        Ok(items)
    }
    
    /// Quick-select partition
    fn quick_select(&self, items: &mut [ScoredItem], left: usize, right: usize, k: usize) {
        if left >= right {
            return;
        }
        
        let pivot_idx = self.partition(items, left, right);
        
        if pivot_idx == k {
            return;
        } else if pivot_idx > k {
            self.quick_select(items, left, pivot_idx.saturating_sub(1), k);
        } else {
            self.quick_select(items, pivot_idx + 1, right, k);
        }
    }
    
    /// Partition for quick-select
    fn partition(&self, items: &mut [ScoredItem], left: usize, right: usize) -> usize {
        let pivot_score = items[right].score;
        let mut i = left;
        
        for j in left..right {
            if items[j].score >= pivot_score {
                items.swap(i, j);
                i += 1;
            }
        }
        
        items.swap(i, right);
        i
    }
    
    /// Partial sort selection (O(n log k))
    fn select_partial_sort(&self, mut items: Vec<ScoredItem>) -> BlockResult<Vec<ScoredItem>> {
        items.partial_sort(self.k, |a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        items.truncate(self.k);
        Ok(items)
    }
}

/// Extension trait for partial sorting
trait PartialSortExt {
    fn partial_sort<F>(&mut self, k: usize, compare: F)
    where
        F: Fn(&ScoredItem, &ScoredItem) -> Ordering;
}

impl PartialSortExt for Vec<ScoredItem> {
    fn partial_sort<F>(&mut self, k: usize, compare: F)
    where
        F: Fn(&ScoredItem, &ScoredItem) -> Ordering,
    {
        if k >= self.len() {
            self.sort_by(compare);
            return;
        }
        
        // Use select_nth_unstable for partial sorting
        let k = k.min(self.len());
        self.select_nth_unstable_by(k.saturating_sub(1), |a, b| compare(a, b));
        
        // Sort the first k elements
        self[..k].sort_by(compare);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_items(n: usize) -> Vec<ScoredItem> {
        (0..n)
            .map(|i| ScoredItem {
                item: FusionItem {
                    id: uuid::Uuid::new_v4(),
                    content: vec![i as u8],
                    relevance: (i as f32) / (n as f32),
                    freshness: 0.5,
                    diversity: 0.5,
                    authority: 0.5,
                    coherence: 0.5,
                    confidence: 0.5,
                    source_engine: super::super::EngineType::Accuracy,
                    timestamp: chrono::Utc::now(),
                },
                score: (i as f32) / (n as f32),
                components: Default::default(),
            })
            .collect()
    }
    
    #[test]
    fn test_heap_selection() {
        let selector = TopKSelector::new(5);
        let items = create_test_items(20);
        
        let selected = selector.select(items).unwrap();
        assert_eq!(selected.len(), 5);
        
        // Check ordering
        for i in 1..selected.len() {
            assert!(selected[i - 1].score >= selected[i].score);
        }
    }
    
    #[test]
    fn test_threshold_selection() {
        let selector = TopKSelector::new(10);
        let items = create_test_items(20);
        
        let selected = selector.select_with_threshold(items, 0.7).unwrap();
        
        // All selected items should meet threshold
        for item in &selected {
            assert!(item.score >= 0.7);
        }
    }
}