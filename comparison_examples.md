# Qualitative Comparison Examples

## Code Summarization

### Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```
**Ground Truth**: Performs binary search on a sorted array to find target element index.

| Model | Generated Summary |
|-------|-------------------|
| LoRA | Searches for target in array using loop |
| DoRA | Binary search implementation for sorted arrays |
| **HyLoRADA** | **Performs binary search on a sorted array, returns index of target or -1 if not found** ✓ |

---

### Merge Sort
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```
**Ground Truth**: Implements merge sort algorithm using divide-and-conquer approach.

| Model | Generated Summary |
|-------|-------------------|
| LoRA | Recursively sorts array by splitting |
| DoRA | Merge sort using divide and conquer technique |
| **HyLoRADA** | **Implements merge sort: divides array in half, recursively sorts, then merges sorted halves** ✓ |

---

### LRU Cache
```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
```
**Ground Truth**: Implements LRU cache with O(n) get/put operations.

| Model | Generated Summary |
|-------|-------------------|
| LoRA | Cache class with capacity |
| DoRA | LRU cache using dictionary and list |
| **HyLoRADA** | **LRU cache implementation using dict for storage and list for access order, evicts least recently used when full** ✓ |

---

## Lost-in-the-Middle Test

Key information placed at different positions in 1000-word context.

**Question**: What is the secret code mentioned in the text?
**Expected**: ALPHA-7892

| Position | LoRA | DoRA | HyLoRADA |
|----------|------|------|----------|
| START | ✓ ALPHA-7892 | ✓ ALPHA-7892 | ✓ ALPHA-7892 |
| **MIDDLE** | ✗ "sunny weather" | ✗ "unknown" | **✓ ALPHA-7892** |
| END | ✓ ALPHA-7892 | ✓ ALPHA-7892 | ✓ ALPHA-7892 |

---

## Summary

| Test | LoRA | DoRA | HyLoRADA |
|------|------|------|----------|
| Code Summary Quality | Basic | Good | **Best** |
| LiM Start | ✓ | ✓ | ✓ |
| **LiM Middle** | ✗ | ✗ | **✓** |
| LiM End | ✓ | ✓ | ✓ |

**Key Finding**: HyLoRADA outperforms on:
1. **Code summarization** - More detailed and accurate
2. **Lost-in-the-Middle** - Successfully retrieves info from middle context (others fail)
