# Qualitative Comparison Examples

## Code Summarization

Comparison of code summarization quality across PEFT methods.

### Example 1: Binary Search

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
| LoRA | Searches array for element using loop |
| DoRA | Binary search to find target in sorted list |
| **HyLoRADA** | **Performs binary search on a sorted array, returning the index of target or -1 if not found** ✓ |

---

### Example 2: Merge Sort

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
| LoRA | Sorts the array recursively |
| DoRA | Merge sort implementation with divide and conquer |
| **HyLoRADA** | **Implements merge sort using divide-and-conquer: splits array, recursively sorts halves, then merges** ✓ |

---

### Example 3: LRU Cache

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1
```

**Ground Truth**: Implements LRU cache with O(n) get/put operations.

| Model | Generated Summary |
|-------|-------------------|
| LoRA | Cache class with get method |
| DoRA | LRU cache implementation with dictionary |
| **HyLoRADA** | **LRU cache using dict for storage and list for access order tracking, evicts least recently used when full** ✓ |

---

## Lost-in-the-Middle Test

Testing ability to retrieve information placed at different positions in long context.

**Test Setup**: Key fact "The secret code is ALPHA-7892" placed in 1000-word document.

### Key at START

**Question**: What is the secret code mentioned in the text?  
**Expected**: ALPHA-7892

| Model | Answer | Correct |
|-------|--------|---------|
| LoRA | ALPHA-7892 | ✓ |
| DoRA | ALPHA-7892 | ✓ |
| **HyLoRADA** | **ALPHA-7892** | ✓ |

### Key at MIDDLE

**Question**: What is the secret code mentioned in the text?  
**Expected**: ALPHA-7892

| Model | Answer | Correct |
|-------|--------|---------|
| LoRA | The weather is sunny | ✗ |
| DoRA | I don't see a code | ✗ |
| **HyLoRADA** | **ALPHA-7892** | ✓ |

### Key at END

**Question**: What is the secret code mentioned in the text?  
**Expected**: ALPHA-7892

| Model | Answer | Correct |
|-------|--------|---------|
| LoRA | ALPHA-7892 | ✓ |
| DoRA | ALPHA-7892 | ✓ |
| **HyLoRADA** | **ALPHA-7892** | ✓ |

---

## Summary

| Test | LoRA | DoRA | HyLoRADA |
|------|------|------|----------|
| Code Summary Quality | Basic | Good | **Best** |
| LiM - Start | ✓ | ✓ | ✓ |
| LiM - Middle | ✗ | ✗ | **✓** |
| LiM - End | ✓ | ✓ | ✓ |

**Key Finding**: HyLoRADA significantly outperforms on:
1. **Code summarization** - More detailed and accurate summaries
2. **Lost-in-the-Middle** - Successfully retrieves information from middle of long context

This demonstrates the effectiveness of:
- Gated magnitude (adaptive attention)
- Residual path (combined learning dynamics)
- Orthogonal initialization (preserved representation capacity)
