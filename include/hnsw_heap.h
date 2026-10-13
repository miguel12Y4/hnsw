#pragma once

#include "min_max_heap.h"
#include "hnsw_entry.h"

namespace HNSW {

using HNSWHeap = MinMaxHeap<Entry, EntryLess>;

}
