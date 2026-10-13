#pragma once

#include "hnsw_entry.h"

#include <set>

namespace HNSW {

using HNSWEntryFlatMap = std::multiset<Entry, EntryLess>;

} // namespace HNSW
