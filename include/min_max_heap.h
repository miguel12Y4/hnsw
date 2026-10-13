#pragma once
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>
template <typename T, typename Compare = std::less<T>>
class MinMaxHeap
{
public:
    using value_type = T;
    using size_type = std::size_t;
    static constexpr size_type DEFAULT_INITIAL_SIZE = 8;
    static constexpr size_type GROWTH_FACTOR = 2;
    MinMaxHeap() : MinMaxHeap(DEFAULT_INITIAL_SIZE) {};
    explicit MinMaxHeap(size_type c) : size_{0}, capacity_{c}, heap{std::make_unique<T[]>(capacity_)} {}
    std::vector<value_type> extract_n_min(size_type n)
    {
        n = std::min(size_, n);
        std::vector<value_type> r;
        r.reserve(n);
        while (r.size() < n)
        {
            r.emplace_back(get_min());
            pop_min();
        }
        return r;
    }
    const T &get_min() const
    {
        assert(!empty());
        return heap[0];
    }
    T &get_min()
    {
        assert(!empty());
        return heap[0];
    }
    const T &get_max() const
    {
        assert(!empty());
        switch (size_)
        {
        case 1:
            return heap[0];
        case 2:
            return heap[1];
        default:
            return compare(heap[1], heap[2]) ? heap[2] : heap[1];
        }
    }
    T &get_max()
    {
        assert(!empty());
        switch (size_)
        {
        case 1:
            return heap[0];
        case 2:
            return heap[1];
        default:
            return compare(heap[1], heap[2]) ? heap[2] : heap[1];
        }
    }
    template <typename... Args>
    void emplace(Args &&...a)
    {
        ensure_capacity();
        heap[size_] = T(std::forward<Args>(a)...);
        push_up(size_);
        ++size_;
    }
    void push(const T &e) { emplace(e); }
    void push(T &&e) { emplace(std::move(e)); }
    void pop_min()
    {
        assert(!empty());
        heap[0] = std::move(heap[size_ - 1]);
        --size_;
        if (size_ > 0)
            push_down(0);
    }
    void pop_max()
    {
        assert(!empty());
        if (size_ == 1)
        {
            --size_;
            return;
        }
        size_type p = (size_ > 2 && compare(heap[1], heap[2])) ? 2 : 1;
        heap[p] = std::move(heap[size_ - 1]);
        --size_;
        push_down(p);
    }
    inline void clear() { size_ = 0; }
    inline size_type size() const { return size_; }
    inline bool empty() const { return size_ == 0; }

private:
    size_type size_;
    size_type capacity_;
    Compare compare;
    std::unique_ptr<T[]> heap;
    static inline bool is_min_level(size_type i)
    {
        int l = 0;
        while (i > 0)
        {
            i = (i - 1) / 2;
            ++l;
        }
        return l % 2 == 0;
    }
    static inline size_type parent(size_type i) { return (i - 1) / 2; }
    static inline size_type grandparent(size_type i) { return parent(parent(i)); }
    static inline size_type left_child(size_type i) { return 2 * i + 1; }
    static inline size_type right_child(size_type i) { return 2 * i + 2; }
    inline bool has_children(size_type i) const { return left_child(i) < size_; }
    inline bool has_grandchildren(size_type i) const { return left_child(left_child(i)) < size_; }
    inline void ensure_capacity()
    {
        if (size_ == capacity_)
        {
            size_type nc = GROWTH_FACTOR * capacity_;
            auto nh = std::make_unique<T[]>(nc);
            std::memcpy(nh.get(), heap.get(), sizeof(T) * size_);
            heap = std::move(nh);
            capacity_ = nc;
        }
    }
    inline void push_up(size_type i)
    {
        if (i == 0)
            return;
        if (is_min_level(i))
        {
            if (compare(heap[parent(i)], heap[i]))
            {
                std::swap(heap[parent(i)], heap[i]);
                push_up_max(parent(i));
            }
            else
                push_up_min(i);
        }
        else
        {
            if (compare(heap[i], heap[parent(i)]))
            {
                std::swap(heap[i], heap[parent(i)]);
                push_up_min(parent(i));
            }
            else
                push_up_max(i);
        }
    }
    inline void push_up_min(size_t i)
    {
        if (i > 2)
        {
            auto gp = grandparent(i);
            if (compare(heap[i], heap[gp]))
            {
                std::swap(heap[i], heap[gp]);
                push_up_min(gp);
            }
        }
    }
    inline void push_up_max(size_t i)
    {
        if (i > 2)
        {
            auto gp = grandparent(i);
            if (compare(heap[gp], heap[i]))
            {
                std::swap(heap[gp], heap[i]);
                push_up_max(gp);
            }
        }
    }
    inline void push_down(size_type i)
    {
        if (is_min_level(i))
            push_down_min(i);
        else
            push_down_max(i);
    }
    inline void push_down_min(size_type i)
    {
        while (has_children(i))
        {
            size_type l = left_child(i), r = right_child(i), m = i;
            if (l < size_ && compare(heap[l], heap[m]))
                m = l;
            if (r < size_ && compare(heap[r], heap[m]))
                m = r;
            if (has_grandchildren(i))
            {
                size_type ll = left_child(l), lr = right_child(l), rl = left_child(r), rr = right_child(r);
                if (ll < size_ && compare(heap[ll], heap[m]))
                    m = ll;
                if (lr < size_ && compare(heap[lr], heap[m]))
                    m = lr;
                if (rl < size_ && compare(heap[rl], heap[m]))
                    m = rl;
                if (rr < size_ && compare(heap[rr], heap[m]))
                    m = rr;
            }
            if (m == i)
                break;
            std::swap(heap[i], heap[m]);
            if (m >= left_child(left_child(i)))
            {
                auto mp = parent(m);
                if (compare(heap[mp], heap[m]))
                    std::swap(heap[mp], heap[m]);
                i = m;
            }
            else
                break;
        }
    }
    inline void push_down_max(size_type i)
    {
        while (has_children(i))
        {
            size_type l = left_child(i), r = right_child(i), m = i;
            if (l < size_ && compare(heap[m], heap[l]))
                m = l;
            if (r < size_ && compare(heap[m], heap[r]))
                m = r;
            if (has_grandchildren(i))
            {
                size_type ll = left_child(l), lr = right_child(l), rl = left_child(r), rr = right_child(r);
                if (ll < size_ && compare(heap[m], heap[ll]))
                    m = ll;
                if (lr < size_ && compare(heap[m], heap[lr]))
                    m = lr;
                if (rl < size_ && compare(heap[m], heap[rl]))
                    m = rl;
                if (rr < size_ && compare(heap[m], heap[rr]))
                    m = rr;
            }
            if (m == i)
                break;
            std::swap(heap[i], heap[m]);
            if (m >= left_child(left_child(i)))
            {
                auto mp = parent(m);
                if (compare(heap[m], heap[mp]))
                    std::swap(heap[m], heap[mp]);
                i = m;
            }
            else
                break;
        }
    }
};
