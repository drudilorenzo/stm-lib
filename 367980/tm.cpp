/**
 * @file   tm.c
 * @author Lorenzo Drudi 
 *         <email: lorenzo.drudi@epfl.ch>
 *         <sciper: 367980>
 *
 * @section LICENSE
 *
 * Copyright © 2023 Lorenzo Drudi.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * any later version. Please see https://gnu.org/licenses/gpl.html
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * @section DESCRIPTION
 *
 * Implementation of a transactional memory library based on the 
 * Transactional Locking II (TL2) algorithm.
 * 
 * Project for the course "Concurrent Programming (CS-453)" at EPFL.
 * 
 * @cite 
 * [1] Dice, D., Shalev, O., & Shavit, N. (2006, September).
 *     Transactional locking II.
 *     In International Symposium on Distributed Computing (pp. 194-208).
 *     Berlin, Heidelberg: Springer Berlin Heidelberg.
**/

/**
 * TODO:
 * - [ ] Why it initializes 0 and 1 pos
 * - [ ] What is GET VIRTUAL ADDRESS
 * - [ ] Explain map composition
 * - [ ] Change get_word function (what is &?)
 * - [ ] Why list and not vector?
*/


/**
 * CONSTANTS & MACROS
*/


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#ifdef __STDC_NO_ATOMICS__
#error Current C11 compiler does not support atomic operations
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "macros.h"
#include <tm.hpp>

// Virtual memory address length: 64 bits
#define VIRTUAL_ADDRESS_LENGTH 64

// Offset from the base address: 48 bits (lsb of the virtual address)
#define OFFSET_LENGTH 48

// Index of the segment: 16 bits (msb of the virtual address)
#define SEG_IDX_LENGTH (VIRTUAL_ADDRESS_LENGTH - OFFSET_LENGTH)

// Bit mask for the offset
#define OFFSET_MASK ((uint64_t(1) << OFFSET_LENGTH) - 1)

// Number of segments preallocated
#define NUM_SEG ((uint64_t(1) << SEG_IDX_LENGTH) - 1)

// Offset of the first segment
#define START_ADDRESS 0

// Get the segment index from the virtual address
#define GET_SEG_IDX(x) ((uint64_t)x >> OFFSET_LENGTH)

// Get the offset from the virtual address
#define GET_OFFSET(x, align) (((uint64_t)x & OFFSET_MASK) / align)

// TODO: understand
#define GET_VIRTUAL_ADDRESS(x) ((uint64_t)x << OFFSET_LENGTH)

#define LOCKED_MASK 1
#define VERSION_SHIFT 1

// Get the version from the lock (remove the locked bit)
#define GET_VERSION(lock) (lock >> VERSION_SHIFT)
#define IS_LOCKED(lock) (lock & LOCKED_MASK)

// Create a lock from a version and a locked bit
#define CREATE_VERSION(lock, version) (lock | (version << VERSION_SHIFT))

// Need +1 because fetch_add returns the previous value
#define ATOMIC_FETCH_INC(atom) (atom.fetch_add(1) + 1)

#define COMMIT(transaction) clean_transaction(transaction); return true;
#define ABORT(transaction) clean_transaction(transaction); return false;


/*
 * STRUCTURES
*/


typedef struct transaction_t {
    bool is_read_only = false;                           // True if the transaction is read-only
    uint64_t read_version;                               // Read version
    uint64_t write_version;                              // Write version
    std::unordered_set<std::atomic_uint64_t *> read_set; // Word -> value
    std::unordered_map<uintptr_t, uint64_t> write_set;   // Word -> value
} transaction_t;

typedef struct word_t {
    uint64_t value;            // 64-bit value
    std::atomic_uint64_t lock; // Versioning lock
    word_t() : value(0), lock(0) {}
    word_t(const word_t &w) : value(w.value), lock(w.lock.load()) {}
} word_t;

typedef struct stm_t {
    size_t size;                             // Size of the first allocated segment (in bytes)
    size_t align;                            // Alignment that the region must provide (in bytes)
    std::atomic_uint64_t next = 1;           // Number of allocated segments
    std::vector<std::vector<word_t>> memory; // Array of segments (each segment is an array of words)
    stm_t() : memory(NUM_SEG, std::vector<word_t>(START_ADDRESS)) {}
} stm_t;

static std::atomic_uint64_t global_clock = 0; // global version clock
static thread_local transaction_t gtransaction;
static stm_t gstm;


/*
 * HELPERS DECLARATIONS (for the documentation see below)
 */


inline word_t &get_word(stm_t *smt, const uintptr_t addr);
inline bool acquire_lock(std::atomic_uint64_t *lock);
inline bool release_lock(std::atomic_uint64_t *lock, const uint64_t new_version);
inline void clean_transaction(transaction_t *transaction);
inline void release_locks_from_list(const std::list<word_t *> locks);


/*
 * STM LIBRARY IMPLEMENTATION
 */


/** Create (i.e. allocate + init) a new shared memory region, with one first non-free-able allocated segment of the requested size and alignment.
 * @param size  Size of the first shared segment of memory to allocate (in bytes), must be a positive multiple of the alignment
 * @param align Alignment (in bytes, must be a power of 2) that the shared memory region must support
 * @return Opaque shared memory region handle, 'invalid_shared' on failure
**/
shared_t tm_create(size_t size, size_t align) noexcept {
    gstm.size = size;
    gstm.align = align;
    gstm.memory[0].resize(size / align);
    gstm.memory[1].resize(size / align);
    return &gstm;
}

/** Destroy (i.e. clean-up + free) a given shared memory region.
 * @param shared Shared memory region to destroy, with no running transaction
**/
void tm_destroy(shared_t unused(shared)) noexcept {
    // No dynamic memory allocation
}

/** [thread-safe] Return the start address of the first allocated segment in the shared memory region.
 * @param shared Shared memory region to query
 * @return Start address of the first allocated segment
**/
void *tm_start(shared_t unused(shared)) noexcept {
    return (void *)(GET_VIRTUAL_ADDRESS(1));
}

/** [thread-safe] Return the size (in bytes) of the first allocated segment of the shared memory region.
 * @param shared Shared memory region to query
 * @return First allocated segment size
**/
size_t tm_size(shared_t shared) noexcept {
    return ((stm_t *)shared)->size;
}

/** [thread-safe] Return the alignment (in bytes) of the memory accesses on the given shared memory region.
 * @param shared Shared memory region to query
 * @return Alignment used globally
**/
size_t tm_align(shared_t shared) noexcept {
    return ((stm_t *)shared)->align;
}

/** [thread-safe] Begin a new transaction on the given shared memory region.
 * @param shared Shared memory region to start a transaction on
 * @param is_ro  Whether the transaction is read-only
 * @return Opaque transaction ID, 'invalid_tx' on failure
**/
tx_t tm_begin(shared_t unused(shared), bool is_ro) noexcept {
    gtransaction.read_version = global_clock.load();
    gtransaction.is_read_only = is_ro;
    return (uintptr_t)&gtransaction;
}

/** [thread-safe] Write operation in the given transaction, source in a private region and target in the shared region.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param source Source start address (in a private region)
 * @param size   Length to copy (in bytes), must be a positive multiple of the alignment
 * @param target Target start address (in the shared region)
 * @return Whether the whole transaction can continue
**/
bool tm_write(shared_t shared, tx_t tx, void const *source, size_t size, void *target) noexcept {
    stm_t *reg = (stm_t *)shared;
    transaction_t *transaction = (transaction_t *)tx;

    for (size_t i = 0, t = (size_t)target, s = (size_t)source; i < size / reg->align; i++, t += reg->align, s += reg->align) {
        transaction->write_set.insert_or_assign(t, (*(uint64_t *)s));
    }

  return true;
}

/** [thread-safe] Read operation in the given transaction, source in the shared region and target in a private region.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param source Source start address (in the shared region)
 * @param size   Length to copy (in bytes), must be a positive multiple of the alignment
 * @param target Target start address (in a private region)
 * @return Whether the whole transaction can continue
**/
bool tm_read(shared_t shared, tx_t tx, void const *source, size_t size, void *target) noexcept {
    stm_t *stm = (stm_t *)shared;
    transaction_t *transaction = (transaction_t *)tx;
     
    auto hint(end(transaction->read_set));
    uint64_t version_before;
    for (size_t i = 0, address = (size_t)source, t = (size_t)target; i < size / stm->align; i++, address += stm->align, t += stm->align) {

        // check if the word is in the write set
        if (!transaction->is_read_only) {
            auto it = transaction->write_set.find(address);
            if (it != transaction->write_set.end()) {
                *(uint64_t*)t = it->second;
                continue;
            }
        }

        word_t &word = get_word(stm, address);

        version_before = GET_VERSION(word.lock.load());

        // If the word version is not up-to-date, abort the transaction
        if (version_before > transaction->read_version) {
            ABORT(transaction);
        }

        *(uint64_t*)t = word.value;
        
        if (!transaction->is_read_only) {
            hint = transaction->read_set.emplace_hint(hint, &word.lock);
            if (IS_LOCKED(word.lock.load())) {
                ABORT(transaction);
            }
        }    

        // If the version after the read is greater than the version before, 
        // then the word has been updated by another transaction.
        if (GET_VERSION(word.lock.load()) != version_before || IS_LOCKED(word.lock.load())) {
            ABORT(transaction);
        }
    }

    return true;  
}

/** [thread-safe] End the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to end
 * @return Whether the whole transaction committed
**/
bool tm_end(shared_t shared, tx_t tx) noexcept {
    transaction_t *transaction = (transaction_t *)tx;
    stm_t *reg = (stm_t *)shared;

    // If the transaction is read-only, commit it
    if (transaction->is_read_only || transaction->write_set.empty()) {
        COMMIT(transaction);
    }

    // need to store the lock to be able to release them later
    std::list<word_t *> locks;

    // try to acquire the locks
    for (const auto v : transaction->write_set) {
        word_t &vlock = get_word(reg, v.first);
        if (!acquire_lock(&vlock.lock)) {
            release_locks_from_list(locks);
            ABORT(transaction);
        }
        locks.push_back(&vlock);
    }

    transaction->write_version = ATOMIC_FETCH_INC(global_clock);

    // ensure that anything is changed in the read-set (no concurrent actors)
    if (transaction->read_version != transaction->write_version - 1) {
        uint64_t v;
        for (const std::atomic_uint64_t *l : transaction->read_set) {
            v = l->load();
            if (IS_LOCKED(v) || GET_VERSION(v) > transaction->read_version) {
                release_locks_from_list(locks);
                ABORT(transaction);
            }
        }
    }

    for (const auto v : transaction->write_set) {
        word_t &vlock = get_word(reg, v.first);
        vlock.value = v.second;
        if (!release_lock(&vlock.lock, transaction->write_version)) {
            ABORT(transaction);
        }
    }

    COMMIT(transaction);
}


/** [thread-safe] Memory allocation in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param size   Allocation requested size (in bytes), must be a positive multiple of the alignment
 * @param target Pointer in private memory receiving the address of the first byte of the newly allocated, aligned segment
 * @return Whether the whole transaction can continue (success/nomem), or not (abort_alloc)
**/
Alloc tm_alloc(shared_t shared, tx_t unused(tx), size_t unused(size), void **target) noexcept {
    stm_t *stm = ((stm_t *)shared);
    *target = (void *)(GET_VIRTUAL_ADDRESS((ATOMIC_FETCH_INC(stm->next))));
    stm->memory[stm->next.load()].resize(size / stm->align);
    return Alloc::success;
}

/** [thread-safe] Memory freeing in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param target Address of the first byte of the previously allocated segment to deallocate
 * @return Whether the whole transaction can continue
**/
bool tm_free(shared_t unused(shared), tx_t unused(tx), void *unused(segment)) noexcept {
    // No dynamic memory allocation
    return true;
}


/*
 * HELPER FUNCTIONS
 */


/** Get the word from the virtual memory.
 * @param reg Shared memory region
 * @param addr Virtual address
 * @return Word
*/
inline word_t &get_word(stm_t *smt, const uintptr_t addr) {
  return smt->memory[GET_SEG_IDX(addr)][GET_OFFSET(addr, smt->align)];
}

/** Acquire a lock.
 * @param lock Lock to acquire
 * @return Whether the lock has been acquired
*/
inline bool acquire_lock(std::atomic_uint64_t *lock) {
    uint64_t v = lock->load();
    return IS_LOCKED(v) ? false : lock->compare_exchange_strong(v, CREATE_VERSION(true, GET_VERSION(v)));
}

/** Release a lock.
 * @param lock Lock to release
 * @param new_version New version to set
 * @return Whether the lock has been released
*/
inline bool release_lock(std::atomic_uint64_t *lock, const uint64_t new_version) {
    uint64_t v = lock->load();
    return !(IS_LOCKED(v)) ? false : lock->compare_exchange_strong(v, CREATE_VERSION(false, new_version));
 }

/** Clean the transaction state.
 * @param transaction Transaction to clean
*/
inline void clean_transaction(transaction_t *transaction) {
    transaction->write_set.clear();
    transaction->read_set.clear();
    transaction->read_version = 0;
    transaction->is_read_only = false;
}

inline void release_locks_from_list(const std::list<word_t *> locks) {
    for (const auto &vlock : locks) {
        release_lock(&vlock->lock, GET_VERSION(vlock->lock.load()));
    }
}