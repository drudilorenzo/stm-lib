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

// Requested features
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE   200809L
#ifdef __STDC_NO_ATOMICS__
    #error Current C11 compiler does not support atomic operations
#endif

// External headers
#include <atomic>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <unordered_set>
#include <unordered_map>

// Internal headers
#include <tm.hpp>
#include "macros.h"

// Virtual memory address length: 64 bits
#define VIRTUAL_ADDRESS_BITS 64

// 48 bits (lsb) for the offset from the base address
#define OFFSET_BITS 48

// 16 bits (msb) for the segment index
#define SEG_IDX_BITS (VIRTUAL_ADDRESS_BITS - OFFSET_BITS)

// mask used to extract the offset from the virtual address
#define OFFSET_MASK ((uint64_t(1) << OFFSET_BITS) - 1)

// base address
#define BASE_SEG ((uint64_t(1) << SEG_IDX_BITS) - 1) // 65535
#define BASE_OFFSET 0

// macros to extract the segment index, the offset and the virtual address
#define GET_SEG_IDX(s) ((uint64_t)s >> OFFSET_BITS)
#define GET_OFFSET(s, align) (((uint64_t)s & OFFSET_MASK) / align)
#define GET_VIRTUAL_ADDRESS(s) ((uint64_t)s << OFFSET_BITS)

#define GET_WORD_FROM_ADDRESS(address) word_t* word = &stm.memory[GET_SEG_IDX(address)][GET_OFFSET(address, stm.align)]

#define IS_MULTIPLE_OF(x, y) (x % y == 0)

#define LOCKED_MASK 1
#define VERSION_SHIFT 1

#define GET_VERSION(lock) (lock >> VERSION_SHIFT)
#define IS_LOCKED(lock) (lock & LOCKED_MASK)

#define ABORT clean_transaction(); return false;
#define COMMIT clean_transaction(); return true;

// need to do +1 because featch add returns the previous value
#define ATOMIC_FETCH_INC(atom) (atom.fetch_add(1) + 1)

/**
 * @brief Word contained in a shared memory segment.
*/
typedef struct transaction_t {
    bool is_read_only = false;                         // True if the transaction is read-only
    uint64_t read_version;                             // Read version
    uint64_t write_version;                            // Write version
    std::unordered_set<std::atomic_uint64_t*> read_set;// Word -> value
    std::unordered_map<uintptr_t, uint64_t> write_set; // Word -> value
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
    std::atomic_uint64_t next;               // Number of allocated segments
    std::atomic_uint global_clock;           // Global shared clock 
    std::vector<std::vector<word_t>> memory; // Array of segments (each segment is an array of words)
    // TODO: choose prealloc size
    stm_t() : global_clock(0), memory(BASE_SEG, std::vector<word_t>(BASE_OFFSET)) {}
} stm_t;


// Avoid allocation - reallocation of the transaction
static thread_local transaction_t transaction;
static stm_t stm;

/*
 * INTERNAL FUNCTIONS
*/

/** Clean the transaction state.**/
inline void clean_transaction() noexcept {
    if (!transaction.is_read_only) {
        transaction.write_set.clear();
    }
    transaction.read_set.clear();
}

/** Check if the lock is free and acquire it.
 * @param lock Lock to acquire
 * @return Whether the lock has been acquired
 */
inline bool acquire_lock(std::atomic_uint64_t* lock) noexcept {
    uint64_t v = lock->load();
    return IS_LOCKED(v) ? false : lock->compare_exchange_weak(v, v | LOCKED_MASK);
}

/** Release the lock.
 * @param lock Lock to release
 * @return Whether the lock has been released
 */
inline bool release_lock(std::atomic_uint64_t* lock) noexcept {
    uint64_t v = lock->load();
    return IS_LOCKED(v) ? lock->compare_exchange_strong(v, v & ~LOCKED_MASK) : false;
}

/** Release all the locks acquired by the transaction.**/
inline void release_all_locks() noexcept {
    for (auto it = transaction.write_set.begin(); it != transaction.write_set.end(); it++) {
        GET_WORD_FROM_ADDRESS(it->first);
        release_lock(&word->lock);
    }
}

/*
 * STM LIBRARY IMPLEMENTATION
*/

/** Create (i.e. allocate + init) a new shared memory region, with one first non-free-able allocated segment of the requested size and alignment.
 * @param size  Size of the first shared segment of memory to allocate (in bytes), must be a positive multiple of the alignment
 * @param align Alignment (in bytes, must be a power of 2) that the shared memory region must support
 * @return Opaque shared memory region handle, 'invalid_shared' on failure
**/
shared_t tm_create(size_t size, size_t align) noexcept {

    // Check if the size is a multiple of the alignment
     if (unlikely(!IS_MULTIPLE_OF(size, align))) {
        return invalid_shared;
    }

    stm.size = size;
    stm.align = align;

    // Resize the first segment of the shared memory region
    stm.memory[0].resize(size / align);
    stm.memory[1].resize(size / align);
    return &stm;
}

/** Destroy (i.e. clean-up + free) a given shared memory region.
 * @param shared Shared memory region to destroy, with no running transaction
**/
void tm_destroy(shared_t unused(shared)) noexcept {
    // do nothing: static allocation
}

/** [thread-safe] Return the start address of the first allocated segment in the shared memory region.
 * @param shared Shared memory region to query
 * @return Start address of the first allocated segment
**/
void* tm_start(shared_t unused(shared)) noexcept {
    // get the address of the first segment (i.e.the segment with index 1)
    return (void*)GET_VIRTUAL_ADDRESS(1);
}

/** [thread-safe] Return the size (in bytes) of the first allocated segment of the shared memory region.
 * @param shared Shared memory region to query
 * @return First allocated segment size
**/
size_t tm_size(shared_t unused(shared)) noexcept {
    return ((stm_t*)shared)->size;
}

/** [thread-safe] Return the alignment (in bytes) of the memory accesses on the given shared memory region.
 * @param shared Shared memory region to query
 * @return Alignment used globally
**/
size_t tm_align(shared_t unused(shared)) noexcept {
    return ((stm_t*)shared)->align;
    //return stm.align;
}

/** [thread-safe] Begin a new transaction on the given shared memory region.
 * @param shared Shared memory region to start a transaction on
 * @param is_ro  Whether the transaction is read-only
 * @return Opaque transaction ID, 'invalid_tx' on failure
**/
tx_t tm_begin(shared_t unused(shared), bool is_ro) noexcept {
    transaction.is_read_only = is_ro;
    // value of the global clock at the beginning of the transaction
    transaction.read_version = stm.global_clock.load();
    return (tx_t)&transaction;
}

/** [thread-safe] End the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to end
 * @return Whether the whole transaction committed
**/
bool tm_end(shared_t unused(shared), tx_t unused(tx)) noexcept {
    stm_t* stm = (stm_t*)shared;
    transaction_t* transaction = (transaction_t*)tx;
    // if the transaction is read-only or the write set is empty,
    // no need to check the locks
    if (transaction->is_read_only || transaction->write_set.empty()) {
        COMMIT;
    }
    
    // Try to acquire the locks
    for (auto it = transaction->write_set.begin(); it != transaction->write_set.end(); it++) {
        //GET_WORD_FROM_ADDRESS(it->first);
        word_t* word = &stm->memory[GET_SEG_IDX(it->first)][GET_OFFSET(it->first, stm->align)];
        if (!acquire_lock(&word->lock)) {
            release_all_locks();
            ABORT;
        }
    }

    // Increment the global clock
    uint64_t write_version = ATOMIC_FETCH_INC(stm->global_clock);
    transaction->write_version = write_version;

    // We need to ensure that nothing has changed in terms of our
    // read set, in-between running the user's code and locking everything.
    // If read-version == write-version - 1, then no other agent were involved
    // in the transaction, so we don't need to check the read set.
    if (transaction->read_version != write_version - 1) {
        uint64_t lock;
        for (auto it = transaction->read_set.begin(); it != transaction->read_set.end(); it++) {
            lock = (*it)->load();
            // if the lock is taken or the version is greater than the read version
            if (GET_VERSION(lock) > transaction->read_version || IS_LOCKED(lock)) {
                release_all_locks();
                ABORT;
            }
        }
    }

    // Commit the transaction
    for (auto it = transaction->write_set.begin(); it != transaction->write_set.end(); it++) {
        //GET_WORD_FROM_ADDRESS(it->first);
        word_t* word = &stm->memory[GET_SEG_IDX(it->first)][GET_OFFSET(it->first, stm->align)];
        word->value = it->second;
        // release the lock and update the version
        word->lock.store(write_version << VERSION_SHIFT);
    }

    COMMIT
}

/** [thread-safe] Read operation in the given transaction, source in the shared region and target in a private region.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param source Source start address (in the shared region)
 * @param size   Length to copy (in bytes), must be a positive multiple of the alignment
 * @param target Target start address (in a private region)
 * @return Whether the whole transaction can continue
**/
bool tm_read(shared_t unused(shared), tx_t unused(tx), void const* source, size_t size, void* target) noexcept {
    stm_t* stm = (stm_t*)shared;
    transaction_t* transaction = (transaction_t*)tx;
    
    uint64_t version_before;
    for (size_t i = 0, address = (size_t)source;
        i < size / stm->align; i ++, address += stm->align) {

        if (!transaction->is_read_only) {
            auto it = transaction->write_set.find(address);
            if (it != transaction->write_set.end()) {
                *(uint64_t*)((size_t)target + i) = it->second;
                continue;
            }
        }

        // get the word
        //GET_WORD_FROM_ADDRESS(address);
        word_t* word = &stm->memory[GET_SEG_IDX(address)][GET_OFFSET(address, stm->align)];

        version_before = GET_VERSION(word->lock.load());

        // If the word version is not up-to-date, abort the transaction
        if (version_before > transaction->read_version) {
            ABORT;
        }

        *(uint64_t*)((size_t)target + i) = word->value;

        // If the version after the read is greater than the version before, 
        // then the word has been updated by another transaction.
        if (GET_VERSION(word->lock.load()) != version_before) {
            ABORT;
        }
        
        if (!transaction->is_read_only) {
            transaction->read_set.emplace(&word->lock);
            if (IS_LOCKED(word->lock.load())) {
                ABORT;
            }
        }        

    }

    return true;
}

/** [thread-safe] Write operation in the given transaction, source in a private region and target in the shared region.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param source Source start address (in a private region)
 * @param size   Length to copy (in bytes), must be a positive multiple of the alignment
 * @param target Target start address (in the shared region)
 * @return Whether the whole transaction can continue
**/
bool tm_write(shared_t unused(shared), tx_t unused(tx), void const* source, size_t size, void* target) noexcept {
    stm_t* stm = (stm_t*)shared;
    transaction_t* transaction = (transaction_t*)tx;
    for (size_t i = 0, address = (size_t)target, s = (size_t)source;
        i < size / stm->align; i++, s += stm->align, address += stm->align) {
        // put in the write set the content of source
        transaction->write_set.insert_or_assign(address, *(uint64_t*)(s));
    }

    return true;
}

/** [thread-safe] Memory allocation in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param size   Allocation requested size (in bytes), must be a positive multiple of the alignment
 * @param target Pointer in private memory receiving the address of the first byte of the newly allocated, aligned segment
 * @return Whether the whole transaction can continue (success/nomem), or not (abort_alloc)
**/
Alloc tm_alloc(shared_t unused(shared), tx_t unused(tx), size_t unused(size), void** unused(target)) noexcept {
    stm_t* stm = (stm_t*)shared;
    
    // Check if the size is a multiple of the alignment
    if (unlikely(!IS_MULTIPLE_OF(size, stm->align))) {
        return Alloc::nomem; // Maybe abort_alloc?
    }
    *target = (void*)GET_VIRTUAL_ADDRESS(ATOMIC_FETCH_INC(stm->next)); // address of the first byte of the newly allocated segment
    stm->memory[stm->next.load()].resize(size / stm->align);
    return Alloc::success;
}

/** [thread-safe] Memory freeing in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param target Address of the first byte of the previously allocated segment to deallocate
 * @return Whether the whole transaction can continue
**/
bool tm_free(shared_t unused(shared), tx_t unused(tx), void* unused(target)) noexcept {
    // no dynamic allocation
    return true;
}