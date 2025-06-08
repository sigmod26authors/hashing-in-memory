
// Copyright (c) Simon Fraser University & The Chinese University of Hong Kong. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <immintrin.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <stdint.h>

#include <cstdint>
#include <iostream>

#define IS_POWER_OF_TWO(x) (x && (x & (x - 1)) == 0)

static constexpr const uint32_t kCacheLineSize = 64;

static bool FileExists(const char *pool_path) {
  struct stat buffer;
  return (stat(pool_path, &buffer) == 0);
}

#ifdef PMEM
#define CREATE_MODE_RW (S_IWUSR | S_IRUSR)
#endif

#define LOG_FATAL(msg)      \
  std::cout << msg << "\n"; \
  exit(-1)

#define LOG(msg) std::cout << msg << "\n"

#define CAS(_p, _u, _v)                                             \
  (__atomic_compare_exchange_n(_p, _u, _v, false, __ATOMIC_ACQUIRE, \
                               __ATOMIC_ACQUIRE))

// ADD and SUB return the value after add or sub
#define ADD(_p, _v) (__atomic_add_fetch(_p, _v, __ATOMIC_SEQ_CST))
#define SUB(_p, _v) (__atomic_sub_fetch(_p, _v, __ATOMIC_SEQ_CST))
#define LOAD(_p) (__atomic_load_n(_p, __ATOMIC_SEQ_CST))
#define STORE(_p, _v) (__atomic_store_n(_p, _v, __ATOMIC_SEQ_CST))

#define SIMD 1
#define bitScan(x)  __builtin_ffs(x)
#define SIMD_CMP8(src, key)                                         \
  do {                                                              \
    const __m256i key_data = _mm256_set1_epi8(key);                 \
    __m256i seg_data =                                              \
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src)); \
    __m256i rv_mask = _mm256_cmpeq_epi8(seg_data, key_data);        \
    mask = _mm256_movemask_epi8(rv_mask);                           \
  } while (0)

#define SSE_CMP8(src, key)                                       \
  do {                                                           \
    const __m128i key_data = _mm_set1_epi8(key);                 \
    __m128i seg_data =                                           \
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)); \
    __m128i rv_mask = _mm_cmpeq_epi8(seg_data, key_data);        \
    mask = _mm_movemask_epi8(rv_mask);                           \
  } while (0)

#define MASK_CMP8(src, key)                                      \
  do {                                                           \
    const __m128i key_data = _mm_set1_epi8(key);                 \
    __m128i seg_data =                                           \
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)); \
    mask = _mm_cmpeq_epi8_mask(seg_data, key_data);              \
  } while (0)

#define CHECK_BIT(var, pos) ((((var) & (1 << pos)) > 0) ? (1) : (0))

inline void mfence(void) { asm volatile("mfence" ::: "memory"); }

int msleep(uint64_t msec) {
  struct timespec ts;
  int res;

  if (msec < 0) {
    errno = EINVAL;
    return -1;
  }

  ts.tv_sec = msec / 1000;
  ts.tv_nsec = (msec % 1000) * 1000000;

  do {
    res = nanosleep(&ts, &ts);
  } while (res && errno == EINTR);

  return res;
}

inline uint64_t Murmur3_64(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccd;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33;
  return h;
}

#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8) +(uint32_t)(((const uint8_t *)(d))[0]) )

uint32_t key_to_dpu_hash(uint32_t key) {
  int len = 4;
  uint8_t* data_;
  uint8_t data[4];
  data_ = data;
  *(uint32_t*)(data) = key;
  uint32_t hash = 4, tmp;
	int rem;

  rem = len & 3;
  len >>= 2;

  /* Main loop */
  for (;len > 0; len--) {
      hash  += get16bits (data);
      tmp    = (get16bits (data+2) << 11) ^ hash;
      hash   = (hash << 16) ^ tmp;
      data_  += 2*sizeof (uint16_t);
      hash  += hash >> 11;
  }

  /* Handle end cases */
  switch (rem) {
      case 3: hash += get16bits (data);
    hash ^= hash << 16;
    hash ^= data[sizeof (uint16_t)] << 18;
    hash += hash >> 11;
    break;
      case 2: hash += get16bits (data);
    hash ^= hash << 11;
    hash += hash >> 17;
    break;
      case 1: hash += *data;
    hash ^= hash << 10;
    hash += hash >> 1;
  }

  /* Force "avalanching" of final 127 bits */
  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 4;
  hash += hash >> 17;
  hash ^= hash << 25;
  hash += hash >> 6;

  return hash;

}
