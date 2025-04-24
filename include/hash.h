#ifndef pimindex_hash_h
#define pimindex_hash_h


/* https://github.com/rurban/smhasher/tree/master */

#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8) +(uint32_t)(((const uint8_t *)(d))[0]) )
#define ROTL(x, b) (uint32_t)(((x) << (b)) | ((x) >> (32 - (b))))
#define U8TO32_LE(p)                                                    \
    (((uint32_t)((p)[0])) | ((uint32_t)((p)[1]) << 8) |                 \
     ((uint32_t)((p)[2]) << 16) | ((uint32_t)((p)[3]) << 24))
#define SIPROUND                                       \
    do {                                               \
        v0 += v1;                                      \
        v1 = ROTL(v1, 5);                              \
        v1 ^= v0;                                      \
        v0 = ROTL(v0, 16);                             \
        v2 += v3;                                      \
        v3 = ROTL(v3, 8);                              \
        v3 ^= v2;                                      \
        v0 += v3;                                      \
        v3 = ROTL(v3, 7);                              \
        v3 ^= v0;                                      \
        v2 += v1;                                      \
        v1 = ROTL(v1, 13);                             \
        v1 ^= v2;                                      \
        v2 = ROTL(v2, 16);                             \
    } while (0)

uint32_t crctab[256] = {
  0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
  0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
  0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
  0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
  0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
  0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
  0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
  0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
  0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
  0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
  0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
  0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
  0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
  0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
  0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
  0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
  0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
  0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
  0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
  0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
  0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
  0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
  0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
  0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
  0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
  0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
  0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
  0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
  0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
  0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
  0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
  0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
  0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
  0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
  0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
  0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
  0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
  0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
  0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
  0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
  0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
  0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
  0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d};

uint32_t superfast(uint32_t key) {
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

uint32_t fnv32a(uint32_t a) {
  const void *key = (void*)&a;
  int len = 4;
  uint32_t seed = 0xE3CBBE91;

  uint32_t h = seed;
  const uint8_t  *data = (const uint8_t *)key;

  h ^= UINT32_C(2166136261);
  for (int i = 0; i < len; i++) {
    h ^= data[i];
    h *= 16777619;
  }
  return h;
}

uint32_t x17(uint32_t a) {
  const char *key = (char*)&a;
  int len = 4;
  uint32_t h = 0x8128E14C;

  uint8_t *data = (uint8_t *)key;
  const uint8_t *const end = &data[len];

  while (data < end) {
    h = 17 * h + (*data++ - ' ');
  }
  return h ^ (h >> 16);
}

/* the faster half 32bit variant for the linux kernel */
uint32_t siphash32(uint32_t a) {
    int len = 4;
    uint32_t seed = 0xA7A05F72;
    unsigned char	key[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    memcpy(key, &seed, sizeof(seed));
    const unsigned char *m = (unsigned char*)&a;

    uint32_t v0 = 0;
    uint32_t v1 = 0;
    uint32_t v2 = 0x6c796765;
    uint32_t v3 = 0x74656462;
    uint32_t k0 = U8TO32_LE(key);
    uint32_t k1 = U8TO32_LE(key + 8);
    uint32_t mi;
    const uint8_t *end = m + len - (len % sizeof(uint32_t));
    const int left = len & 3;
    uint32_t b = ((uint32_t)len) << 24;
    v3 ^= k1;
    v2 ^= k0;
    v1 ^= k1;
    v0 ^= k0;

    for (; m != end; m += 4) {
        mi = U8TO32_LE(m);
        v3 ^= mi;
        SIPROUND;
        SIPROUND;
        v0 ^= mi;
    }

    switch (left) {
    case 3:
        b |= ((uint32_t)m[2]) << 16;
    case 2:
        b |= ((uint32_t)m[1]) << 8;
    case 1:
        b |= ((uint32_t)m[0]);
        break;
    case 0:
        break;
    }

    v3 ^= b;
    SIPROUND;
    SIPROUND;
    v0 ^= b;
    v2 ^= 0xff;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    return v1 ^ v3;
}


/* https://github.com/gcc-mirror/gcc/blob/master/libstdc++-v3/libsupc++/hash_bytes.cc */

inline uint64_t unaligned_load(const char *p) {
    uint64_t result;
    memcpy(&result, p, sizeof(result));
    return result;
}

/* Loads n bytes, where 1 <= n < 8. */
inline uint64_t load_bytes(const char *p, int n) {
    uint64_t result = 0;
    --n;
    do
        result = (result << 8) + (unsigned char)(p[n]);
    while (--n >= 0);
    return result;
}

inline uint64_t shift_mix(uint64_t v) { return v ^ (v >> 47); }

/* Implementation of Murmur hash for 64-bit size_t. */
inline uint64_t Hash_bytes(const void *ptr, uint64_t len, uint64_t seed) {
    static const uint64_t mul = (0xc6a4a793UL << 32UL) + 0x5bd1e995UL;
    const char *const buf = (const char *)(ptr);

    // Remove the bytes not divisible by the sizeof(uint64_t).  This
    // allows the main loop to process the data as 64-bit integers.
    const int len_aligned = len & ~0x7;
    const char *const end = buf + len_aligned;
    uint64_t hash = seed ^ (len * mul);
    for (const char *p = buf; p != end; p += 8) {
        const uint64_t data = shift_mix(unaligned_load(p) * mul) * mul;
        hash ^= data;
        hash *= mul;
    }
    if ((len & 0x7) != 0) {
        const uint64_t data = load_bytes(end, len & 0x7);
        hash ^= data;
        hash *= mul;
    }
    hash = shift_mix(hash) * mul;
    hash = shift_mix(hash);
    return hash;
}

inline uint64_t murmur(const void *_ptr, uint64_t _len, uint64_t _seed) {
    return Hash_bytes(_ptr, _len, _seed);
}

static uint64_t (*hash_funcs[1])(const void *key, uint64_t len, uint64_t seed) = {murmur};

static inline uint64_t h(const void *key, uint64_t len, uint64_t seed) {
    return hash_funcs[0](key, len, seed);
}

uint64_t murmur64(uint32_t key) {
    return h(&key, sizeof(uint32_t), 0xc70697UL);
}


/* https://github.com/rurban/smhasher/tree/master */

uint32_t sdbm(uint32_t a) {
  int len = 4;
  uint32_t hash = 0x582AF769;
  const char* key = (char*)&a;
  unsigned char  *str = (unsigned char *)key;
  const unsigned char *const end = (const unsigned char *)str + len;
  //note that perl5 adds the seed to the end of key, which looks like cargo cult
  while (str < end) {
    hash = (hash << 6) + (hash << 16) - hash + *str++;
  }
  return hash;
}

uint32_t goodoaat(uint32_t a) {
#define grol(x,n) (((x)<<(n))|((x)>>(32-(n))))
#define gror(x,n) (((x)>>(n))|((x)<<(32-(n))))
  int len = 4;
  uint32_t seed = 0x7B14EEE5;
  const char* key = (char*)&a;
  unsigned char  *str = (unsigned char *)key;
  const unsigned char *const end = (const unsigned char *)str + len;
  uint32_t h1 = seed ^ 0x3b00;
  uint32_t h2 = grol(seed, 15);
  for (;str != end; str++) {
    h1 += str[0];
    h1 += h1 << 3; // h1 *= 9
    h2 += h1;
    // the rest could be as in MicroOAAT: h1 = grol(h1, 7)
    // but clang doesn't generate ROTL instruction then.
    h2 = grol(h2, 7);
    h2 += h2 << 2; // h2 *= 5
  }
  h1 ^= h2;
  /* now h1 passes all collision checks,
   * so it is suitable for hash-tables with prime numbers. */
  h1 += grol(h2, 14);
  h2 ^= h1; h2 += gror(h1, 6);
  h1 ^= h2; h1 += grol(h2, 5);
  h2 ^= h1; h2 += gror(h1, 8);
  return h2;
#undef grol
#undef gror
}

uint32_t crc32(uint32_t key) {
    // tabulation    
    uint32_t hash = 0;
    // key = key >> 4;
    hash = (0xd9d65adc & crctab[((key & 0xff))])
        ^ crctab[((key & 0xff00) >> 8)] 
        ^ crctab[((key & 0xff0000) >> 16)] 
        ^ crctab[((key & 0xff000000) >> 24)]; 
    return hash;

}


/* http://burtleburtle.net/bob/hash/integer.html */

uint32_t key_to_tasklet_hash(uint32_t key) {
  key -= (key<<6);
  key ^= (key>>17);
  key -= (key<<9);
  key ^= (key<<4);
  key -= (key<<3);
  key ^= (key<<10);
  key ^= (key>>15);
  return key;
}

#endif  /* #ifndef pimindex_hash_h */
