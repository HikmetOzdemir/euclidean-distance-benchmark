// l2.c
// Batch L2 kernel for float32 vectors.
// Build (Linux/macOS):
//   cc -O3 -march=native -ffast-math -fPIC -shared l2.c -o libl2.so   (Linux)
//   cc -O3 -march=native -ffast-math -fPIC -shared -undefined dynamic_lookup l2.c -o libl2.dylib (macOS)
// Windows (example with MSVC, adapt as needed):
//   cl /O2 /LD l2.c /Fel2.dll

#include <math.h>
#include <stddef.h>

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void l2_batch(const float* a,
                     const float* b,
                     int n_dim,
                     int n_vecs,
                     float* out)
{
    const int dim = n_dim;
    for (int v = 0; v < n_vecs; ++v) {
        const float *pa = a + (size_t)v * dim;
        const float *pb = b + (size_t)v * dim;

        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float d = pa[i] - pb[i];
            acc += d * d;
        }
        out[v] = sqrtf(acc);
    }
}
