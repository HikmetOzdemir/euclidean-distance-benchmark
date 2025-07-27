#ifndef L2_H
#define L2_H

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT void l2_batch(const float* a,
                     const float* b,
                     int n_dim,
                     int n_vecs,
                     float* out);

#ifdef __cplusplus
}
#endif

#endif // L2_H
