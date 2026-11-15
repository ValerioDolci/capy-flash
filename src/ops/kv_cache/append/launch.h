#pragma once

#include "ninfer/ops/kv_cache_append.h"

namespace ninfer::ops::detail {

void kv_cache_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                            PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_nvfp4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_nvfp4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        PagedKVBatchLayerView cache, cudaStream_t stream);

void kv_cache_append_k8v4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                 PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_k8v4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       const Tensor& valid_columns, const Tensor& table_rows,
                                       PagedKVBatchLayerView cache, cudaStream_t stream);

void kv_cache_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  PagedKVBatchLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops::detail
