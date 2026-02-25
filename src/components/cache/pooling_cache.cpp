#include "components/cache/pooling_cache.h"

namespace layers {

    // MaxIndex implementation
    PoolingCache::MaxIndex::MaxIndex() 
        : batch(0), channel(0), output_row(0), output_col(0), input_index(-1) {}

    PoolingCache::MaxIndex::MaxIndex(int b, int c, int orow, int ocol, int idx) 
        : batch(b), channel(c), output_row(orow), output_col(ocol), input_index(idx) {}

    // PoolingCache implementation
    PoolingCache::PoolingCache() 
        : BasicCache(), input_height_(0), input_width_(0), channels_(0) {}

    void PoolingCache::clear() {
        BasicCache::clear();
        max_indices_.clear();
        input_height_ = 0;
        input_width_ = 0;
        channels_ = 0;
    }

    void PoolingCache::set_input_shape(int height, int width, int channels) {
        input_height_ = height;
        input_width_ = width;
        channels_ = channels;
    }

    void PoolingCache::add_max_index(const MaxIndex& index) {
        max_indices_.push_back(index);
    }

    int PoolingCache::calculate_input_index(int batch, int channel, int h, int w) const {
        return batch * (channels_ * input_height_ * input_width_) 
            + channel * (input_height_ * input_width_) 
            + h * input_width_ + w;
    }

} // namespace layers
