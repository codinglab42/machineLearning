#ifndef POOLING_CACHE_H
#define POOLING_CACHE_H

#include "basic_cache.h"
#include <vector>

namespace layers {

    class PoolingCache : public BasicCache {
    public:
        struct MaxIndex {
            int batch;
            int channel;
            int output_row;
            int output_col;
            int input_index;
            
            MaxIndex();
            MaxIndex(int b, int c, int orow, int ocol, int idx);
        };
        
        PoolingCache();
        ~PoolingCache() override = default;
        
        void clear() override;
        std::string get_type() const override { return "PoolingCache"; }
        
        // Metodi specifici per pooling
        void set_input_shape(int height, int width, int channels);
        void add_max_index(const MaxIndex& index);
        const std::vector<MaxIndex>& get_max_indices() const { return max_indices_; }
        
        int get_input_height() const { return input_height_; }
        int get_input_width() const { return input_width_; }
        int get_channels() const { return channels_; }
        
        // Utility
        int calculate_input_index(int batch, int channel, int h, int w) const;

    private:
        std::vector<MaxIndex> max_indices_;
        int input_height_;
        int input_width_;
        int channels_;
    };

} // namespace layers

#endif
