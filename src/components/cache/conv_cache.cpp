#include "components/cache/conv_cache.h"
#include "exceptions/exception_macros.h"

namespace layers {

    ConvCache::ConvCache()
        : input_height(0), input_width(0), input_channels(0),
          output_height(0), output_width(0), filters(0),
          batch_size(0), kernel_size(0), strides(1), padding("valid") {
        
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
        col_cache.resize(0, 0);
    }

    void ConvCache::clear() {
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
        col_cache.resize(0, 0);
        
        input_height = 0;
        input_width = 0;
        input_channels = 0;
        output_height = 0;
        output_width = 0;
        filters = 0;
        batch_size = 0;
        kernel_size = 0;
        strides = 1;
        padding = "valid";
    }

    bool ConvCache::is_valid() const {
        if (input_cache.size() == 0 || output_cache.size() == 0) {
            return false;
        }
        return validate_dimensions();
    }

    bool ConvCache::validate_dimensions() const {
        int expected_input_size = batch_size * input_height * input_width * input_channels;
        if (input_cache.rows() != expected_input_size || input_cache.cols() != 1) {
            return false;
        }
        
        int expected_output_size = batch_size * output_height * output_width * filters;
        if (output_cache.rows() != expected_output_size || output_cache.cols() != 1) {
            return false;
        }
        
        if (z_cache.rows() != expected_output_size || z_cache.cols() != 1) {
            return false;
        }
        
        int expected_cols = output_height * output_width;
        int expected_col_features = kernel_size * kernel_size * input_channels;
        if (col_cache.rows() != batch_size || 
            col_cache.cols() != expected_cols * expected_col_features) {
            return false;
        }
        
        return true;
    }

    void ConvCache::set_input_shape(int h, int w, int c) {
        input_height = h;
        input_width = w;
        input_channels = c;
    }

    void ConvCache::set_output_shape(int h, int w, int f) {
        output_height = h;
        output_width = w;
        filters = f;
    }

    void ConvCache::set_batch_size(int bs) {
        batch_size = bs;
    }

    void ConvCache::set_kernel_info(int k_size, int str, const std::string& pad) {
        kernel_size = k_size;
        strides = str;
        padding = pad;
    }

} // namespace layers

