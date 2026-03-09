#ifndef CONV_CACHE_H
#define CONV_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <string>

namespace layers {

    class ConvCache : public LayerCache {
    public:
        ConvCache();
        ~ConvCache() override = default;
        
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "ConvCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return true; }
        
        Eigen::MatrixXd input_cache;
        Eigen::MatrixXd z_cache;
        Eigen::MatrixXd output_cache;
        Eigen::MatrixXd col_cache;
        
        int input_height;
        int input_width;
        int input_channels;
        int output_height;
        int output_width;
        int filters;
        int batch_size;
        
        int kernel_size;
        int strides;
        std::string padding;
        
        void set_input_shape(int h, int w, int c);
        void set_output_shape(int h, int w, int f);
        void set_batch_size(int bs);
        void set_kernel_info(int k_size, int str, const std::string& pad);
        
        Eigen::MatrixXd& mutable_input() { return input_cache; }
        Eigen::MatrixXd& mutable_z() { return z_cache; }
        Eigen::MatrixXd& mutable_output() { return output_cache; }
        Eigen::MatrixXd& mutable_col() { return col_cache; }

    private:
        bool validate_dimensions() const;
    };

} // namespace layers

#endif

