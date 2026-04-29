// include/components/cache/conv_cache.h
#ifndef CONV_CACHE_H
#define CONV_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <string>

namespace layers {

class ConvCache : public LayerCache {
public:
    ConvCache() 
        : input_height(0), input_width(0), input_channels(0),
          output_height(0), output_width(0), filters(0),
          batch_size(0), kernel_size(0), strides(1), padding("valid") {
        
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
        col_cache.resize(0, 0);
    }
    
    ~ConvCache() override = default;
    
    void clear() override {
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
    
    bool is_valid() const override {
        return input_cache.size() > 0 && output_cache.size() > 0;
    }
    
    std::string get_type() const override { return "ConvCache"; }
    
    const Eigen::MatrixXd& get_input() const override { return input_cache; }
    const Eigen::MatrixXd& get_output() const override { return output_cache; }
    
    // Dati temporanei
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd z_cache;
    Eigen::MatrixXd output_cache;
    Eigen::MatrixXd col_cache;
    
    // Shape info
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
    
    // Metodi helper
    void set_input_shape(int h, int w, int c) {
        input_height = h;
        input_width = w;
        input_channels = c;
    }
    
    void set_output_shape(int h, int w, int f) {
        output_height = h;
        output_width = w;
        filters = f;
    }
    
    void set_batch_size(int bs) { batch_size = bs; }
    
    void set_kernel_info(int k_size, int str, const std::string& pad) {
        kernel_size = k_size;
        strides = str;
        padding = pad;
    }
    
    Eigen::MatrixXd& mutable_input() { return input_cache; }
    Eigen::MatrixXd& mutable_z() { return z_cache; }
    Eigen::MatrixXd& mutable_output() { return output_cache; }
    Eigen::MatrixXd& mutable_col() { return col_cache; }
};

} // namespace layers

#endif