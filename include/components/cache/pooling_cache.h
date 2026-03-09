#ifndef POOLING_CACHE_H
#define POOLING_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

    class PoolingCache : public LayerCache {
    public:
        struct MaxIndex {
            int batch;
            int channel;
            int output_row;
            int output_col;
            int input_h;
            int input_w;
            
            MaxIndex() : batch(0), channel(0), output_row(0), output_col(0), 
                        input_h(0), input_w(0) {}
            MaxIndex(int b, int c, int orow, int ocol, int ih, int iw)
                : batch(b), channel(c), output_row(orow), output_col(ocol),
                  input_h(ih), input_w(iw) {}
        };
        
        PoolingCache();
        ~PoolingCache() override = default;
        
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "PoolingCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input; }
        const Eigen::MatrixXd& get_output() const override { return output; }
        bool has_activation() const override { return false; }
        
        void set_input(const Eigen::MatrixXd& in) { input = in; }
        void set_output(const Eigen::MatrixXd& out) { output = out; }
        void set_training(bool t) { training_ = t; }
        void set_input_shape(int h, int w, int c) {
            input_height = h;
            input_width = w;
            channels = c;
        }
        
        bool get_training() const { return training_; }
        const std::vector<MaxIndex>& get_max_indices() const { return max_indices_; }
        
        void add_max_index(int batch, int channel, int orow, int ocol, int ih, int iw) {
            max_indices_.emplace_back(batch, channel, orow, ocol, ih, iw);
        }
        
        Eigen::MatrixXd& mutable_input() { return input; }
        Eigen::MatrixXd& mutable_output() { return output; }
        std::vector<MaxIndex>& mutable_max_indices() { return max_indices_; }

    private:
        bool validate_dimensions() const;
        
        Eigen::MatrixXd input;
        Eigen::MatrixXd output;
        bool training_;
        
        int input_height;
        int input_width;
        int channels;
        
        std::vector<MaxIndex> max_indices_;
    };

} // namespace layers

#endif

