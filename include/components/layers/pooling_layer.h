#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"
#include "components/cache/pooling_cache.h"
#include <Eigen/Dense>
#include <string>

namespace layers {

    class PoolingLayer : public Layer {
    public:
        enum PoolType { MAX, AVG };
        
        PoolingLayer(int pool_size = 2, int stride = 2, 
                PoolType type = MAX, int channels = 1);
        ~PoolingLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "Pooling"; }
        LayerType get_layer_type() const override { return LayerType::MAX_POOLING; }
        std::string get_config() const override;
        uint32_t get_version() const override { return 1; }
        
        bool has_weights() const override { return false; }
        Eigen::MatrixXd get_weights() const override { return Eigen::MatrixXd(); }
        void set_weights(const Eigen::MatrixXd& weights) override {}
        int get_parameter_count() const override { return 0; }
        
        int get_input_size() const override;
        int get_output_size() const override;
        
        void clear_cache() override { if (cache_) cache_->clear(); }
        std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<LayerCache> cache) override { 
            cache_ = std::dynamic_pointer_cast<PoolingCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override { return Eigen::VectorXd(); }
        void set_biases(const Eigen::VectorXd& biases) override {}
        void set_input_shape(int input_size) override;

    private:
        std::shared_ptr<PoolingCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<PoolingCache>(cache_);
        }
        
        int get_output_height() const;
        int get_output_width() const;
        void compute_output_dimensions();
        Eigen::MatrixXd extract_channel(const Eigen::MatrixXd& input, int batch, int channel) const;
        void insert_channel(Eigen::MatrixXd& output, int batch, int channel, const Eigen::MatrixXd& channel_data) const;
        
        int pool_size_;
        int stride_;
        int channels_;
        PoolType pool_type_;
        
        int input_height_;
        int input_width_;
        int input_size_;
        
        std::shared_ptr<PoolingCache> cache_;
    };

} // namespace layers

#endif
