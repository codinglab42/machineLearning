#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"
#include "components/cache/pooling_cache.h"
#include "Eigen/Dense"
#include <vector>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace layers {

    // Struttura per memorizzare le dimensioni dell'input
    struct InputShape {
        int batch;
        int channels;
        int height;
        int width;
        
        InputShape() : batch(0), channels(0), height(0), width(0) {}
        InputShape(int b, int c, int h, int w) : batch(b), channels(c), height(h), width(w) {}
        
        int total_elements() const { return batch * channels * height * width; }
        bool is_valid() const { return batch > 0 && channels > 0 && height > 0 && width > 0; }
    };

    class Pooling : public Layer {
    public:
        enum PoolType { MAX, AVERAGE };
        
        // Costruttore
        Pooling(int pool_size = 2, int stride = 2, 
                PoolType type = MAX, int channels = 1);
        
        // Layer interface
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override;
        
        // Nuovi metodi per gestire le dimensioni
        void set_input_shape(int height, int width) { input_height_ = height; input_width_ = width; }
        void set_input_shape(const InputShape& shape) { 
            input_shape_ = shape; 
            input_height_ = shape.height;
            input_width_ = shape.width;
        }
        InputShape get_input_shape() const { return input_shape_; }
        
        // Metodi per calcolare output shape
        int get_output_height() const;
        int get_output_width() const;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "Pooling"; }
        std::string get_config() const override;
        int get_input_size() const override;
        int get_output_size() const override;
        int get_parameter_count() const override { return 0; }
        
        void clear_cache() override;
        const LayerCache& get_cache() const override { return cache_; }
        
        bool has_weights() const override { return false; }
        Eigen::MatrixXd get_weights() const override { return MatrixXd(); }
        Eigen::VectorXd get_biases() const override { return VectorXd(); }
        void set_weights(const Eigen::MatrixXd& weights) override {}
        void set_biases(const Eigen::VectorXd& biases) override {}
        
        // Metodi specifici
        void set_pool_type(PoolType type) { pool_type_ = type; }
        PoolType get_pool_type() const { return pool_type_; }

        // Aggiungi temporaneamente per debug
        void print_debug() const {
            std::cout << "DEBUG: height=" << input_height_ 
                    << ", width=" << input_width_ << std::endl;
        }
        
    private:
        int pool_size_;
        int stride_;
        int channels_;
        PoolType pool_type_;
        
        // Dimensioni dell'input (memorizzate dal forward)
        int input_height_;
        int input_width_;
        InputShape input_shape_;
        
        // Cache per backward (indici per max pooling)
        PoolingCache cache_;
        
        // Metodi privati
        std::vector<int> calculate_output_shape(const std::vector<int>& input_shape) const;
        MatrixXd pool_2d(const MatrixXd& input, int channel, int batch);
        MatrixXd pool_backward_2d(const MatrixXd& gradient, int channel, int batch);
        
        // Utility per convertire tra flatten e 2D
        MatrixXd flatten_to_2d(const MatrixXd& flatten, int batch, int channel) const;
        MatrixXd extract_channel(const MatrixXd& input, int batch, int channel) const;
        void insert_channel(MatrixXd& output, int batch, int channel, const MatrixXd& channel_data) const;
        
        // Validazione dimensioni
        void validate_input_shape(int total_elements) const;
    };

} // namespace layers

#endif
