#ifndef RECURRENT_LAYER_H
#define RECURRENT_LAYER_H

#include "layer.h"
#include <Eigen/Dense>
#include <memory>

namespace layers {

    class RecurrentLayer : public Layer {
    public:
        RecurrentLayer() = default;
        ~RecurrentLayer() override = default;

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override = 0;
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override = 0;
        virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override = 0;
        
        virtual void serialize(std::ostream& out) const override = 0;
        virtual void deserialize(std::istream& in) override = 0;
        
        virtual std::string get_type() const override = 0;
        virtual std::string get_config() const override = 0;
        
        virtual bool has_weights() const override = 0;
        virtual Eigen::MatrixXd get_weights() const override = 0;
        virtual void set_weights(const Eigen::MatrixXd& weights) override = 0;
        virtual int get_parameter_count() const override = 0;
        
        virtual int get_input_size() const override = 0;
        virtual int get_output_size() const override = 0;
        
        virtual void clear_cache() override = 0;
        virtual std::shared_ptr<LayerCache> get_cache() const override = 0;
        virtual void set_cache(std::shared_ptr<LayerCache> cache) override = 0;
        
        virtual Eigen::VectorXd get_biases() const override = 0;
        virtual void set_biases(const Eigen::VectorXd& biases) override = 0;
        
        virtual void set_input_shape(int input_size) override = 0;
        
        // Metodi specifici per layer ricorrenti
        virtual void reset_state() = 0;
        virtual Eigen::MatrixXd get_hidden_state() const = 0;
    };

} // namespace layers

#endif

