#ifndef RECURRENT_LAYER_H
#define RECURRENT_LAYER_H

#include "layer.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace layers {

    struct SequenceTensor {
        std::vector<double> data;
        int batch_size = 0;
        int time_steps = 0;
        int features = 0;
        bool is_sequence = true;

        void validate() const;
    };

    class RecurrentLayer : public Layer {
    public:
        RecurrentLayer() = default;
        ~RecurrentLayer() override = default;

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override = 0;
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override = 0;
        virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override= 0;
        
        virtual void serialize(std::ostream& out) const override = 0;
        virtual void deserialize(std::istream& in) override = 0;
        
        virtual std::string get_type() const override = 0;
        virtual std::string get_config() const override = 0;
        
        virtual bool has_weights() const override = 0;
        virtual void initialize_weights() override = 0;
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

        void set_return_sequences(bool value) { return_sequences_ = value; }
        bool get_return_sequences() const { return return_sequences_; }
        bool last_forward_was_sequence() const { return last_forward_was_sequence_; }

        SequenceTensor forward_sequence(const SequenceTensor& input, bool training = false);
        SequenceTensor backward_sequence(const SequenceTensor& gradient);

    protected:
        void clear_sequence_cache() {
            sequence_caches_.clear();
            sequence_time_steps_ = 0;
            last_forward_was_sequence_ = false;
        }
        virtual Eigen::MatrixXd backward_step(const Eigen::MatrixXd& gradient,
                                              const Eigen::MatrixXd& cell_gradient,
                                              Eigen::MatrixXd& previous_hidden_gradient,
                                              Eigen::MatrixXd& previous_cell_gradient) = 0;
        bool return_sequences_ = false;

    private:
        std::vector<std::shared_ptr<LayerCache>> sequence_caches_;
        int sequence_batch_size_ = 0;
        int sequence_time_steps_ = 0;
        int sequence_input_size_ = 0;
        bool sequence_returns_all_ = false;
        bool last_forward_was_sequence_ = false;
    };

} // namespace layers

#endif

