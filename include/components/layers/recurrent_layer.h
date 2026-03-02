#ifndef RECURRENT_LAYER_H
#define RECURRENT_LAYER_H

#include "layer.h"
#include "components/cache/recurrent_cache.h"
#include <memory>

namespace layers {

    class RecurrentLayer : public Layer {
    public:
        RecurrentLayer(int hidden_size, int input_size);
        virtual ~RecurrentLayer() = default;
        
        // Metodi comuni a tutti i ricorrenti
        virtual void set_input_shape(int input_size) = 0;
        virtual void set_sequence_length(int seq_len) { sequence_length_ = seq_len; }
        virtual int get_sequence_length() const { return sequence_length_; }
        virtual int get_hidden_size() const { return hidden_size_; }
        
        // Gestione stato iniziale
        virtual void set_initial_state(const Eigen::MatrixXd& h0);
        virtual Eigen::MatrixXd get_initial_state() const { return h0_; }
        
        // Layer interface (alcuni possono restare puri virtuali)
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override = 0;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override = 0;
        
        // Serializzazione (gestisce parametri comuni)
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        // Info
        std::string get_type() const override = 0;  // Ancora puro virtuale
        int get_parameter_count() const override;
        
        // Cache (polimorfica!)
        const LayerCache& get_cache() const override { return *cache_; }
        
    protected:
        int hidden_size_;
        int input_size_;
        int sequence_length_;
        Eigen::MatrixXd h0_;  // stato iniziale
        
        // Cache polimorfica (puntatore alla classe base)
        std::unique_ptr<RecurrentCache> cache_;
        
        // Utility comuni
        virtual void initialize_cache(int batch_size);
        Eigen::MatrixXd extract_timestep(const Eigen::MatrixXd& input, int t) const;
    };

} // namespace layers

#endif