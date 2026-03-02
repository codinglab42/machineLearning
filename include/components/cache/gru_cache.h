#ifndef GRU_CACHE_H
#define GRU_CACHE_H

#include "recurrent_cache.h"

namespace layers {

    class GRUCache : public RecurrentCache {
    public:
        GRUCache();
        ~GRUCache() override = default;
        
        void init(int sequence_length, int batch_size, int hidden_size) override;
        void add_state(int timestep, const Eigen::MatrixXd& hidden_state, 
                      const Eigen::MatrixXd& input) override;  // non basta per GRU!
        
        // Metodo specifico GRU
        void add_gru_state(int timestep,
                          const Eigen::MatrixXd& hidden_state,
                          const Eigen::MatrixXd& input,
                          const Eigen::MatrixXd& update_gate,
                          const Eigen::MatrixXd& reset_gate,
                          const Eigen::MatrixXd& new_gate);
        
        Eigen::MatrixXd get_hidden_state(int timestep) const override;
        
        int get_sequence_length() const override { return seq_len_; }
        int get_hidden_size() const override { return hidden_size_; }
        
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "GRUCache"; }

    private:
        struct GRUStep {
            Eigen::MatrixXd h;
            Eigen::MatrixXd x;
            Eigen::MatrixXd z;  // update gate
            Eigen::MatrixXd r;  // reset gate
            Eigen::MatrixXd n;  // new gate
        };
        
        std::vector<GRUStep> steps_;
        int seq_len_;
        int batch_size_;
        int hidden_size_;
    };

} // namespace layers

#endif
