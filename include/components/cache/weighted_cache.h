#ifndef WEIGHTED_CACHE_H
#define WEIGHTED_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <unordered_map>

namespace layers {

    /**
     * @brief Cache per layer con pesi (Dense, Conv2D, RNN, etc.)
     * 
     * Questa cache memorizza non solo input/output ma anche:
     * - Pesi e bias del layer
     * - Gradienti calcolati durante il backward
     * - Statistiche per ottimizzatori (momentum, Adam, etc.)
     * - Versioni dei pesi per checkpoint
     */
    class WeightedCache : public LayerCache {
    public:
        struct WeightGradient {
            Eigen::MatrixXd weights;
            Eigen::MatrixXd gradient;
            int step;
            
            WeightGradient() : step(0) {}
            WeightGradient(const Eigen::MatrixXd& w, const Eigen::MatrixXd& g, int s) 
                : weights(w), gradient(g), step(s) {}
        };
        
        struct OptimizerState {
            // Per momentum
            Eigen::MatrixXd momentum;
            // Per Adam
            Eigen::MatrixXd first_moment;
            Eigen::MatrixXd second_moment;
            int timestep;
            
            OptimizerState() : timestep(0) {}
        };
        
        WeightedCache();
        ~WeightedCache() override = default;
        
        // Implementazione interfaccia LayerCache
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "WeightedCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return true; }
        
        // Getters/Setters per input/output
        void set_input(const Eigen::MatrixXd& in) { input_cache = in; }
        void set_output(const Eigen::MatrixXd& out) { output_cache = out; }
        void set_z(const Eigen::MatrixXd& z_val) { z_cache = z_val; }
        
        const Eigen::MatrixXd& get_z() const { return z_cache; }
        
        // Gestione pesi
        void set_weights(const Eigen::MatrixXd& w) { weights_ = w; }
        void set_biases(const Eigen::VectorXd& b) { biases_ = b; }
        
        const Eigen::MatrixXd& get_weights() const { return weights_; }
        const Eigen::VectorXd& get_biases() const { return biases_; }
        
        Eigen::MatrixXd& mutable_weights() { return weights_; }
        Eigen::VectorXd& mutable_biases() { return biases_; }
        
        // Gestione gradienti
        void set_weight_gradient(const Eigen::MatrixXd& grad) { weight_gradient_ = grad; }
        void set_bias_gradient(const Eigen::VectorXd& grad) { bias_gradient_ = grad; }
        
        const Eigen::MatrixXd& get_weight_gradient() const { return weight_gradient_; }
        const Eigen::VectorXd& get_bias_gradient() const { return bias_gradient_; }
        
        Eigen::MatrixXd& mutable_weight_gradient() { return weight_gradient_; }
        Eigen::VectorXd& mutable_bias_gradient() { return bias_gradient_; }
        
        // Storico gradienti per ottimizzatori che ne hanno bisogno
        void push_gradient_history(const Eigen::MatrixXd& grad);
        const std::vector<Eigen::MatrixXd>& get_gradient_history() const { return gradient_history_; }
        void clear_gradient_history();
        
        // Stato ottimizzatore
        OptimizerState& get_optimizer_state(const std::string& optimizer_name);
        bool has_optimizer_state(const std::string& optimizer_name) const;
        
        // Checkpointing
        void save_checkpoint(int step);
        bool load_checkpoint(int step, Eigen::MatrixXd& w, Eigen::VectorXd& b);
        void clear_checkpoints();
        
        // Versionamento pesi
        void set_weight_version(int version) { weight_version_ = version; }
        int get_weight_version() const { return weight_version_; }
        
        // Regularizzazione
        void set_regularization_loss(double loss) { reg_loss_ = loss; }
        double get_regularization_loss() const { return reg_loss_; }
        
        // Accesso modificabile (per layer che devono popolare la cache)
        Eigen::MatrixXd& mutable_input() { return input_cache; }
        Eigen::MatrixXd& mutable_z() { return z_cache; }
        Eigen::MatrixXd& mutable_output() { return output_cache; }

    private:
        bool validate_dimensions() const;
        
        // Dati di forward/backward
        Eigen::MatrixXd input_cache;
        Eigen::MatrixXd z_cache;
        Eigen::MatrixXd output_cache;
        
        // Pesi e bias
        Eigen::MatrixXd weights_;
        Eigen::VectorXd biases_;
        
        // Gradienti
        Eigen::MatrixXd weight_gradient_;
        Eigen::VectorXd bias_gradient_;
        
        // Storico gradienti (per alcuni ottimizzatori)
        std::vector<Eigen::MatrixXd> gradient_history_;
        
        // Stati ottimizzatori (per layer condivisi tra batch)
        std::unordered_map<std::string, OptimizerState> optimizer_states_;
        
        // Checkpoint dei pesi
        std::unordered_map<int, WeightGradient> checkpoints_;
        
        // Metadati
        int weight_version_;
        double reg_loss_;
        bool training_;
    };

} // namespace layers

#endif
