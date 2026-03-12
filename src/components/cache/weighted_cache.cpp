#include "components/cache/weighted_cache.h"
#include "exceptions/exception_macros.h"
#include <algorithm>

namespace layers {

    WeightedCache::WeightedCache() 
        : weight_version_(0), reg_loss_(0.0), training_(false) {
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
        weights_.resize(0, 0);
        biases_.resize(0);
        weight_gradient_.resize(0, 0);
        bias_gradient_.resize(0);
        gradient_history_.clear();
        optimizer_states_.clear();
        checkpoints_.clear();
    }

    void WeightedCache::clear() {
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
        weights_.resize(0, 0);
        biases_.resize(0);
        weight_gradient_.resize(0, 0);
        bias_gradient_.resize(0);
        gradient_history_.clear();
        optimizer_states_.clear();
        checkpoints_.clear();
        weight_version_ = 0;
        reg_loss_ = 0.0;
        training_ = false;
    }

    bool WeightedCache::is_valid() const {
        if (input_cache.size() == 0 || output_cache.size() == 0) {
            return false;
        }
        return validate_dimensions();
    }

    bool WeightedCache::validate_dimensions() const {
        // Verifica consistenza dimensioni pesi e bias
        if (weights_.size() > 0) {
            if (biases_.size() > 0 && weights_.cols() != biases_.size()) {
                return false;
            }
        }
        
        // Verifica gradienti
        if (weight_gradient_.size() > 0) {
            if (weight_gradient_.rows() != weights_.rows() || 
                weight_gradient_.cols() != weights_.cols()) {
                return false;
            }
        }
        
        if (bias_gradient_.size() > 0) {
            if (bias_gradient_.size() != biases_.size()) {
                return false;
            }
        }
        
        return true;
    }

    void WeightedCache::push_gradient_history(const Eigen::MatrixXd& grad) {
        gradient_history_.push_back(grad);
        // Mantieni solo ultimi 100 gradienti per memoria
        const size_t MAX_HISTORY = 100;
        if (gradient_history_.size() > MAX_HISTORY) {
            gradient_history_.erase(gradient_history_.begin());
        }
    }

    void WeightedCache::clear_gradient_history() {
        gradient_history_.clear();
    }

    WeightedCache::OptimizerState& WeightedCache::get_optimizer_state(const std::string& optimizer_name) {
        auto it = optimizer_states_.find(optimizer_name);
        if (it == optimizer_states_.end()) {
            OptimizerState new_state;
            // Inizializza con dimensioni appropriate
            if (weights_.size() > 0) {
                new_state.momentum = Eigen::MatrixXd::Zero(weights_.rows(), weights_.cols());
                new_state.first_moment = Eigen::MatrixXd::Zero(weights_.rows(), weights_.cols());
                new_state.second_moment = Eigen::MatrixXd::Zero(weights_.rows(), weights_.cols());
            }
            optimizer_states_[optimizer_name] = new_state;
            return optimizer_states_[optimizer_name];
        }
        return it->second;
    }

    bool WeightedCache::has_optimizer_state(const std::string& optimizer_name) const {
        return optimizer_states_.find(optimizer_name) != optimizer_states_.end();
    }

    void WeightedCache::save_checkpoint(int step) {
        WeightGradient checkpoint(weights_, weight_gradient_, step);
        checkpoints_[step] = checkpoint;
        
        // Mantieni solo ultimi 10 checkpoint
        const int MAX_CHECKPOINTS = 10;
        if (checkpoints_.size() > MAX_CHECKPOINTS) {
            auto oldest = std::min_element(checkpoints_.begin(), checkpoints_.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            if (oldest != checkpoints_.end()) {
                checkpoints_.erase(oldest);
            }
        }
    }

    bool WeightedCache::load_checkpoint(int step, Eigen::MatrixXd& w, Eigen::VectorXd& b) {
        auto it = checkpoints_.find(step);
        if (it != checkpoints_.end()) {
            w = it->second.weights;
            // Nota: i bias non sono salvati nei checkpoint per semplicità
            // In una implementazione reale, salveresti anche i bias
            return true;
        }
        return false;
    }

    void WeightedCache::clear_checkpoints() {
        checkpoints_.clear();
    }


    // Gestione pesi
    void WeightedCache::set_weights(const Eigen::MatrixXd& w) { 
        weights_ = w; 
        // Ridimensiona tutti gli stati degli ottimizzatori
        for (auto& pair : optimizer_states_) {
            auto& state = pair.second;
            state.momentum = Eigen::MatrixXd::Zero(weights_.rows(), weights_.cols());
            state.first_moment = Eigen::MatrixXd::Zero(weights_.rows(), weights_.cols());
            state.second_moment = Eigen::MatrixXd::Zero(weights_.rows(), weights_.cols());
        }
    }



} // namespace layers

