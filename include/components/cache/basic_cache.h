#ifndef BASIC_CACHE_H
#define BASIC_CACHE_H

#include "layer_cache.h"

namespace layers {

    class BasicCache : public LayerCache {
    public:
        BasicCache();
        ~BasicCache() override = default;
        
        // Implementazione interfaccia
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "BasicCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_; }
        const Eigen::MatrixXd& get_output() const override { return output_; }
        bool has_activation() const override { return has_activation_; }
        
        // Setter per i dati (per i layer)
        void set_input(const Eigen::MatrixXd& new_input) { input_ = new_input; }
        void set_output(const Eigen::MatrixXd& new_output) { output_ = new_output; }
        void set_has_activation(bool has) { has_activation_ = has; }
        
        // Accesso modificabile (per i layer che devono popolare la cache)
        Eigen::MatrixXd& mutable_input() { return input_; }
        Eigen::MatrixXd& mutable_output() { return output_; }

    protected:
        Eigen::MatrixXd input_;
        Eigen::MatrixXd output_;
        bool has_activation_;
    };

} // namespace layers

#endif
