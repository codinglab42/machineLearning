#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace models {

    // Enum per i tipi di ottimizzatori
    enum class OptimizerType {
        SGD,
        MOMENTUM,
        ADAM,
        RMSPROP,
        ADAGRAD
    };

    // Classe base astratta per tutti gli ottimizzatori
    class Optimizer {
    public:
        Optimizer(double learning_rate = 0.01, double decay = 0.0);
        virtual ~Optimizer() = default;
        
        // Metodo principale: aggiorna i pesi
        virtual void update_weights(Eigen::Ref<Eigen::MatrixXd> weights, const Eigen::Ref<const Eigen::MatrixXd>& gradient) = 0;
        virtual void update_bias(Eigen::Ref<Eigen::VectorXd> bias, const Eigen::Ref<const Eigen::VectorXd>& gradient) = 0;
        
        // Resetta lo stato dell'ottimizzatore (per nuovo training)
        virtual void reset() = 0;
        
        // Getter/Setter
        double get_learning_rate() const { return learning_rate_; }
        double get_current_learning_rate() const;
        void set_learning_rate(double lr) { learning_rate_ = lr; }
        
        double get_decay() const { return decay_; }
        void set_decay(double decay) { decay_ = decay; }
        
        int get_iterations() const { return iterations_; }
        
        // Restituisce il tipo di ottimizzatore
        virtual OptimizerType get_type() const = 0;
        virtual std::string get_type_str() const = 0;
        
        // Serializzazione
        virtual void serialize(std::ostream& out) const;
        virtual void deserialize(std::istream& in);
        
        // Clona l'ottimizzatore
        virtual std::unique_ptr<Optimizer> clone() const = 0;
        
    protected:
        double learning_rate_;
        double decay_;
        int iterations_;  // numero di iterazioni eseguite

    };

} // namespace models

#endif
