#ifndef REGULARIZER_H
#define REGULARIZER_H

#include <Eigen/Dense>
#include <string>
#include <memory>

namespace models {

    // Enum per i tipi di regolarizzatori
    enum class RegularizerType {
        NONE,
        L1,
        L2,
        ELASTIC_NET
    };

    // Classe base astratta per tutti i regolarizzatori
    class Regularizer {
    public:
        Regularizer(double strength = 0.01);
        virtual ~Regularizer() = default;
        
        // Calcola il termine di regolarizzazione per i pesi
        virtual double compute_loss(const Eigen::MatrixXd& weights) const = 0;
        virtual double compute_loss(const Eigen::VectorXd& bias) const = 0;
        
        // Calcola il gradiente della regolarizzazione per i pesi
        virtual Eigen::MatrixXd compute_gradient(const Eigen::MatrixXd& weights) const = 0;
        virtual Eigen::VectorXd compute_gradient(const Eigen::VectorXd& bias) const = 0;
        
        // Getter/Setter
        double get_strength() const { return strength_; }
        void set_strength(double strength) { strength_ = strength; }
        
        // Restituisce il tipo di regolarizzatore
        virtual RegularizerType get_type() const = 0;
        virtual std::string get_type_str() const = 0;
        
        // Serializzazione
        virtual void serialize(std::ostream& out) const;
        virtual void deserialize(std::istream& in);
        
        // Clona il regolarizzatore
        virtual std::unique_ptr<Regularizer> clone() const = 0;
        
    protected:
        double strength_;  // coefficiente di regolarizzazione (lambda)
    };

} // namespace models

#endif
