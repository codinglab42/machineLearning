#include "components/loss/loss_factory.h"
#include "components/loss/binary_cross_entropy_loss.h"
#include "components/loss/categorical_cross_entropy_loss.h"
#include "components/loss/mean_squared_error_loss.h"
#include "components/loss/mean_absolute_error_loss.h"
#include "components/loss/huber_loss.h"
#include "exceptions/exception_macros.h"

namespace loss {

    std::unordered_map<std::string, LossFactory::Creator>& LossFactory::get_registry() {
        static std::unordered_map<std::string, Creator> registry;
        return registry;
    }

    void LossFactory::register_loss(const std::string& name, Creator creator) {
        get_registry()[name] = creator;
    }

    void LossFactory::register_all_losses() {
        static bool registered = []() {
            register_loss("binary_crossentropy", 
                        []() { return std::make_unique<BinaryCrossEntropyLoss>(); });
            register_loss("categorical_crossentropy", 
                        []() { return std::make_unique<CategoricalCrossEntropyLoss>(); });
            register_loss("mse", 
                        []() { return std::make_unique<MeanSquaredErrorLoss>(); });
            register_loss("mae", 
                        []() { return std::make_unique<MeanAbsoluteErrorLoss>(); });
            register_loss("huber", 
                        []() { return std::make_unique<HuberLoss>(); });
            return true;
        }();
    }

    std::unique_ptr<Loss> LossFactory::create(const std::string& name) {
        register_all_losses();  // Assicura che tutte le loss siano registrate
        
        auto& registry = get_registry();
        auto it = registry.find(name);
        if (it == registry.end()) {
            ML_THROW_PARAMETER_ERROR("loss_name", 
                                    "unknown loss function: " + name + 
                                    ". Available: binary_crossentropy, categorical_crossentropy, mse, mae, huber",
                                    "LossFactory::create");
        }
        
        return it->second();
    }

} // namespace loss