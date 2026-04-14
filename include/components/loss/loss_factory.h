#ifndef LOSS_FACTORY_H
#define LOSS_FACTORY_H

#include "components/loss/loss.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

namespace loss {

class LossFactory {
public:
    using Creator = std::function<std::unique_ptr<Loss>()>;
    
    static void register_loss(const std::string& name, Creator creator);
    static void register_all_losses();
    static std::unique_ptr<Loss> create(const std::string& name);
    
private:
    static std::unordered_map<std::string, Creator>& get_registry();
};

} // namespace loss

#endif // LOSS_FACTORY_H