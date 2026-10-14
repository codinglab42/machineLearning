#!/bin/bash
# fix_layer_signatures.sh

echo "Fixing layer signatures..."

# 1. Modifica tutti gli header dei layer per cambiare la firma di backward
find include/components/layers -name "*.h" -type f -exec sed -i 's/backward(const Eigen::MatrixXd& gradient, double learning_rate)/backward(const Eigen::MatrixXd& gradient)/g' {} \;

# 2. Modifica tutti i file .cpp
find src/components/layers -name "*.cpp" -type f -exec sed -i 's/backward(const Eigen::MatrixXd& gradient, double learning_rate)/backward(const Eigen::MatrixXd& gradient)/g' {} \;

# 3. Rimuovi has_activation dalle cache
find include/components/cache -name "*.h" -type f -exec sed -i '/bool has_activation/d' {} \;

# 4. Modifica i getter dei gradienti per restituire const reference
# (più complesso, meglio farlo a mano)

echo "Done! Now manually fix const reference returns for gradients."