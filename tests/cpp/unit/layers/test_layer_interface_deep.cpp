// test/layers/layer_interface_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/layer.h"
#include <sstream>

using namespace layers;
using namespace testing;

// Mock Layer per testare l'interfaccia
class MockLayer : public Layer {
public:
    MOCK_METHOD(Eigen::MatrixXd, forward, (const Eigen::MatrixXd&), (override));
    MOCK_METHOD(Eigen::MatrixXd, forward, (const Eigen::MatrixXd&, bool), (override));
    //MOCK_METHOD(Eigen::MatrixXd, backward, (const Eigen::MatrixXd&, double), (override));
    MOCK_METHOD(void, serialize, (std::ostream&), (const, override));
    MOCK_METHOD(void, deserialize, (std::istream&), (override));
    MOCK_METHOD(std::string, get_type, (), (const, override));
    MOCK_METHOD(std::string, get_config, (), (const, override));
    MOCK_METHOD(bool, has_weights, (), (const, override));
    MOCK_METHOD(Eigen::MatrixXd, get_weights, (), (const, override));
    MOCK_METHOD(void, set_weights, (const Eigen::MatrixXd&), (override));
    MOCK_METHOD(int, get_parameter_count, (), (const, override));
    MOCK_METHOD(int, get_input_size, (), (const, override));
    MOCK_METHOD(int, get_output_size, (), (const, override));
    MOCK_METHOD(void, clear_cache, (), (override));
    MOCK_METHOD(std::shared_ptr<LayerCache>, get_cache, (), (const, override));
    MOCK_METHOD(void, set_cache, (std::shared_ptr<LayerCache>), (override));
    MOCK_METHOD(Eigen::VectorXd, get_biases, (), (const, override));
    MOCK_METHOD(void, set_biases, (const Eigen::VectorXd&), (override));
    MOCK_METHOD(void, set_input_shape, (int), (override));

    LayerType get_layer_type() const override { return LayerType::DENSE; }
    uint32_t get_version() const override { return 1; }

    //MOCK_METHOD(Eigen::MatrixXd, get_weights_gradient, (), (const, override));
    //MOCK_METHOD(Eigen::VectorXd, get_bias_gradient, (), (const, override));
    MOCK_METHOD(bool, get_use_bias, (), (const, override));};

/*TEST(LayerInterfaceTest, VirtualDestructor) {
    EXPECT_NO_THROW({
        std::unique_ptr<Layer> layer = std::make_unique<MockLayer>();
    });
}
*/
/*TEST(LayerInterfaceTest, ForwardWithTrainingFlag) {
    MockLayer layer;
    Eigen::MatrixXd input(2, 3);
    input.setRandom();
    
    EXPECT_CALL(layer, forward(Eigen::MatrixXd(input), true))
        .Times(1)
        .WillOnce(Return(Eigen::MatrixXd::Zero(2, 3)));
    
    layer.forward(input, true);
}*/