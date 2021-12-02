class MomentumStrategy:
    def setup(self, momentum_rate, error_matrix_generator):
        pass

    def apply_momentum(self, batch_size, loss, bias_loss, gradient_fun, layer_number):
        return gradient_fun(batch_size, loss[layer_number]), gradient_fun(batch_size, bias_loss[layer_number])

    def apply_momentum_softmax(self, batch_size, loss, bias_loss, gradient_fun):
        return gradient_fun(batch_size, loss), gradient_fun(batch_size, bias_loss)

    def modify_gradient_input(self, to_modify, to_modify_weights, layer_number):
        return to_modify_weights, to_modify

    def modify_gradient_input_softmax(self, to_modify, to_modify_weights):
        return to_modify_weights, to_modify

    @staticmethod
    def is_momentum():
        return False


class Momentum(MomentumStrategy):
    def __init__(self):
        self.momentum_rate = None
        self.prev_update = None
        self.prev_bias_update = None
        self.prev_softmax_update = None
        self.prev_softmax_bias_update = None

    def setup(self, momentum_rate, update_matrix_generator):
        self.momentum_rate = momentum_rate
        self.prev_update = update_matrix_generator.init_loss()
        self.prev_bias_update = update_matrix_generator.init_loss_bias()
        self.prev_softmax_update = update_matrix_generator.init_softmax_loss()
        self.prev_softmax_bias_update = update_matrix_generator.init_softmax_loss_bias()

    def apply_momentum(self, batch_size, loss, bias_loss, gradient_fun, layer_number):
        v_weights = self.prev_update[layer_number]
        v_bias = self.prev_bias_update[layer_number]
        delta_weights = gradient_fun(batch_size, loss[layer_number]) + v_weights * self.momentum_rate
        delta_bias = gradient_fun(batch_size, bias_loss[layer_number]) + v_bias * self.momentum_rate
        self.prev_update[layer_number] = delta_weights
        self.prev_bias_update[layer_number] = delta_bias
        return delta_weights, delta_bias

    def apply_momentum_softmax(self, batch_size, loss, bias_loss, gradient_fun):
        v_weights = self.momentum_rate
        v_bias = self.momentum_rate
        delta_weights = gradient_fun(batch_size, loss) + v_weights * self.prev_softmax_update
        delta_bias = gradient_fun(batch_size, bias_loss) + v_bias * self.prev_softmax_bias_update
        self.prev_softmax_update = delta_weights
        self.prev_softmax_bias_update = delta_bias
        return delta_weights, delta_bias

    @staticmethod
    def is_momentum():
        return True


class Nag(MomentumStrategy):
    def __init__(self):
        self.momentum_rate = None
        self.prev_update = None
        self.prev_bias_update = None
        self.prev_softmax_update = None
        self.prev_softmax_bias_update = None

    def setup(self, momentum_rate, update_matrix_generator):
        self.momentum_rate = momentum_rate
        self.prev_update = update_matrix_generator.init_loss()
        self.prev_bias_update = update_matrix_generator.init_loss_bias()
        self.prev_softmax_update = update_matrix_generator.init_softmax_loss()
        self.prev_softmax_bias_update = update_matrix_generator.init_softmax_loss_bias()

    def apply_momentum(self, batch_size, loss, bias_loss, gradient_fun, layer_number):
        v_weights = self.prev_update[layer_number]
        v_bias = self.prev_bias_update[layer_number]
        delta_weights = gradient_fun(batch_size, loss[
            layer_number] - self.momentum_rate * v_weights) + v_weights * self.momentum_rate
        delta_bias = gradient_fun(batch_size,
                                  bias_loss[layer_number] - self.momentum_rate * v_bias) + v_bias * self.momentum_rate
        self.prev_update[layer_number] = delta_weights
        self.prev_bias_update[layer_number] = delta_bias
        return delta_weights, delta_bias

    def apply_momentum_softmax(self, batch_size, loss, bias_loss, gradient_fun):
        v_weights = self.prev_softmax_update
        v_bias = self.prev_softmax_bias_update
        delta_weights = gradient_fun(batch_size, loss - self.momentum_rate * v_weights) + v_weights * self.momentum_rate
        delta_bias = gradient_fun(batch_size, bias_loss - self.momentum_rate * v_bias) + v_bias * self.momentum_rate
        self.prev_softmax_update = delta_weights
        self.prev_softmax_bias_update = delta_bias
        return delta_weights, delta_bias

    def modify_gradient_input(self, to_modify, to_modify_weights, layer_number):
        return self.momentum_rate * (to_modify_weights - self.prev_update[layer_number]), \
               self.momentum_rate * (to_modify - self.prev_bias_update[layer_number])

    def modify_gradient_input_softmax(self, to_modify_bias, to_modify_weights):
        return self.momentum_rate * (to_modify_weights - self.prev_softmax_update), \
               self.momentum_rate * (to_modify_bias - self.prev_softmax_bias_update)

    @staticmethod
    def is_momentum():
        return True
