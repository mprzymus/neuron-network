class MomentumStrategy:
    def setup(self, momentum_rate, error_matrix_generator):
        pass

    def apply_momentum(self, batch_size, loss, bias_loss, gradient_fun, layer_number):
        return gradient_fun(batch_size, loss[layer_number]), gradient_fun(batch_size, bias_loss[layer_number])

    def apply_momentum_softmax(self, batch_size, loss, bias_loss, gradient_fun):
        return gradient_fun(batch_size, loss), gradient_fun(batch_size, bias_loss)


class Momentum(MomentumStrategy):
    def __init__(self):
        self.momentum_rate = None
        self.prev_errors = None
        self.prev_bias_errors = None
        self.prev_softmax_error = None
        self.prev_softmax_bias_error = None

    def setup(self, momentum_rate, error_matrix_generator):
        self.momentum_rate = momentum_rate
        self.prev_errors = error_matrix_generator.init_loss()
        self.prev_bias_errors = error_matrix_generator.init_loss_bias()
        self.prev_softmax_error = error_matrix_generator.init_softmax_loss()
        self.prev_softmax_bias_error = error_matrix_generator.init_softmax_loss_bias()

    def apply_momentum(self, batch_size, loss, bias_loss, gradient_fun, layer_number):
        v_weights = self.prev_errors[layer_number]
        v_bias = self.prev_bias_errors[layer_number]
        delta_weights = gradient_fun(batch_size, loss[layer_number]) + v_weights * self.momentum_rate
        delta_bias = gradient_fun(batch_size, bias_loss[layer_number]) + v_bias * self.momentum_rate
        self.prev_errors[layer_number] = delta_weights
        self.prev_bias_errors[layer_number] = delta_bias
        return delta_weights, delta_bias

    def apply_momentum_softmax(self, batch_size, loss, bias_loss, gradient_fun):
        v_weights = self.momentum_rate
        v_bias = self.momentum_rate
        delta_weights = gradient_fun(batch_size, loss) + v_weights * self.prev_softmax_error
        delta_bias = gradient_fun(batch_size, bias_loss) + v_bias * self.prev_softmax_bias_error
        self.prev_softmax_error = delta_weights
        self.prev_softmax_bias_error = delta_bias
        return delta_weights, delta_bias


class Nag(MomentumStrategy):
    def __init__(self):
        self.momentum_rate = None
        self.prev_errors = None
        self.prev_bias_errors = None
        self.prev_softmax_error = None
        self.prev_softmax_bias_error = None

    def setup(self, momentum_rate, error_matrix_generator):
        self.momentum_rate = momentum_rate
        self.prev_errors = error_matrix_generator.init_loss()
        self.prev_bias_errors = error_matrix_generator.init_loss_bias()
        self.prev_softmax_error = error_matrix_generator.init_softmax_loss()
        self.prev_softmax_bias_error = error_matrix_generator.init_softmax_loss_bias()

    def apply_momentum(self, batch_size, loss, bias_loss, gradient_fun, layer_number):
        v_weights = self.prev_errors[layer_number]
        v_bias = self.prev_bias_errors[layer_number]
        delta_weights = gradient_fun(batch_size, loss[
            layer_number] - self.momentum_rate * v_weights) + v_weights * self.momentum_rate
        delta_bias = gradient_fun(batch_size,
                                  bias_loss[layer_number] - self.momentum_rate * v_bias) + v_bias * self.momentum_rate
        self.prev_errors[layer_number] = delta_weights
        self.prev_bias_errors[layer_number] = delta_bias
        return delta_weights, delta_bias

    def apply_momentum_softmax(self, batch_size, loss, bias_loss, gradient_fun):
        v_weights = self.prev_softmax_error
        v_bias = self.prev_softmax_bias_error
        delta_weights = gradient_fun(batch_size, loss - self.momentum_rate * v_weights) + v_weights * self.momentum_rate
        delta_bias = gradient_fun(batch_size, bias_loss - self.momentum_rate * v_bias) + v_bias * self.momentum_rate
        self.prev_softmax_error = delta_weights
        self.prev_softmax_bias_error = delta_bias
        return delta_weights, delta_bias
