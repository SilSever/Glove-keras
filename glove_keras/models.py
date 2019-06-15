import tensorflow.python.keras.backend as k
from tensorflow.python.keras.layers import Input, Embedding, Dot, Reshape, Add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adagrad

from glove_keras.config import CNTRL_EMB, CTX_EMB, CNTRL_BS, CTX_BS


class Glove:
    def __init__(
        self,
        vocab_size: int = 10,
        vector_dim: int = 3,
        alpha: float = 0.75,
        lr: float = 0.05,
        x_max: int = 100
    ):
        self.vocab_size = vocab_size
        self.vector_dim = vector_dim
        self.alpha = alpha
        self.lr = lr
        self.x_max = x_max

        self.model = self.build_model(self.vocab_size, self.vector_dim)

    def build_model(self, vocab_size: int, vector_dim: int):
        """
        Builds the Keras model.
        :param vocab_size: The number of distinct words.
        :param vector_dim: The vector dimension of each word.
        :return: the Keras GloVe model.
        """
        input_target = Input((1,), name="central_word_id")
        input_context = Input((1,), name="context_word_id")

        central_embedding = Embedding(
            vocab_size, vector_dim, input_length=1, name=CNTRL_EMB
        )(input_target)
        central_bias = Embedding(vocab_size, 1, input_length=1, name=CNTRL_BS)(
            input_target
        )

        context_embedding = Embedding(
            vocab_size, vector_dim, input_length=1, name=CTX_EMB
        )(input_context)
        context_bias = Embedding(vocab_size, 1, input_length=1, name=CTX_BS)(
            input_context
        )

        dot_product = Dot(axes=-1)([central_embedding, context_embedding])
        dot_product = Reshape((1,))(dot_product)
        bias_target = Reshape((1,))(central_bias)
        bias_context = Reshape((1,))(context_bias)

        prediction = Add()([dot_product, bias_target, bias_context])

        model = Model(inputs=[input_target, input_context], outputs=prediction)
        model.compile(loss=self.custom_loss, optimizer=Adagrad(lr=self.lr))
        print(model.summary())
        return model

    def custom_loss(self, y_true, y_pred):
        """
        GloVe's loss function, view section 3.1 on the original paper for details.
        :param y_true: The actual values, y_true = X_ij.
        :param y_pred: The predicted occurrences from the model ( w_i^T*w_j ).
        :return: The loss associated with this batch.
        """
        x_max = self.x_max
        alpha = self.alpha
        fxij = k.pow(k.clip(y_true / x_max, 0.0, 1.0), alpha)

        return k.sum(fxij * k.square(y_pred - k.log(y_true)), axis=-1)
