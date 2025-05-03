import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore

from AudioTransformer.model import TransformerEncoderBlock, Classifier
from AudioTransformer.loader import data

class MainModel(keras.Model):
    def __init__(self, encoder, classifier, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.classifier = classifier
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

    def call(self, inputs, training=False):
        """Forward pass through encoder and classifier"""
        encoder_out = self.encoder(inputs, training=training)
        preds = self.classifier(encoder_out)
        return preds  # Returns predicted probabilities

    def calculate_loss(self, y_pred, y_true):
        return self.loss(y_true, y_pred)

    def calculate_acc(self, y_pred, y_true):
        predicted_classes = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, predicted_classes), dtype=tf.float32))
        return accuracy

    def train_step(self, batch_data):
        batch, batch2, label = batch_data

        with tf.GradientTape() as tape:
            preds = self(inputs=[batch, batch2], training=True)
            batch_loss = self.calculate_loss(preds, label)
            batch_acc = self.calculate_acc(preds, label)

        train_vars = self.trainable_variables
        grads = tape.gradient(batch_loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch, batch2, label = batch_data
        preds = self(inputs=[batch, batch2], training=False)
        batch_loss = self.calculate_loss(preds, label)
        batch_acc = self.calculate_acc(preds, label)

        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]