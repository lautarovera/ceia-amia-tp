import numpy as np
import numpy.linalg as LA

from base.bayesian import BaseBayesianClassifier


class QDA(BaseBayesianClassifier):

  def _fit_params(self, X, y):
    # estimate each covariance matrix
    self.inv_covs = [LA.inv(np.cov(X[:,y.flatten()==idx], bias=True))
                      for idx in range(len(self.log_a_priori))]
    # Q5: por que hace falta el flatten y no se puede directamente X[:,y==idx]?
    # Q6: por que se usa bias=True en vez del default bias=False?
    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]
    # Q7: que hace axis=1? por que no axis=0?

  def _predict_log_conditional(self, x, class_idx):
    # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
    # this should depend on the model used
    inv_cov = self.inv_covs[class_idx]
    unbiased_x =  x - self.means[class_idx]
    return 0.5*np.log(LA.det(inv_cov)) -0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        # ask plain QDA to fit params
        super()._fit_params(X,y)

        # stack onto new dimension
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self,x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_x

        return 0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))

class FasterQDA(TensorizedQDA):
    '''
    Como queremos ahora predecir X con forma (p, n), resulta óptimo aplicar polimorfismo sobre el método predict de la clase base BaseBayesianClassifier, ya que éste llama a _predict_one para cada observacion (bucle for), tanto en QDA (inherited) y TensorizedQDA (overrided). De esta forma, al sobreescribir predict en FasterQDA, podemos eliminar el bucle for y aprovechar la tensorización.
    '''

    def predict(self, X):
        unbiased_X = X - self.tensor_means
        inner_prod = unbiased_X.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_X

        log_conditionals =0.5*np.log(LA.det(self.tensor_inv_cov))[:, None] - 0.5 * np.diagonal(inner_prod, axis1=1, axis2=2)

        return np.argmax(self.log_a_priori[:, None] + log_conditionals, axis=0).reshape(1, -1)

class EfficientQDA(TensorizedQDA):
    '''
    Utilizando la propiedad demostrada en el punto 5, se puede reimplementar la predicción del modelo FasterQDA de forma eficiente. Se puede calcular directamente la diagonal utilizando productos elemento a elemento y sumas.
    '''

    def predict(self, X):
        unbiased_X = X - self.tensor_means
        tmp = self.tensor_inv_cov @ unbiased_X
        inner_prod_diag = np.sum(tmp * unbiased_X, axis=1)

        log_conditionals = 0.5*np.log(LA.det(self.tensor_inv_cov))[:, None] - 0.5 * inner_prod_diag

        return np.argmax(self.log_a_priori[:, None] + log_conditionals, axis=0).reshape(1, -1)
