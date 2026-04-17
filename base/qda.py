import numpy as np
import numpy.linalg as LA

from base.bayesian import BaseBayesianClassifier


class QDA(BaseBayesianClassifier):

  # construye: 'inv_covs' -> lista de matrices de covarianza inversas, una por clase (cada row corresponde a un feature)
  #           'means' -> lista de vectores de medias, uno por clase (cada row corresponde a un feature)
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

# Optimización: Pregunta 3: en este punto se puede optimizar el código para evitar el for-loop. Ver implementación de FasterQDA en qda.py
class FasterQDA(TensorizedQDA):

    # override `predict`: en vez de iterar sobre cada observación, calcula todo el producto matricial de una vez (sin for-loop)
    def predict(self, X):
        # Pregunta 3: evitar el for-loop sobre observaciones.
        # tensor_means shape: (k, p, 1)
        # unbiased_X shape: (k, p, n)
        unbiased_X = X - self.tensor_means  # (p,n) - (k,p,1) -> (k,p,n)

        # (k,n,p) @ (k,p,p) @ (k,p,n) -> (k,n,n)
        inner = unbiased_X.transpose(0, 2, 1) @ self.tensor_inv_cov @ unbiased_X

        # tomar la diagonal de la matriz -> (k, n)
        quad_forms = np.diagonal(inner, axis1=1, axis2=2)

        # 0.5 * log(det(Σ^{-1})) term (broadcast a (k, n))
        log_dets = 0.5 * np.log(LA.det(self.tensor_inv_cov)).reshape(-1, 1)

        # log posteriori, shape (k, n)
        log_posteriori = self.log_a_priori.reshape(-1, 1) + (log_dets - 0.5 * quad_forms)

        # return shape (1, n)
        return np.argmax(log_posteriori, axis=0).reshape(1, -1)

# Optimización: Pregunta 6: Utilizar la propiedad antes demostrada para reimplementar la predicción del modelo `FasterQDA` de forma eficiente en un nuevo modelo `EfficientQDA`
class EfficientQDA(TensorizedQDA):

    # override `predict`: en vez de iterar sobre cada observación, calcula todo el producto matricial de una vez (sin for-loop)
    def predict(self, X):
        # tensor_means shape: (k, p, 1)
        # unbiased_X shape: (k, p, n) — extraemos el mean de cada clase y se lo restamos a cada observación (broadcasting)
        unbiased_X = X - self.tensor_means  # broadcasts (p,n) - (k,p,1) -> (k,p,n)

        # tensor_inv_cov shape: (k, p, p)
        # producto matricial entre la matriz de covarianza y el vector de cada observación (obs - tensor_mean)
        # (k,p,p) @ (k,p,n) -> (k,p,n)
        AX = self.tensor_inv_cov @ unbiased_X

        # devuelve la diagonal en formato -> (k, n)
        quad_forms = np.sum(AX * unbiased_X, axis=1)

        # log determinants: (k,)  -> reshape a (k,1) para broadcasting
        log_dets = 0.5 * np.log(LA.det(self.tensor_inv_cov)).reshape(-1, 1)

        # log conditionals shape: (k, n)
        log_conditionals = log_dets - 0.5 * quad_forms

        # log posteriori shape: (k, n)
        log_posteriori = self.log_a_priori.reshape(-1, 1) + log_conditionals

        # return shape (1, n)
        return np.argmax(log_posteriori, axis=0).reshape(1, -1)
