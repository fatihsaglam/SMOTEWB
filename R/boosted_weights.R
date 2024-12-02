#' @title Boosted Weights for SMOTE with Boosting (SMOTEWB)
#'
#' @description Calculation of Boosted Weights for SMOTE with Boosting (SMOTEWB).
#'
#' @param x feature matrix.
#' @param y a factor class variable. Can work with more than two classes.
#' @param n_iter number of trees.
#'
#' @details
#' Calculation of Boosted Weights for SMOTE with Boosting using SAMME algorithm.
#'
#' @return a vector of case weights.
#'  \item{w}{last weight vector of boosting process}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @import rpart
#' @importFrom stats predict
#'
#' @references
#' Freund, Y., & Schapire, R. E. (1996, July). Experiments with a new boosting
#' algorithm. In icml (Vol. 96, pp. 148-156).
#'
#' Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class adaboost.
#' Statistics and its Interface, 2(3), 349-360.
#'
#' @noRd

boosted_weights <- function(x, y, n_iter = 100, lr = 1) {
  n <- nrow(x)
  w <- rep(1/n, n)
  k_class <- length(levels(y))

  for (i in 1:n_iter) {
    dat <- data.frame(x, y = y)
    model <- rpart::rpart(
      y~., data = dat,
      weights = w,
      control = rpart::rpart.control(minsplit = 3, cp = 0.01, maxdepth = 30))

    preds <- predict(model, data = x, type = "class")

    w_error <- sum(w[preds != y])
    alpha <- lr * (k_class - 1)/(k_class) * log((1 - w_error) / w_error) + log(k_class - 1)

    w[preds != y] <- w[preds != y] * exp(alpha)

    w <- w / sum(w)
  }
  return(w)
}
