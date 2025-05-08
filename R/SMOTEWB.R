#' @title  SMOTE with boosting (SMOTEWB)
#'
#' @description Resampling with SMOTE with boosting.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#' @param n_weak_classifier number of weak classifiers for boosting.
#' @param class_weights numeric vector of length two. First number is for
#' positive class, and second is for negative. Higher the relative weight,
#' lesser noises for that class. By default,  \eqn{2\times n_{neg}/n} for
#' positive and \eqn{2\times n_{pos}/n} for negative class.
#' @param k_max to increase maximum number of neighbors. Default is
#' \code{ceiling(n_neg/n_pos)}.
#' @param n_needed vector of desired number of synthetic samples for each class.
#' A vector of integers for each class. Default is NULL meaning full balance.
#' @param ... additional inputs for ada::ada().
#'
#' @details
#' SMOTEWB (Saglam & Cengiz, 2022) is a SMOTE-based oversampling method which
#' can handle noisy data and adaptively decides the appropriate number of neighbors
#' to link during resampling with SMOTE.
#'
#' Trained model based on this method gives significantly better Matthew
#' Correlation Coefficient scores compared to others.
#'
#' Can work with classes more than 2.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'  \item{x_syn}{Generated synthetic data.}
#'  \item{y_syn}{Generated synthetic data labels.}
#'  \item{w}{Boosting weights for original dataset.}
#'  \item{k}{Number of nearest neighbors for positive class samples.}
#'  \item{C}{Number of synthetic samples for each positive class samples.}
#'  \item{fl}{"good", "bad" and "lonely" sample labels}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @importFrom  FNN knnx.index
#' @importFrom  stats runif
#' @importFrom  stats sd
#'
#' @references
#' Sağlam, F., & Cengiz, M. A. (2022). A novel SMOTE-based resampling technique
#' trough noise detection and the boosting procedure. Expert Systems with
#' Applications, 200, 117023.
#'
#' Can work with 2 classes only yet.
#' @examples
#'
#' set.seed(1)
#' x <- rbind(matrix(rnorm(2000, 3, 1), ncol = 2, nrow = 1000),
#'            matrix(rnorm(100, 5, 1), ncol = 2, nrow = 50))
#' y <- as.factor(c(rep("negative", 1000), rep("positive", 50)))
#'
#' plot(x, col = y)
#'
#' # resampling
#' m <- SMOTEWB(x = x, y = y, n_weak_classifier = 150)
#'
#' plot(m$x_new, col = m$y_new)
#'
#'
#' @rdname SMOTEWB
#' @export

SMOTEWB <- function(
    x,
    y,
    n_weak_classifier = 100,
    class_weights = NULL,
    k_max = NULL,
    n_needed = NULL,
    ...) {

  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.factor(y)) {
    stop("y must be a factor")
  }

  var_names <- colnames(x)
  x <- as.matrix(x)
  n <- length(y)
  p <- ncol(x)

  class_names <- as.character(unique(y))
  n_classes <- sapply(class_names, function(m) sum(y == m))
  k_class <- length(class_names)
  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])

  if (is.null(n_needed)) {
    n_needed <- max(n_classes) - n_classes
  }

  x_syn <- matrix(NA, nrow = 0, ncol = p)
  y_syn <- factor(c(), levels = class_names)
  w <- list()
  k <- list()
  C <- list()
  fl <- list()

  for (j in 1:k_class) {
    m_syn <- generateSMOTEWB(
      x_pos = x_classes[[j]],
      x_neg = do.call(rbind, x_classes[-j]),
      n_syn = n_needed[j],
      k_max = k_max,
      n_weak_classifier = n_weak_classifier,
      class_pos = class_names[j],
      class_names = class_names,
      class_weights = class_weights
    )

    x_syn <- rbind(x_syn, m_syn$x_syn)
    y_syn <- c(y_syn, m_syn$y_syn)

    w[[j]] <- m_syn$w
    k[[j]] <- m_syn$k
    C[[j]] <- m_syn$C
    fl[[j]] <- m_syn$fl
  }

  x_new <- rbind(x, x_syn)
  y_new <- c(y, y_syn)
  colnames(x_new) <- var_names
  names(w) <- names(k) <- names(C) <- names(fl) <- class_names

  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = y_syn,
    w = w,
    k = k,
    C = C,
    fl = fl
  ))
}
