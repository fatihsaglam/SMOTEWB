#' @title  Safe-level Synthetic Minority Oversampling Technique
#'
#' @description \code{SLSMOTE()} generates synthetic samples by considering a
#' safe level of the nearest minority class examples.
#'
#' @param x feature matrix or data.frame.
#' @param y a factor class variable with two classes.
#' @param k1 number of neighbors to link. Default is 5.
#' @param k2 number of neighbors to determine safe levels. Default is 5.
#' @param ovRate Oversampling rate multiplied by the difference between maximum
#' and other of class sizes. Default is 1 meaning full balance.
#'
#' @details
#' SLSMOTE uses the safe-level distance metric to identify the minority class
#' samples that are safe to oversample. Safe-level distance measures the
#' distance between a minority class sample and its k-nearest minority class
#' neighbors. A sample is considered safe to oversample if its safe-level is
#' greater than a threshold. The safe-level of a sample is the ratio of minority
#' class samples among its k-nearest neighbors.
#'
#' In SLSMOTE, the oversampling process only applies to the safe minority class
#' samples, which avoids the generation of noisy samples that can lead to
#' overfitting. To generate synthetic samples, SLSMOTE randomly selects a
#' minority class sample and finds its k-nearest minority class neighbors.
#' Then, a random minority class neighbor is selected, and a synthetic sample
#' is generated by adding a random proportion of the difference between the
#' selected sample and its neighbor to the selected sample.
#'
#' Can work with classes more than 2.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'  \item{x_syn}{Generated synthetic data.}
#'  \item{C}{Number of synthetic samples for each positive class samples.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @importFrom  FNN knnx.index
#' @importFrom  stats rnorm
#' @importFrom  stats sd
#'
#' @references
#' Bunkhumpornpat, C., Sinapiromsaran, K., & Lursinsap, C. (2009).
#' Safe-level-smote: Safe-level-synthetic minority over-sampling technique for
#' handling the class imbalanced problem. In Advances in Knowledge Discovery
#' and Data Mining: 13th Pacific-Asia Conference, PAKDD 2009 Bangkok, Thailand,
#' April 27-30, 2009 Proceedings 13 (pp. 475-482). Springer Berlin Heidelberg.
#'
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
#' m <- SLSMOTE(x = x, y = y, k1 = 5, k2 = 5)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname SLSMOTE
#' @export

SLSMOTE <- function(x, y, k1 = 5, k2 = 5,
                    ovRate = 1) {

  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.factor(y)) {
    stop("y must be a factor")
  }

  if (!is.numeric(k1)) {
    stop("k1 must be numeric")
  }

  if (k1 < 1) {
    stop("k1 must be positive")
  }

  if (!is.numeric(k2)) {
    stop("k2 must be numeric")
  }

  if (k2 < 1) {
    stop("k2 must be positive")
  }

  var_names <- colnames(x)
  x <- as.matrix(x)
  p <- ncol(x)

  class_names <- as.character(levels(y))
  n_classes <- sapply(class_names, function(m) sum(y == m))
  k_class <- length(class_names)
  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])

  n_needed <- round((max(n_classes) - n_classes)*ovRate)

  x_syn <- matrix(NA, nrow = 0, ncol = p)
  y_syn <- factor(c(), levels = class_names)
  C <- list()

  for (j in 1:k_class) {
    m_syn <- generateSLSMOTE(
      x_pos = x_classes[[j]],
      x_neg = do.call(rbind, x_classes[-j]),
      n_syn = n_needed[j],
      k1 = k1,
      k2 = k2,
      class_pos = class_names[j],
      class_names = class_names
    )

    x_syn <- rbind(x_syn, m_syn$x_syn)
    y_syn <- c(y_syn, m_syn$y_syn)

    C[[j]] <- m_syn$C
  }

  x_new <- rbind(x, x_syn)
  y_new <- c(y, y_syn)
  colnames(x_new) <- var_names
  names(C) <- class_names

  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = y_syn,
    C = C
  ))
}
