#' @title  Randomly Over Sampling Examples
#'
#' @description Generates synthetic data for each class to balance imbalanced
#' datasets using kernel density estimations. Can be used for multiclass datasets.
#'
#' @param x feature matrix or data.frame.
#' @param y a factor class variable. Can be multiclass.
#' @param h A numeric vector of length one or number of classes in y. If one is
#' given, all classes will have same shrink factor. If a value is given for each
#' classes, it will match respectively to \code{levels(y)}. Default is 1.
#' @param ovRate Oversampling rate multiplied by the difference between maximum
#' and other of class sizes. Default is 1 meaning full balance.
#'
#' @details
#' Randomly Over Sampling Examples (ROSE) (Menardi and Torelli, 2014) is an
#' oversampling method which uses conditional kernel densities to balance dataset.
#' There is already an R package called `ROSE` (Lunardon et al., 2014).
#' Difference is that this one is much faster and can be applied for more than two classes.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @importFrom  stats rnorm
#' @importFrom  stats sd
#'
#' @references
#' Lunardon, N., Menardi, G., and Torelli, N. (2014). ROSE: a Package for Binary
#' Imbalanced Learning. R Jorunal, 6:82–92.
#'
#' Menardi, G. and Torelli, N. (2014). Training and assessing classification
#' rules with imbalanced data. Data Mining and Knowledge Discovery, 28:92–122.
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
#' m <- ROSE(x = x, y = y, h = c(0.12, 1))
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname ROSE
#' @export

ROSE <- function(
    x,
    y,
    h = 1,
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

  if (any(h < 0)) {
    stop("h can not be negative")
  }

  if (!is.numeric(h)) {
    stop("h must be numeric")
  }

  var_names <- colnames(x)
  x <- as.matrix(x)
  n <- length(y)
  p <- ncol(x)

  class_names <- as.character(levels(y))
  k_class <- length(class_names)

  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])
  n_classes <- sapply(class_names, function(m) sum(y == m))

  n_needed <- round((max(n_classes) - n_classes)*ovRate)

  i_new_classes <- lapply(1:k_class, function(m) {
    sample(1:n_classes[m], n_needed[m], replace = TRUE)
  })

  cons_kernel_classes <- sapply(n_classes, function(m) {
    4/((p + 2) * m)
  })^(1/(p + 4))

  if (length(h) == 1) {
    h_classes <- rep(h, k_class)
  } else {
    if (length(h) == k_class) {
      h_classes <- h
    } else {
      stop(paste0("h must be length one or number of classes, ", k_class))
    }
  }

  H_classes <- lapply(1:k_class, function(m) {
    h_classes[m] *
      cons_kernel_classes[m] *
      diag(apply(x_classes[[m]], 2, sd) + 1e-7, p)
  })

  x_noise_classes <- lapply(1:k_class, function(m) {
    matrix(rnorm(n_needed[m]*p), n_needed[m], p) %*% H_classes[[m]]
  })

  x_new_classes <- lapply(1:k_class, function(m) {
    x_noise_classes[[m]] + x_classes[[m]][i_new_classes[[m]],,drop = FALSE]
  })

  x_new <- do.call(rbind, x_new_classes)

  y_new <- factor(unlist(sapply(1:k_class, function(m) {
    rep(class_names[m], n_needed[m])
  })), levels = class_names, labels = class_names)

  colnames(x_new) <- var_names

  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_new,
    y_syn = y_new
  ))
}
