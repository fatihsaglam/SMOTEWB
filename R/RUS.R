#' @title  Random Undersampling (RUS)
#'
#' @description Resampling with RUS.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#' @param n_neededToRemove vector of desired number removal for each class.
#' A vector of integers for each class. Default is NULL meaning full balance.
#' Must be equal or lower than the number of samples in each class.
#' @param ... not used.
#'
#' @details
#' Random Undersampling (RUS) is a method of removing negative
#' samples until balance is achieved.
#'
#' Can work with classes more than 2.
#'
#' @return a list with resampled dataset.
#'
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
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
#' m <- RUS(x = x, y = y)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname RUS
#' @export

RUS <- function(x, y, n_neededToRemove = NULL, ...) {
  x <- as.matrix(x)

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }

  if (!is.factor(y)) {
    stop("y must be a factor")
  }

  var_names <- colnames(x)
  x <- as.matrix(x)
  p <- ncol(x)
  n <- nrow(x)

  class_names <- levels(y)
  n_classes <- sapply(class_names, function(m) sum(y == m))
  k_class <- length(class_names)
  n_classes_min <- min(n_classes)

  if (is.null(n_neededToRemove)) {
    n_neededToRemove <- n_classes - n_classes_min
  }
  if (length(n_neededToRemove) != k_class) {
    stop("n_needed must be an integer vector matching the number of classes.")
  }
  if (any(n_neededToRemove > n_classes)) {
    stop("number of removal cannot be higher than the number of classses")
  }

  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])
  y_classes <- lapply(class_names, function(m) y[y == m])

  for (i in 1:k_class) {
    if (n_neededToRemove[i] == 0) {
      next
    }
    i_sample <- sample(1:n_classes[i], size = n_neededToRemove[i])

    x_classes[[i]] <- x_classes[[i]][-i_sample,, drop = FALSE]
    y_classes[[i]] <- y_classes[[i]][-i_sample]
  }

  x_new <- do.call(rbind, x_classes)
  y_new <- unlist(y_classes)
  colnames(x_new) <- var_names

  return(list(
    x_new = x_new,
    y_new = y_new
  ))
}

