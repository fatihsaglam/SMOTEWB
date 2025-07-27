#' @title  Random Walk Oversampling (SMOTE)
#'
#' @description Resampling with RWO
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#' @param ovRate Oversampling rate multiplied by the difference between maximum
#' and other of class sizes. Default is 1 meaning full balance.
#'
#' @details
#' RWO (Zhang and Li, 2014) is an oversampling method which generates data using
#' variable standard error in a way that it preserves the variances of all variables.
#'
#' Can work with classes more than 2.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'  \item{x_syn}{Generated synthetic feature data.}
#'  \item{y_syn}{Generated synthetic label data.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @importFrom  stats rnorm
#' @importFrom  stats sd
#'
#' @references
#' Zhang, H., & Li, M. (2014). RWO-Sampling: A random walk over-sampling
#' approach to imbalanced data classification. Information Fusion, 20, 99-116.
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
#' m <- RWO(x = x, y = y)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname RWO
#' @export

RWO <- function(x, y,
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
  var_names <- colnames(x)
  x <- as.matrix(x)
  p <- ncol(x)
  n <- nrow(x)

  class_names <- levels(y)
  n_classes <- sapply(class_names, function(m) sum(y == m))
  k_class <- length(class_names)
  n_classes_max <- max(n_classes)
  n_needed <- round((max(n_classes) - n_classes)*ovRate)

  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])
  x_syn_list <- list()

  for (i in 1:k_class) {
    if (n_needed[i] == 0) {
      next
    }

    x_main <- x_classes[[i]]
    se_main <- apply(x_main, 2, function(m) sd(m)/sqrt(length(m)))

    noise <- sapply(1:p, function(m) {
      rnorm(n_needed[i], sd = se_main[m])
    })

    x_syn_list[[i]] <- x_main[sample(1:n_classes[i], replace = TRUE, size = n_needed[i]),, drop = FALSE] + noise
  }

  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- factor(unlist(sapply(1:k_class, function(m) rep(class_names[m], n_needed[m]))), levels = class_names, labels = class_names)

  x_new <- rbind(x, x_syn)
  y_new <- c(y, y_syn)

  colnames(x_new) <- var_names
  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = y_syn
  ))
}
