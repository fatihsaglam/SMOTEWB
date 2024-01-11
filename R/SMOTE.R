#' @title  Synthetic Minority Oversampling Technique (SMOTE)
#'
#' @description Resampling with SMOTE.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#' @param k number of neighbors. Default is 5.
#'
#' @details
#' SMOTE (Chawla et al., 2002) is an oversampling method which creates links
#' between positive samples and nearest neighbors and generates synthetic
#' samples along that link.
#'
#' It is well known that SMOTE is sensitive to noisy data. It may create more
#' noise.
#'
#' Can work with classes more than 2.
#'
#' Note: Much faster than \code{smotefamily::SMOTE()}.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'  \item{x_syn}{Generated synthetic feature data.}
#'  \item{y_syn}{Generated synthetic label data.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @importFrom  FNN knnx.index
#' @importFrom  stats runif
#' @importFrom  stats sd
#'
#' @references
#' Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE:
#' synthetic minority over-sampling technique. Journal of artificial
#' intelligence research, 16, 321-357.
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
#' m <- SMOTE(x = x, y = y, k = 7)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname SMOTE
#' @export

SMOTE <- function(x, y, k = 5) {

  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.factor(y)) {
    stop("y must be a factor")
  }

  if (!is.numeric(k)) {
    stop("k must be numeric")
  }

  if (k < 1) {
    stop("k must be positive")
  }

  var_names <- colnames(x)
  x <- as.matrix(x)
  p <- ncol(x)
  n <- nrow(x)

  class_names <- levels(y)
  n_classes <- sapply(class_names, function(m) sum(y == m))
  k_class <- length(class_names)
  n_classes_max <- max(n_classes)
  n_needed <- n_classes_max - n_classes
  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])
  x_syn_list <- list()

  for (i in 1:k_class) {
    counter <- 0
    NN_main2main <- FNN::get.knnx(data = x_classes[[i]], query = x_classes[[i]], k = k + 1)$nn.index[,-1]
    x_main <- x_classes[[i]]

    x_syn_list[[i]] <- matrix(data = NA, nrow = 0, ncol = p)
    while (TRUE) {
      if (counter == n_needed[i]) {
        break
      }
      counter <- counter + 1

      i_sample <- sample(1:n_classes[i], size = 1)
      x_main_selected <- x_main[i_sample,,drop = FALSE]
      x_target <- x_main[sample(NN_main2main[i_sample,], size = 1),,drop = FALSE]
      r <- runif(1)

      x_syn_list[[i]] <- rbind(x_syn_list[[i]], x_main_selected + r*(x_target - x_main_selected))
    }
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
