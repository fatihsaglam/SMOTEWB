#' @title  Adaptive Synthetic Sampling
#'
#' @description Generates synthetic data for minority class to balance imbalanced
#' datasets using ADASYN.
#'
#' @param x feature matrix or data.frame.
#' @param y a factor class variable with two classes.
#' @param k number of neighbors. Default is 5.
#'
#' @details
#' Adaptive Synthetic Sampling (ADASYN) is an extension of the Synthetic Minority Over-sampling Technique
#' (SMOTE) algorithm, which is used to generate synthetic examples for the
#' minority class (He et al., 2008). In contrast to SMOTE, ADASYN adaptively generates synthetic
#' examples by focusing on the minority class examples that are harder to
#' learn, meaning those examples that are closer to the decision boundary.
#'
#' Note: Much faster than \code{smotefamily::ADAS()}.
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
#' @importFrom  RANN nn2
#' @importFrom  stats rnorm
#' @importFrom  stats sd
#'
#' @references
#' He, H., Bai, Y., Garcia, E. A., & Li, S. (2008, June). ADASYN: Adaptive
#' synthetic sampling approach for imbalanced learning. In 2008 IEEE
#' international joint conference on neural networks (IEEE world congress on
#' computational intelligence) (pp. 1322-1328). IEEE.
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
#' m <- ADASYN(x = x, y = y, k = 3)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname ADASYN
#' @export

ADASYN <- function(x, y, k = 5) {

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

  class_names <- as.character(unique(y))
  n_classes <- sapply(class_names, function(m) sum(y == m))
  k_class <- length(class_names)
  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])

  n_needed <- max(n_classes) - n_classes
  x_syn_list <- list()

  for (j in 1:k_class) {
    x_syn_list[[j]] <- matrix(nrow = 0, ncol = p)
    if (n_needed[j] == 0) {
      next
    }

    n_main <- n_classes[j]
    n_other <- sum(n_classes[-j])

    nn_main2all <- RANN::nn2(data = x, query = x_classes[[j]], k = k + 1)$nn.idx[,-1]

    count_main <- rowSums(matrix(y[nn_main2all] == class_names[j], nrow = nrow(nn_main2all), ncol = ncol(nn_main2all)))
    count_other <- k - count_main
    nn_main2all_classcounts <- cbind(
      count_main,
      count_other
    )
    nn_main2main <- RANN::nn2(data = x_classes[[j]], query = x_classes[[j]], k = k + 1)$nn.idx[,-1]

    if (sum(nn_main2all_classcounts[,2]) == 0) {
      w <- rep(1/n_needed[j], n_main)
    } else {
      w <- nn_main2all_classcounts[,2]/sum(nn_main2all_classcounts[,2])
    }
    C <- round(n_needed[j]*w)

    for (i in 1:n_main) {
      if (C[i] == 0) {
        next
      }
      NN_i <- nn_main2main[i,]
      i_k <- sample(1:k, C[i], replace = TRUE)
      lambda <- runif(C[i])
      kk <- x_classes[[j]][NN_i,,drop = FALSE]
      kk <- kk[i_k,]
      x_main_i_temp <- x_classes[[j]][rep(i, C[i]),,drop = FALSE]
      x_syn_step <- x_main_i_temp + (kk - x_main_i_temp)*lambda
      x_syn_list[[j]] <- rbind(x_syn_list[[j]], x_syn_step)
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
    y_syn = y_syn,
    C = C
  ))
}
