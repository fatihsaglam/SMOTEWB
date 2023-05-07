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
  class_pos <- names(which.min(table(y)))
  class_neg <- class_names[class_names != class_pos]

  x_pos <- x[y == class_pos,,drop = FALSE]
  x_neg <- x[y == class_neg,,drop = FALSE]

  n_pos <- nrow(x_pos)
  n_neg <- nrow(x_neg)

  x <- rbind(x_pos, x_neg)

  k <- min(k, n_pos - 1)
  nn_pos2all <- RANN::nn2(data = x, query = x_pos, k = k + 1)$nn.idx[,-1]
  nn_pos2all_classcounts <- cbind(
    rowSums(nn_pos2all <= n_pos),
    rowSums(nn_pos2all > n_pos)
  )
  nn_pos2pos <- RANN::nn2(data = x_pos, query = x_pos, k = k + 1)$nn.idx[,-1]

  n_syn <- (n_neg - n_pos)
  if (sum(nn_pos2all_classcounts[,2]) == 0) {
    w <- rep(1/n_syn, n_syn)
  } else {
    w <- nn_pos2all_classcounts[,2]/sum(nn_pos2all_classcounts[,2])
  }
  C <- round(n_syn*w)

  x_syn <- matrix(nrow = 0, ncol = p)
  for (i in 1:n_pos) {
    if (C[i] == 0) {
      next
    }
    NN_i <- nn_pos2pos[i,]
    i_k <- sample(1:k, C[i], replace = TRUE)
    lambda <- runif(C[i])
    kk <- x_pos[NN_i,,drop = FALSE]
    kk <- kk[i_k,]
    x_pos_i_temp <- x_pos[rep(i, C[i]),,drop = FALSE]
    x_syn_step <- x_pos_i_temp + (kk - x_pos_i_temp)*lambda
    x_syn <- rbind(x_syn, x_syn_step)
  }

  x_new <- rbind(
    x_syn,
    x_pos,
    x_neg
  )
  y_new <- c(
    rep(class_pos, nrow(x_syn) + n_pos),
    rep(class_neg, n_neg)
  )
  y_new <- factor(y_new, levels = levels(y), labels = levels(y))
  colnames(x_new) <- var_names

  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_new[1:nrow(x_syn),,drop = FALSE],
    C = C
  ))
}
