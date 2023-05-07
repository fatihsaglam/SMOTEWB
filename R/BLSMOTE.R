#' @title  Borderline Synthetic Minority Oversampling Technique
#'
#' @description \code{BLSMOTE()} applies BLSMOTE (Borderline-SMOTE) which is a
#' variation of the SMOTE algorithm that generates synthetic samples only in the
#' vicinity of the borderline instances in imbalanced datasets.
#'
#' @param x feature matrix or data.frame.
#' @param y a factor class variable with two classes.
#' @param k1 number of neighbors to link. Default is 5.
#' @param k2 number of neighbors to determine safe levels. Default is 5.
#' @param type "type1" or "type2". Default is "type1".
#'
#' @details
#' BLSMOTE works by focusing on the instances that are near the decision
#' boundary between the minority and majority classes, known as borderline
#' instances. These instances are more informative and potentially more
#' challenging for classification, and thus generating synthetic samples in
#' their vicinity can be more effective than generating them randomly.
#'
#' Note: Much faster than \code{smotefamily::BLSMOTE()}.
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
#' Han, H., Wang, W. Y., & Mao, B. H. (2005). Borderline-SMOTE: a new
#' over-sampling method in imbalanced data sets learning. In Advances in
#' Intelligent Computing: International Conference on Intelligent Computing,
#' ICIC 2005, Hefei, China, August 23-26, 2005, Proceedings, Part I 1
#' (pp. 878-887). Springer Berlin Heidelberg.
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
#' m <- BLSMOTE(x = x, y = y, k1 = 5, k2 = 5)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname BLSMOTE
#' @export

BLSMOTE <- function(x, y, k1 = 5, k2 = 5, type = "type1") {

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
  n <- length(y)
  p <- ncol(x)

  class_names <- as.character(unique(y))
  class_pos <- names(which.min(table(y)))
  class_neg <- class_names[class_names != class_pos]

  x_pos <- x[y == class_pos,,drop = FALSE]
  x_neg <- x[y == class_neg,,drop = FALSE]

  n_pos <- nrow(x_pos)
  n_neg <- nrow(x_neg)

  x <- rbind(x_pos, x_neg)

  i_danger <- c()

  while (length(i_danger) < 2) {
    nn_pos2all <- FNN::knnx.index(data = x, query = x_pos, k = k2 + 1)[,-1]

    nn_pos2all_classcounts <- cbind(
      rowSums(nn_pos2all <= n_pos),
      rowSums(nn_pos2all > n_pos)
    )
    safe_levels <- nn_pos2all_classcounts[,1]/k2

    i_danger <- which(safe_levels <= 0.5 & safe_levels > 0)
    i_safe <- which(safe_levels > 0.5)
    i_outcast <- which(safe_levels == 0)

    if (length(i_danger) < 2) {
      k2 <- k2 + 1
    }
  }
  x_pos_danger <- x_pos[i_danger,,drop = FALSE]
  # x_pos_safe <- x_pos[i_safe,,drop = FALSE]
  # x_pos_outcast <- x_pos[i_outcast,,drop = FALSE]

  n_danger <- nrow(x_pos_danger)

  k1 <- pmin(k1, n_danger - 1)

  nn_danger2danger <- FNN::knnx.index(data = x_pos_danger, query = x_pos_danger, k = k1 + 1)[,-1,drop = FALSE]

  n_syn <- (n_neg - n_pos)
  C <- rep(ceiling(n_syn/n_danger) - 1, n_danger)

  n_diff <- (n_syn - sum(C))
  ii <- sample(1:n_danger, size = abs(n_diff))
  C[ii] <- C[ii] + n_diff/abs(n_diff)

  x_syn <- matrix(nrow = 0, ncol = p)

  if (type == "type1") {
    if (k1 >= n_danger) {
      stop("k1 exceeded the number of dangerous samples.")
    }
    for (i in 1:n_danger) {
      i_k <- sample(1:k1, C[i], replace = TRUE)
      r <- runif(C[i])
      x_pos_danger_i <- x_pos_danger[rep(i, C[i]),]
      x_pos_danger_neighbour <- x_pos_danger[nn_danger2danger[i, i_k],, drop = FALSE]
      x_syn_step <- x_pos_danger_i + (x_pos_danger_neighbour - x_pos_danger_i)*r
      x_syn <- rbind(x_syn, x_syn_step)
    }
  }

  if (type == "type2") {
    for (i in 1:n_danger) {
      CC <- C[i]
      i <- i_danger[i]
      i_k <- sample(1:k2, CC, replace = TRUE)
      i_nn_pos2all_i <- nn_pos2all[i, i_k]
      r <- rep(0, CC)
      if (sum(i_nn_pos2all_i <= n_pos) > 0) {
        r[i_nn_pos2all_i < n_pos] <- runif(sum(i_nn_pos2all_i <= n_pos))
      }
      if (sum(i_nn_pos2all_i > n_pos) > 0) {
        r[i_nn_pos2all_i >= n_pos] <- runif(sum(i_nn_pos2all_i > n_pos), 0, 0.5)
      }
      x_pos_i <- x_pos[rep(i, CC),]
      x_pos_neighbour <- x[i_nn_pos2all_i,, drop = FALSE]

      x_syn_step <- x_pos_i + (x_pos_neighbour - x_pos_i)*r
      x_syn <- rbind(x_syn, x_syn_step)
    }
  }

  x_new <- rbind(
    x_syn,
    x_pos,
    x_neg
  )
  y_new <- c(
    rep(class_pos, n_syn + n_pos),
    rep(class_neg, n_neg)
  )
  y_new <- factor(y_new, levels = levels(y), labels = levels(y))
  colnames(x_new) <- var_names

  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_new[1:n_syn,,drop = FALSE],
    C = C
  ))
}
