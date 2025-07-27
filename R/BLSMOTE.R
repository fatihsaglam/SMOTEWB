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
#' @param ovRate Oversampling rate multiplied by the difference between maximum
#' and other of class sizes. Default is 1 meaning full balance.
#'
#' @details
#' BLSMOTE works by focusing on the instances that are near the decision
#' boundary between the minority and majority classes, known as borderline
#' instances. These instances are more informative and potentially more
#' challenging for classification, and thus generating synthetic samples in
#' their vicinity can be more effective than generating them randomly.
#'
#' Can work with classes more than 2.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'  \item{x_syn}{Generated synthetic data.}
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

BLSMOTE <- function(x, y, k1 = 5, k2 = 5, type = "type1", ovRate = 1) {

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

  x_syn_list <- list()

  for (j in 1:k_class) {
    x_syn_list[[j]] <- matrix(nrow = 0, ncol = p)
    if (n_needed[j] == 0) {
      next
    }

    n_main <- n_classes[j]
    n_other <- sum(n_classes[-j])
    x_main <- x_classes[[j]]

    i_danger <- c()

    while(length(i_danger) < 2) {
      nn_main2all <- RANN::nn2(data = x, query = x_main, k = k2 + 1)$nn.idx[,-1]

      count_main <- rowSums(matrix(y[nn_main2all] == class_names[j], nrow = nrow(nn_main2all), ncol = ncol(nn_main2all)))
      count_other <- k2 - count_main
      nn_main2all_classcounts <- cbind(
        count_main,
        count_other
      )
      safe_levels <- nn_main2all_classcounts[,1]/k2

      i_danger <- which(safe_levels <= 0.5 & safe_levels > 0)
      i_safe <- which(safe_levels > 0.5)
      i_outcast <- which(safe_levels == 0)
      n_danger <- length(i_danger)

      if (n_danger < 2) {
        k2 <- k2 + 1
      }
    }

    x_main_danger <- x_main[i_danger,,drop = FALSE]
    k1 <- pmin(k1, n_danger - 1)

    if (type == "type1") {
      if (k1 >= n_danger) {
        stop("k1 exceeded the number of dangerous samples.")
      }
      nn_danger2danger <- FNN::knnx.index(data = x_main_danger, query = x_main_danger, k = k1 + 1)[,-1,drop = FALSE]

      while (nrow(x_syn_list[[j]]) < n_needed[j]) {
        i_main <- sample(1:nrow(x_main_danger), 1)
        i_k <- sample(1:k1, 1, replace = TRUE)
        r <- runif(1)
        x_main_danger_i <- x_main_danger[i_main,]
        x_main_danger_neighbour <- x_main_danger[nn_danger2danger[i_main, i_k],, drop = FALSE]
        x_syn_step <- x_main_danger_i + (x_main_danger_neighbour - x_main_danger_i)*r
        x_syn_list[[j]] <- rbind(x_syn_list[[j]], x_syn_step)
      }
    }

    if (type == "type2") {
      nn_danger2all <- RANN::nn2(data = x, query = x_main_danger, k = k1 + 1)$nn.idx[,-1,drop = FALSE]
      while (nrow(x_syn_list[[j]]) < n_needed[j]) {
        i_main <- sample(1:nrow(x_main_danger), 1)
        i_k <- sample(1:k1, 1, replace = TRUE)
        x_main_danger_i <- x_main_danger[i_main,]

        i_nn_danger2all_i <- nn_danger2all[i_main, i_k]
        r <- ifelse(y[i_nn_danger2all_i] == class_names[j], runif(1), runif(1, 0, 0.5))
        x_main_danger_neighbour <- x[i_nn_danger2all_i, , drop = FALSE]
        x_syn_step <- x_main_danger_i + (x_main_danger_neighbour - x_main_danger_i)*r
        x_syn_list[[j]] <- rbind(x_syn_list[[j]], x_syn_step)
      }
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




