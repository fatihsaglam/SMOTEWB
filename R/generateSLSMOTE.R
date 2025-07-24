#' @title Data generation for Safe-level SMOTE
#'
#' @description Data generation for Safe-level SMOTE
#'
#' @param x_pos positive class feature matrix.
#' @param x_neg negative class feature matrix.
#' @param n_syn number of synthetic samples to generate.
#' @param k1 number of neighbors to link.
#' @param k2 number of neighbors to determine safe levels.
#' @param class_pos label of the positive class.
#' @param class_names class labels.
#'
#' @details
#' Data generation for Safe-level SMOTE.
#'
#' @return a vector of case weights.
#'  \item{x_syn}{last weight vector of boosting process}
#'  \item{y_syn}{Generated synthetic data labels.}
#'  \item{C}{Number of synthetic samples for each positive class samples.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @noRd

generateSLSMOTE <- function(
    x_pos,
    x_neg,
    n_syn,
    k1,
    k2,
    class_pos,
    class_names
) {

  if (n_syn == 0) {
    return(list(
      x_syn = matrix(data = NA, nrow = 0, ncol = ncol(x_pos)),
      y_syn = factor(c(), levels = class_names),
      C = list()
    ))
  }

  class_neg <- paste0(class_names[class_names != class_pos], collapse = "-")
  n_pos <- nrow(x_pos)
  n_neg <- nrow(x_neg)
  p <- ncol(x_pos)

  x <- rbind(x_pos, x_neg)

  nn_pos2all <- FNN::knnx.index(data = x, query = x_pos, k = k2 + 1)[,-1]
  nn_pos2pos <- FNN::knnx.index(data = x_pos, query = x_pos, k = k1 + 1)[,-1]
  nn_pos2all_classcounts <- cbind(
    rowSums(nn_pos2all <= n_pos),
    rowSums(nn_pos2all > n_pos)
  )

  safe_levels <- nn_pos2all_classcounts[,1]

  i_safe <- which(safe_levels > 0)

  x_pos_safe <- x_pos[i_safe,,drop = FALSE]

  n_safe <- nrow(x_pos_safe)

  C <- rep(0, n_pos)
  C[i_safe] <- rep(ceiling(n_syn/n_safe) - 1, n_safe)

  n_diff <- (n_syn - sum(C))
  ii <- sample(1:n_safe, size = abs(n_diff))
  C[i_safe][ii] <- C[i_safe][ii]  + n_diff/abs(n_diff)

  x_syn <- matrix(nrow = 0, ncol = p)

  for (i in 1:n_pos) {
    if (safe_levels[i] > 0 & C[i] > 0) {
      i_k <- sample(1:k1, C[i], replace = TRUE)
      i_nn_pos2pos <- nn_pos2pos[i, i_k]
      k_safe_levels <- safe_levels[i_nn_pos2pos]

      r <- rep(0, C[i])
      for (j in 1:C[i]) {
        if (k_safe_levels[j] == 0) {
          r[j] <- 0
        } else if (k_safe_levels[j] == safe_levels[i]) {
          r[j] <- runif(1, 0, 1)
        } else if (k_safe_levels[j] < safe_levels[i]) {
          r[j] <- runif(1, 0, k_safe_levels[j]/safe_levels[i])
        } else {
          r[j] <- runif(1, 1 - safe_levels[i]/k_safe_levels[j], 1)
        }
      }

      x_pos_step <- x_pos[rep(i, C[i]),,drop = FALSE]
      x_pos_k <- x_pos[i_nn_pos2pos,,drop = FALSE]
      x_syn_step <- x_pos_step + (x_pos_k - x_pos_step)*r
      x_syn <- rbind(x_syn, x_syn_step)
    }
  }

  return(list(
    x_syn = x_syn,
    y_syn = factor(rep(class_pos, nrow(x_syn)), class_names),
    C = C
  ))
}
