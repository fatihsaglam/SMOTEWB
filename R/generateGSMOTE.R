#' @title Data generation for Geometric SMOTE (GSMOTE)
#'
#' @description Data generation for Geometric SMOTE (GSMOTE)
#'
#' @param x_pos positive class feature matrix.
#' @param x_neg negative class feature matrix.
#' @param n_syn number of synthetic samples to generate.
#' @param k number of neighbors.
#' @param alpha_sel selection method. Can be "minority", "majority" or "combined".
#' @param alpha_trunc truncation factor. A numeric value in \eqn{[-1,1]}.
#' @param alpha_def deformation factor. A numeric value in \eqn{[0,1]}.
#' @param class_pos label of the positive class.
#' @param class_names class labels.
#'
#' @details
#' Data generation for Geometric SMOTE (GSMOTE)
#'
#' @return a vector of case weights.
#'  \item{x_syn}{last weight vector of boosting process}
#'  \item{y_syn}{Generated synthetic data labels.}
#'  \item{C}{Number of synthetic samples for each positive class samples.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @noRd

generateGSMOTE <- function(
    x_pos,
    x_neg,
    n_syn,
    k,
    alpha_sel,
    alpha_trunc,
    alpha_def,
    class_pos,
    class_names) {

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

  C <- rep(ceiling(n_syn / n_pos) - 1, n_pos)

  n_diff <- (n_syn - sum(C))

  ii <- sample(1:n_pos, size = abs(n_diff))
  C[ii] <- C[ii] + n_diff / abs(n_diff)

  m_pos2neg <- FNN::get.knnx(data = x_neg,
                             query = x_pos,
                             k = 1 + 1)
  NN_pos2neg <- m_pos2neg$nn.index[, -1, drop = FALSE]
  D_pos2neg <- m_pos2neg$nn.dist[, -1, drop = FALSE]

  m_pos2pos <- FNN::get.knnx(data = x_pos,
                             query = x_pos,
                             k = k + 1)
  NN_pos2pos <- m_pos2pos$nn.index[, -1, drop = FALSE]
  D_pos2pos <- m_pos2pos$nn.dist[, -1, drop = FALSE]

  x_syn <- matrix(nrow = 0, ncol = p)

  for (i in 1:n_pos) {
    x_center <- x_pos[i, , drop = FALSE]

    for (j in 1:C[i]) {

      ### Surface ###
      if (alpha_sel == "minority") {
        i_selected_neighbor_pos <- sample(1:k, 1)
        x_surface <-
          x_pos[NN_pos2pos[i, i_selected_neighbor_pos], , drop = FALSE]
      } else if (alpha_sel == "majority") {
        i_selected_neighbor_neg <- sample(1:1, 1)
        x_surface <-
          x_neg[NN_pos2neg[i, i_selected_neighbor_neg], , drop = FALSE]
      } else {
        i_selected_neighbor_pos <- sample(1:k, 1)
        i_selected_neighbor_neg <- sample(1:1, 1)

        if (D_pos2pos[i, i_selected_neighbor_pos] < D_pos2neg[i, i_selected_neighbor_neg]) {
          x_surface <-
            x_pos[NN_pos2pos[i, i_selected_neighbor_pos], , drop = FALSE]
        } else {
          x_surface <-
            x_neg[NN_pos2neg[i, i_selected_neighbor_neg], , drop = FALSE]
        }
      }
      ###############

      ### Hyperball ###
      v_normal <- rnorm(p)
      e_sphere <- v_normal / norm(v_normal, type = "2")
      r <- runif(1)
      x_gen <- matrix(r ^ (1 / p) * e_sphere, nrow = 1)
      #################

      ### Vectors ###
      R <- norm(x_surface - x_center, type = "2")

      if (R == 0) {
        x_syn <- rbind(x_syn, x_center)
        next
      }

      e_parallel <- (x_surface - x_center) / R
      x_parallel_proj <- c(tcrossprod(x_gen, e_parallel))
      x_parallel <- x_parallel_proj * e_parallel
      x_perpendicular <- x_gen - x_parallel
      ###############

      ### Truncate ###
      if (abs(alpha_trunc - x_parallel_proj) > 1) {
        x_gen <- x_gen - 2 * x_parallel
      }
      ################

      ### Deform ###
      x_gen <- x_gen - alpha_def * x_perpendicular
      ##############

      ### Translate ###
      x_syn_step <- x_center + R * x_gen
      #################

      x_syn <- rbind(x_syn, x_syn_step)
    }
  }

  return(list(
    x_syn = x_syn,
    y_syn = factor(rep(class_pos, nrow(x_syn)), class_names),
    C = C
  ))
}
