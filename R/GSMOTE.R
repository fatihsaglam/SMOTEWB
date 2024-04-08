#' @title  Geometric Synthetic Minority Oversamplnig Technique (GSMOTE)
#'
#' @description Resampling with GSMOTE.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#' @param k number of neighbors. Default is 5.
#' @param alpha_sel selection method. Can be "minority", "majority" or "combined".
#' Default is "combined".
#' @param alpha_trunc truncation factor. A numeric value in \eqn{[-1,1]}.
#' Default is 0.5.
#' @param alpha_def deformation factor. A numeric value in \eqn{[0,1]}.
#' Default is 0.5
#'
#' @details
#' GSMOTE (Douzas & Bacao, 2019) is an oversampling method which creates synthetic
#' samples geometrically around selected minority samples. Details are in the
#' paper (Douzas & Bacao, 2019).
#'
#'
#' NOTE: Can not work with classes more than 2. Only numerical variables are
#' allowed.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'  \item{x_syn}{Generated synthetic feature data.}
#'  \item{y_syn}{Generated synthetic label data.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @importFrom  FNN get.knnx
#' @importFrom  stats runif
#' @importFrom  stats sd
#'
#' @references
#' Douzas, G., & Bacao, F. (2019). Geometric SMOTE a geometrically enhanced
#' drop-in replacement for SMOTE. Information sciences, 501, 118-135.
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
#' m <- GSMOTE(x = x, y = y, k = 7)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname GSMOTE
#' @export


GSMOTE <-
  function(x,
           y,
           k = 5,
           alpha_sel = "combined",
           alpha_trunc = 0.5,
           alpha_def = 0.5) {

    match.arg(alpha_sel, c("minority", "majority", "combined"))

    if (alpha_trunc < -1 | alpha_trunc > 1) {
      stop("alpha_trunc must be between [-1,1]")
    }
    if (alpha_def < 0 | alpha_def > 1) {
      stop("alpha_def must be between [0,1]")
    }
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

    x_pos <- x[y == class_pos, , drop = FALSE]
    x_neg <- x[y == class_neg, , drop = FALSE]

    n_pos <- nrow(x_pos)
    n_neg <- nrow(x_neg)

    x <- rbind(x_pos, x_neg)

    n_syn <- (n_neg - n_pos)
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

    x_new <- rbind(x_syn,
                   x_pos,
                   x_neg)
    y_new <- c(rep(class_pos, n_syn + n_pos),
               rep(class_neg, n_neg))
    y_new <- factor(y_new, levels = levels(y), labels = levels(y))
    colnames(x_new) <- var_names

    return(list(
      x_new = x_new,
      y_new = y_new,
      x_syn = x_new[1:n_syn, , drop = FALSE],
      C = C
    ))
  }
