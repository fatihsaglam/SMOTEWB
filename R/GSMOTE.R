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
#' @param ovRate Oversampling rate multiplied by the difference between maximum
#' and other of class sizes. Default is 1 meaning full balance.
#'
#' @details
#' GSMOTE (Douzas & Bacao, 2019) is an oversampling method which creates synthetic
#' samples geometrically around selected minority samples. Details are in the
#' paper (Douzas & Bacao, 2019).
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
           alpha_def = 0.5,
           ovRate = 1) {

    match.arg(alpha_sel, c("minority", "majority", "combined"))

    if (alpha_trunc < -1 | alpha_trunc > 1) {
      stop("alpha_trunc must be between [-1,1].")
    }
    if (alpha_def < 0 | alpha_def > 1) {
      stop("alpha_def must be between [0,1].")
    }
    if (!is.data.frame(x) & !is.matrix(x)) {
      stop("x must be a matrix or dataframe.")
    }

    if (is.data.frame(x)) {
      x <- as.matrix(x)
    }

    if (!is.factor(y)) {
      stop("y must be a factor.")
    }
    if (!is.numeric(k)) {
      stop("k must be numeric.")
    }
    if (k < 1) {
      stop("k must be positive.")
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

    x_syn <- matrix(NA, nrow = 0, ncol = p)
    y_syn <- factor(c(), levels = class_names)
    C <- list()

    for (j in 1:k_class) {
      m_syn <- generateGSMOTE(
        x_pos = x_classes[[j]],
        x_neg = do.call(rbind, x_classes[-j]),
        n_syn = n_needed[j],
        k = k,
        alpha_sel = alpha_sel,
        alpha_trunc = alpha_trunc,
        alpha_def = alpha_def,
        class_pos = class_names[j],
        class_names = class_names
      )
      x_syn <- rbind(x_syn, m_syn$x_syn)
      y_syn <- c(y_syn, m_syn$y_syn)


      C[[j]] <- m_syn$C
    }

    x_new <- rbind(x, x_syn)
    y_new <- c(y, y_syn)
    colnames(x_new) <- var_names
    names(C) <- class_names

    return(list(
      x_new = x_new,
      y_new = y_new,
      x_syn = x_syn,
      C = C
    ))
  }
